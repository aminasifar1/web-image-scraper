#!/usr/bin/env python3
"""
wayback_scraper.py — Scraper de Wayback Machine con CDX API + Scrapy

Flujo:
  1. CDX API  →  obtener N snapshots por año con paginación y backoff
  2. Scrapy   →  descargar cada snapshot con id_ (HTML original)
  3. Output   →  HTML + imágenes + metadata por dominio/año/timestamp

Mejoras:
  - Paginación con resumeKey para sitios grandes
  - Exponential backoff en 429/5xx en CDX
  - URL normalization
  - Sufijo id_ garantizado en URLs de descarga HTML
  - Descarga de imágenes desde Wayback usando el timestamp del snapshot
  - Extracción de URL original limpia desde URLs de Wayback reescritas
  - Caché de snapshots ya procesados para reanudar ejecuciones
  - Evita sobrescribir imágenes con el mismo nombre
  - Valida que lo descargado sea realmente una imagen
"""

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode

import requests
import scrapy
from scrapy.crawler import CrawlerProcess


logger = logging.getLogger("wayback_scraper")


# ─────────────────────────────────────────────
#  Helpers de URL
# ─────────────────────────────────────────────

_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "ref", "referrer", "source", "fbclid", "gclid", "mc_eid",
}

_TS_RE = re.compile(r"^\d{14}[a-z_]*/")


def normalize_url(raw: str) -> str:
    """
    Normaliza una URL antes de enviarla a la CDX API.

    - Añade esquema si falta
    - Elimina parámetros de tracking
    - Conserva path y query relevantes
    - No fuerza trailing slash
    """
    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)

    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs_clean = {k: v for k, v in qs.items() if k.lower() not in _STRIP_PARAMS}
    clean_query = urlencode(qs_clean, doseq=True)

    return urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        clean_query,
        "",
    ))


def extract_original_from_wayback(wayback_url: str) -> str:
    """
    Extrae la URL original de una URL de Wayback Machine reescrita.

    Soporta formatos:
      https://web.archive.org/web/20200101120000/https://example.com/
      https://web.archive.org/web/20200101120000id_/https://example.com/
      https://web.archive.org/web/20200101120000im_/https://example.com/image.jpg
    """
    if "web.archive.org/web/" not in wayback_url:
        return wayback_url

    after_web = wayback_url.split("/web/", 1)[1]
    original = _TS_RE.sub("", after_web)

    if original.startswith(("http://", "https://")):
        return original

    return wayback_url


def build_wayback_url(timestamp: str, original_url: str) -> str:
    """
    Construye URL de Wayback con sufijo id_ para obtener HTML original
    sin reescrituras de Wayback.
    """
    return f"https://web.archive.org/web/{timestamp}id_/{original_url}"


def build_wayback_resource_url(timestamp: str, resource_url: str) -> str:
    """
    Construye URL de Wayback para recursos estáticos, como imágenes.

    Usa el mismo timestamp del snapshot para intentar descargar la versión
    histórica del recurso, no la versión actual del sitio original.
    """
    return f"https://web.archive.org/web/{timestamp}id_/{resource_url}"


# ─────────────────────────────────────────────
#  PASO 1: CDX API — obtener snapshots
# ─────────────────────────────────────────────

CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"

CDX_MIN_INTERVAL = 1.0
_last_cdx_call = 0.0


def _cdx_rate_wait():
    """
    Espera el tiempo necesario para respetar el rate limit de CDX.
    """
    global _last_cdx_call

    elapsed = time.monotonic() - _last_cdx_call
    wait = CDX_MIN_INTERVAL - elapsed

    if wait > 0:
        time.sleep(wait)

    _last_cdx_call = time.monotonic()


def _cdx_get_with_backoff(params: dict, max_retries: int = 5) -> list | None:
    """
    Hace una petición a CDX con exponential backoff en 429/5xx.

    Returns:
        Lista de filas JSON, o None si falla definitivamente.
    """
    for attempt in range(max_retries):
        _cdx_rate_wait()

        try:
            resp = requests.get(CDX_ENDPOINT, params=params, timeout=30)

            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"CDX 429 rate-limited. Reintento en {wait:.1f}s...")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"CDX {resp.status_code}. Reintento en {wait:.1f}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            if not resp.text.strip():
                return []

            return resp.json()

        except requests.exceptions.Timeout:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"CDX timeout. Reintento en {wait:.1f}s...")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"CDX error inesperado: {e}")
            return None

    logger.error(f"CDX falló definitivamente tras {max_retries} intentos.")
    return None


def get_snapshots_for_year(
    url: str,
    year: int,
    limit: int = 5,
    match_type: str = "prefix",
) -> list[dict]:
    """
    Obtiene hasta `limit` snapshots de un año usando paginación con resumeKey.

    Nota:
      - collapse=digest evita snapshots duplicados por contenido idéntico.
      - Si quieres una URL única por cada página distinta, puedes cambiarlo
        por collapse=urlkey.
    """
    url = normalize_url(url)

    base_params = {
        "url":           url,
        "output":        "json",
        "fl":            "timestamp,original,statuscode,mimetype",
        "from":          f"{year}0101000000",
        "to":            f"{year}1231235959",
        "matchType":     match_type,
        "filter":        ["statuscode:200", "mimetype:text/html"],
        "collapse":      "digest",
        "showResumeKey": "true",
        "limit":         min(limit, 500),
    }

    collected: list[dict] = []
    page = 0

    while len(collected) < limit:
        rows = _cdx_get_with_backoff(base_params)

        if rows is None:
            logger.error(f"  {year}: fallo en CDX, saltando año.")
            break

        if not rows:
            break

        headers = rows[0]

        resume_key = None
        data_rows = rows[1:]

        if data_rows and len(data_rows[-1]) == 1:
            resume_key = data_rows[-1][0]
            data_rows = data_rows[:-1]

        for row in data_rows:
            if len(collected) >= limit:
                break

            entry = dict(zip(headers, row))

            ts = entry.get("timestamp", "")
            original = entry.get("original", "")

            if not ts or not original:
                continue

            collected.append({
                "timestamp":   ts,
                "year":        year,
                "original":    original,
                "wayback_url": build_wayback_url(ts, original),
            })

        page += 1

        logger.debug(
            f"  {year} página {page}: {len(data_rows)} filas, "
            f"acumulado {len(collected)}/{limit}"
        )

        if resume_key is None or len(collected) >= limit:
            break

        base_params["resumeKey"] = resume_key

    logger.info(f"  {year}: {len(collected)} snapshots")
    return collected


def collect_all_snapshots(
    url: str,
    years: list[int],
    limit_per_year: int,
    match_type: str = "prefix",
) -> list[dict]:
    """
    Recorre todos los años y acumula snapshots respetando el rate limit.
    """
    all_snapshots = []

    for year in years:
        snaps = get_snapshots_for_year(url, year, limit_per_year, match_type)
        all_snapshots.extend(snaps)

        time.sleep(0.5)

    logger.info(f"Total snapshots a descargar: {len(all_snapshots)}")
    return all_snapshots


# ─────────────────────────────────────────────
#  PASO 2: Scrapy Spider — descargar snapshots
# ─────────────────────────────────────────────

class WaybackSpider(scrapy.Spider):
    """
    Spider que descarga una lista explícita de URLs de Wayback Machine.
    Las URLs vienen ya calculadas desde la CDX API.
    """

    name = "wayback_spider"
    allowed_domains = ["web.archive.org"]

    def __init__(
        self,
        snapshots: list[dict],
        output_dir: str,
        skip_existing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.snapshots = snapshots
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_dir(self, snap: dict) -> Path:
        domain = urlparse(snap["original"]).netloc.replace(":", "_")
        return self.output_dir / domain / str(snap["year"]) / snap["timestamp"]

    def start_requests(self):
        for snap in self.snapshots:
            save_dir = self._save_dir(snap)

            if self.skip_existing and (save_dir / "meta.json").exists():
                logger.debug(f"Saltando snapshot ya procesado: {snap['wayback_url']}")
                continue

            yield scrapy.Request(
                url=snap["wayback_url"],
                callback=self.parse_snapshot,
                errback=self.handle_error,
                meta={
                    "snapshot": snap,
                    "dont_redirect": False,
                    "handle_httpstatus_list": [301, 302, 403, 404],
                },
            )

    def parse_snapshot(self, response):
        snap = response.meta["snapshot"]

        ts = snap["timestamp"]
        year = snap["year"]
        original = snap["original"]

        save_dir = self._save_dir(snap)
        save_dir.mkdir(parents=True, exist_ok=True)

        images = self._extract_images(response, original)

        meta = {
            "timestamp":    ts,
            "year":         year,
            "original_url": original,
            "wayback_url":  response.url,
            "final_url":    extract_original_from_wayback(response.url),
            "status":       response.status,
            "images_count": len(images),
            "downloaded":   0,
            "failed":       0,
            "scraped_at":   datetime.utcnow().isoformat(),
        }

        if images:
            images_dir = save_dir / "images"
            downloaded, failed = self._download_images(images, images_dir, ts)
            meta["downloaded"] = downloaded
            meta["failed"] = failed

        (save_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        if response.status == 200 and len(response.text) < 10_000_000:
            (save_dir / "page.html").write_text(
                response.text,
                encoding="utf-8"
            )

        logger.info(
            f"[{year}/{ts}] {original} → {response.status} | "
            f"{len(images)} imágenes | "
            f"{meta['downloaded']} descargadas, {meta['failed']} fallos"
        )

    def handle_error(self, failure):
        snap = failure.request.meta.get("snapshot", {})

        logger.warning(
            f"Error en {snap.get('wayback_url', '?')}: {failure.value}"
        )

    # ─────────────────────────────────────────
    #  Extracción de imágenes
    # ─────────────────────────────────────────

    def _extract_images(self, response, base_url: str) -> set:
        images = set()

        # <img src / data-src / data-lazy-src / data-original / srcset>
        for img in response.css("img"):
            for attr in ("@src", "@data-src", "@data-lazy-src", "@data-original"):
                src = img.xpath(attr).get()

                if src:
                    images.add(self._abs(src, base_url))

            srcset = img.xpath("@srcset").get()

            if srcset:
                images |= self._parse_srcset(srcset, base_url)

        # <picture><source srcset>
        for source in response.css("picture source"):
            srcset = source.xpath("@srcset").get()

            if srcset:
                images |= self._parse_srcset(srcset, base_url)

        # <amp-img>
        for amp in response.css("amp-img"):
            src = amp.xpath("@src").get()

            if src:
                images.add(self._abs(src, base_url))

        # CSS background-image en style inline
        for el in response.css("[style*='background']"):
            style = el.xpath("@style").get()

            if style:
                images |= self._extract_css_urls(style, base_url)

        # CSS background-image en bloques <style>
        for style_tag in response.css("style"):
            css_text = style_tag.xpath("text()").get()

            if css_text:
                images |= self._extract_css_urls(css_text, base_url)

        return {
            u for u in images
            if u and not u.startswith(("data:", "blob:", "javascript:"))
        }

    def _abs(self, url: str, base: str) -> str:
        """
        Convierte URL a absoluta usando la URL original del sitio.

        Si la URL ya viene reescrita por Wayback, extrae la URL original.
        """
        url = url.strip()

        if not url:
            return ""

        if url.startswith("//"):
            return "https:" + url

        if url.startswith(("http://", "https://")):
            if "web.archive.org/web/" in url:
                return extract_original_from_wayback(url)

            return url

        return urljoin(base, url)

    def _parse_srcset(self, srcset: str, base: str) -> set:
        """
        Parsea srcset='url1 1x, url2 2x, ...' y devuelve URLs absolutas.
        """
        urls = set()

        for entry in srcset.split(","):
            parts = entry.strip().split()

            if parts:
                urls.add(self._abs(parts[0], base))

        return urls

    def _extract_css_urls(self, css_text: str, base: str) -> set:
        """
        Extrae URLs de background-image en CSS inline o bloques <style>.
        """
        urls = set()

        pattern = re.compile(r"url\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)")

        for match in pattern.finditer(css_text):
            url = match.group(1).strip()
            urls.add(self._abs(url, base))

        return urls

    # ─────────────────────────────────────────
    #  Descarga de imágenes
    # ─────────────────────────────────────────

    def _download_images(
        self,
        image_urls: set,
        images_dir: Path,
        timestamp: str,
        max_workers: int = 4
    ) -> tuple:
        """
        Descarga imágenes en paralelo desde Wayback Machine usando el timestamp
        del snapshot.

        Args:
            image_urls: set de URLs originales de imágenes
            images_dir: directorio destino
            timestamp: timestamp del snapshot HTML
            max_workers: threads paralelos

        Returns:
            Tupla (descargadas, fallos)
        """
        images_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for original_image_url in image_urls:
                filename = self._safe_filename(original_image_url)
                output_path = images_dir / filename

                if output_path.exists():
                    downloaded += 1
                    continue

                wayback_image_url = build_wayback_resource_url(
                    timestamp,
                    original_image_url
                )

                future = executor.submit(
                    self._download_single,
                    wayback_image_url,
                    output_path
                )

                futures[future] = (wayback_image_url, filename)

            for future in futures:
                try:
                    success = future.result(timeout=60)

                    if success:
                        downloaded += 1
                    else:
                        failed += 1

                except Exception:
                    failed += 1

        return downloaded, failed

    def _download_single(
        self,
        url: str,
        output_path: Path,
        timeout: int = 20,
        max_retries: int = 3
    ) -> bool:
        """
        Descarga una imagen individual desde Wayback.

        Args:
            url: URL archivada de la imagen en Wayback
            output_path: dónde guardarla
            timeout: timeout en segundos
            max_retries: número de reintentos

        Returns:
            True si éxito, False en caso contrario
        """
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    timeout=timeout,
                    stream=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; WaybackScraper/2.0; "
                            "educational research)"
                        )
                    }
                )

                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "").lower()

                if not content_type.startswith("image/"):
                    return False

                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                return True

            except Exception:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(wait)
                    continue

                return False

        return False

    @staticmethod
    def _safe_filename(url: str) -> str:
        """
        Genera un nombre de archivo seguro desde URL.

        Añade siempre un hash para evitar que imágenes diferentes con el mismo
        nombre se sobrescriban.
        """
        import hashlib
        import os
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path
        original_name = path.split("/")[-1] if path else ""

        hash_val = hashlib.md5(url.encode()).hexdigest()[:12]

        if original_name:
            original_name = re.sub(r"[^\w.-]", "_", original_name)
            name, ext = os.path.splitext(original_name)

            if ext.lower() in (
                ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico"
            ):
                return f"{name[:80]}_{hash_val}{ext.lower()}"

        return f"{hash_val}.jpg"


# ─────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scraper de Wayback Machine con CDX API + Scrapy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # 5 snapshots por año entre 2018 y 2023
  %(prog)s https://example.com --from-year 2018 --to-year 2023 --limit 5

  # Solo la URL exacta
  %(prog)s https://example.com/blog --match exact --limit 3

  # Reanudar ejecución anterior
  %(prog)s https://example.com --from-year 2018 --to-year 2023 --resume

  # Más cobertura
  %(prog)s https://example.com --from-year 2015 --to-year 2024 --limit 20 -o ./output
        """,
    )

    parser.add_argument(
        "url",
        help="URL o dominio a scrapear"
    )

    parser.add_argument(
        "--from-year",
        type=int,
        default=2018,
        metavar="YEAR",
        help="Año de inicio (default: 2018)"
    )

    parser.add_argument(
        "--to-year",
        type=int,
        default=2024,
        metavar="YEAR",
        help="Año de fin (default: 2024)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Snapshots máximos por año (default: 5)"
    )

    parser.add_argument(
        "--match",
        choices=["exact", "prefix", "domain"],
        default="prefix",
        help="Tipo de match CDX (default: prefix)"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="wayback_output",
        help="Directorio de salida (default: wayback_output)"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Requests concurrentes en Scrapy (default: 4)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Saltar snapshots ya descargados"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log detallado"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.from_year > args.to_year:
        logger.error("--from-year no puede ser mayor que --to-year")
        return

    years = list(range(args.from_year, args.to_year + 1))
    normalized = normalize_url(args.url)

    logger.info(f"URL normalizada: {normalized}")
    logger.info(
        f"Años: {years[0]}–{years[-1]} | "
        f"Límite: {args.limit}/año | "
        f"Match: {args.match}"
    )

    snapshots = collect_all_snapshots(
        url=normalized,
        years=years,
        limit_per_year=args.limit,
        match_type=args.match,
    )

    if not snapshots:
        logger.error(
            "No se encontraron snapshots. "
            "Revisa la URL, el rango de años o el tipo de match."
        )
        return

    logger.info(f"Iniciando descarga con Scrapy ({len(snapshots)} snapshots)...")

    process = CrawlerProcess(settings={
        "BOT_NAME": "wayback_scraper",

        "USER_AGENT": (
            "Mozilla/5.0 (compatible; WaybackScraper/2.0; "
            "educational research)"
        ),

        "ROBOTSTXT_OBEY": False,
        "COOKIES_ENABLED": False,

        "CONCURRENT_REQUESTS": args.concurrency,
        "CONCURRENT_REQUESTS_PER_DOMAIN": args.concurrency,
        "DOWNLOAD_DELAY": 1.0,

        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 30.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": max(1, args.concurrency / 2),

        "RETRY_ENABLED": True,
        "RETRY_TIMES": 5,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],

        "DOWNLOAD_TIMEOUT": 45,
        "LOG_LEVEL": "DEBUG" if args.verbose else "WARNING",
    })

    process.crawl(
        WaybackSpider,
        snapshots=snapshots,
        output_dir=args.output,
        skip_existing=args.resume,
    )

    process.start()

    logger.info("¡Completado!")


if __name__ == "__main__":
    main()