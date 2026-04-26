#!/usr/bin/env python3
"""
Differential scraper with robustness recommendations inspired by:
https://gologin.com/blog/web-scraping-with-python/

Added over the current scraper:
- HTTP retries with exponential backoff.
- Respect for robots.txt (optional).
- Randomized delay (rate limiting / anti-burst).
- User-Agent rotation.
- Safer request handling for pages and images.

Note:
This file extends the existing implementation instead of replacing it.
"""

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from advanced_image_scraper import (
    AdvancedImageScraper,
    MIN_IMAGE_SIZE_KB,
    MAX_IMAGE_SIZE_MB,
    SAFE_MAX_PAGES,
)

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

# Semillas por dominio para mejorar cobertura cuando sitemap no ayuda.
DEFAULT_DOMAIN_SEEDS: Dict[str, List[str]] = {
    "dribbble.com": ["/shots/popular", "/shots/recent", "/discover", "/stories"],
    "www.behance.net": ["/galleries", "/joblist", "/assets"],
    "www.deviantart.com": ["/daily-deviations", "/popular-all-time", "/newest"],
    "www.artstation.com": ["/trending", "/search/projects", "/marketplace"],
    "www.pinterest.com": ["/ideas", "/today", "/search/pins/?q=design"],
}


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "site").lower().replace(".", "-")
    path = (parsed.path or "").strip("/").replace("/", "-")
    raw = f"{host}-{path}" if path else host
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "site"


def _is_valid_http_url(candidate: str) -> bool:
    try:
        parsed = urlparse(candidate.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


class DifferentialImageScraper(AdvancedImageScraper):
    """Scraper diferencial con foco en robustez, ética y escalabilidad."""

    def __init__(
        self,
        output_dir: str = "./image_scraper_results",
        delay: float = 1.0,
        convert_to_png: bool = False,
        perceptual_threshold: int = 6,
        perceptual_hash_size: int = 8,
        min_delay: float = 0.8,
        max_delay: float = 2.2,
        max_retries: int = 3,
        backoff_factor: float = 0.8,
        respect_robots: bool = True,
        user_agents: Optional[List[str]] = None,
        seed_config_path: Optional[str] = None,
    ):
        super().__init__(
            output_dir=output_dir,
            delay=delay,
            convert_to_png=convert_to_png,
            perceptual_threshold=perceptual_threshold,
            perceptual_hash_size=perceptual_hash_size,
        )
        self.min_delay = max(0.0, float(min_delay))
        self.max_delay = max(self.min_delay, float(max_delay))
        self.respect_robots = bool(respect_robots)
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self._robots_cache: Dict[str, RobotFileParser] = {}
        self.domain_seeds = self._load_domain_seeds(seed_config_path)

        retry = Retry(
            total=max(0, int(max_retries)),
            connect=max(0, int(max_retries)),
            read=max(0, int(max_retries)),
            status=max(0, int(max_retries)),
            backoff_factor=max(0.0, float(backoff_factor)),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _load_domain_seeds(self, seed_config_path: Optional[str]) -> Dict[str, List[str]]:
        seed_map: Dict[str, List[str]] = {
            domain.lower(): list(paths)
            for domain, paths in DEFAULT_DOMAIN_SEEDS.items()
        }
        if not seed_config_path:
            return seed_map

        try:
            config_path = Path(seed_config_path)
            if not config_path.exists():
                logger.warning(f"No existe seed-config en {seed_config_path}; se usan semillas por defecto")
                return seed_map

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                logger.warning("seed-config inválido: se esperaba un objeto JSON {dominio: [rutas]}")
                return seed_map

            for domain, paths in payload.items():
                if not isinstance(domain, str) or not isinstance(paths, list):
                    continue
                normalized_domain = domain.strip().lower()
                normalized_paths = [str(path).strip() for path in paths if str(path).strip()]
                if normalized_domain and normalized_paths:
                    seed_map[normalized_domain] = normalized_paths
        except Exception as exc:
            logger.warning(f"No se pudo leer seed-config ({seed_config_path}): {exc}")

        return seed_map

    def _discover_home_links(self, root_url: str, allowed_domains: Set[str], max_links: int = 120) -> List[str]:
        """Fallback de descubrimiento desde home cuando el sitemap no aporta URLs."""
        if not self._is_allowed_by_robots(root_url):
            return []
        try:
            response = self._fetch_page(root_url)
        except Exception:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        discovered: List[str] = []
        seen: Set[str] = set()
        for link_url in self._extract_links_from_page(root_url, soup):
            normalized = self._normalize_internal_link(root_url, link_url, allowed_domains)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(normalized)
            if len(discovered) >= max_links:
                break
        return discovered

    def _manual_seed_urls_for_domain(self, root_url: str, allowed_domains: Set[str]) -> List[str]:
        parsed = urlparse(root_url)
        domain = parsed.netloc.lower()
        candidates = self.domain_seeds.get(domain, [])
        seeds: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            resolved = urljoin(root_url, candidate)
            normalized = self._normalize_internal_link(root_url, resolved, allowed_domains)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            seeds.append(normalized)
        return seeds

    def _random_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }

    def _sleep_respectful(self):
        time.sleep(random.uniform(self.min_delay, self.max_delay))

    def _get_robots(self, target_url: str) -> Optional[RobotFileParser]:
        parsed = urlparse(target_url)
        domain = parsed.netloc.lower()
        if not domain:
            return None
        if domain in self._robots_cache:
            return self._robots_cache[domain]

        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            self._robots_cache[domain] = parser
            return parser
        except Exception:
            self._robots_cache[domain] = None  # type: ignore[assignment]
            logger.debug(f"No se pudo leer robots.txt de {domain}; se continúa en modo permisivo")
            return None

    def _is_allowed_by_robots(self, target_url: str) -> bool:
        if not self.respect_robots:
            return True
        parser = self._get_robots(target_url)
        if parser is None:
            return True
        user_agent = self.session.headers.get("User-Agent", "*")
        try:
            return parser.can_fetch(user_agent, target_url)
        except Exception:
            return True

    def _fetch_page(self, url: str) -> requests.Response:
        response = self.session.get(url, timeout=12, headers=self._random_headers())
        response.raise_for_status()
        return response

    def _download_image(self, image_data: Dict):
        """Descarga robusta de imagen con retries y validaciones heredadas."""
        image_url = image_data.get("image_url", "")
        if not image_url:
            return None

        if not self._is_allowed_by_robots(image_url):
            logger.debug(f"Bloqueado por robots.txt: {image_url}")
            return None

        try:
            response = self.session.get(
                image_url,
                timeout=12,
                stream=True,
                headers=self._random_headers(),
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Error descargando {image_url}: {e}")
            self.failed_urls.append(image_url)
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            return None

        content_length = response.headers.get("Content-Length")
        if content_length:
            size_kb = int(content_length) / 1024
            if size_kb < MIN_IMAGE_SIZE_KB or size_kb > MAX_IMAGE_SIZE_MB * 1024:
                return None

        image_data["file_size_kb"] = len(response.content) / 1024
        return response.content, content_type, response.headers

    def scrape(self, url: str, max_pages: int = 10, crawl_site: bool = False) -> Dict:
        logger.info(f"Iniciando scraping diferencial de {url}")

        visited_urls = set()
        allowed_domains = {urlparse(url).netloc.lower()}
        to_visit = [url]
        sitemap_urls: List[str] = []
        home_fallback_urls: List[str] = []
        manual_seed_urls: List[str] = []
        skipped_by_robots = 0
        page_download_errors = 0
        page_extraction_errors = 0
        initial_queue_size = 1

        if max_pages <= 0:
            page_limit = SAFE_MAX_PAGES
            logger.warning(
                f"max_pages={max_pages} detectado; usando límite de seguridad de {SAFE_MAX_PAGES} páginas"
            )
        else:
            page_limit = min(max_pages, SAFE_MAX_PAGES)

        sitemap_limit = page_limit * 20

        if crawl_site:
            sitemap_urls = self._discover_sitemap_urls(url, allowed_domains, limit=sitemap_limit)
            if not sitemap_urls:
                home_fallback_urls = self._discover_home_links(url, allowed_domains, max_links=120)
            manual_seed_urls = self._manual_seed_urls_for_domain(url, allowed_domains)

            merged_discovery: List[str] = []
            seen_discovery: Set[str] = set()
            for candidate in sitemap_urls + home_fallback_urls + manual_seed_urls + to_visit:
                if candidate in seen_discovery:
                    continue
                seen_discovery.add(candidate)
                merged_discovery.append(candidate)
            to_visit = merged_discovery

        initial_queue_size = len(to_visit)

        to_visit = self._prioritize_to_visit_queue(to_visit, visited_urls)
        pages_processed = 0

        while to_visit and pages_processed < page_limit:
            current_url = to_visit.pop(0)

            if current_url in visited_urls or current_url in self.seen_pages:
                continue

            if not self._is_allowed_by_robots(current_url):
                logger.info(f"Saltando por robots.txt: {current_url}")
                visited_urls.add(current_url)
                self.seen_pages.add(current_url)
                skipped_by_robots += 1
                continue

            visited_urls.add(current_url)
            self.seen_pages.add(current_url)
            pages_processed += 1
            logger.info(f"Procesando página {pages_processed}/{page_limit}: {current_url}")

            try:
                response = self._fetch_page(current_url)
            except Exception as e:
                logger.error(f"Error descargando {current_url}: {e}")
                page_download_errors += 1
                continue

            try:
                images_data = self._extract_images_from_html(response.text, current_url)
                logger.info(f"Encontradas {len(images_data)} imágenes en {current_url}")
            except Exception as e:
                logger.error(f"Error extrayendo imágenes de {current_url}: {e}")
                page_extraction_errors += 1
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for link_url in self._extract_links_from_page(current_url, soup):
                normalized = self._normalize_internal_link(current_url, link_url, allowed_domains)
                if normalized and normalized not in visited_urls and normalized not in to_visit:
                    to_visit.append(normalized)

            to_visit = self._prioritize_to_visit_queue(to_visit, visited_urls)

            for img_data in images_data:
                self._process_single_image(img_data)

            self._sleep_respectful()

        if pages_processed >= page_limit:
            crawl_stop_code = "MAX_PAGES_REACHED"
        elif not to_visit:
            crawl_stop_code = "QUEUE_EXHAUSTED"
        else:
            crawl_stop_code = "STOPPED_OTHER"

        no_crawl_reason = ""
        no_crawl_reason_codes: List[str] = []
        diagnostic_context = {
            "crawl_site": bool(crawl_site),
            "requested_max_pages": int(max_pages),
            "effective_page_limit": int(page_limit),
            "sitemap_urls_discovered": len(sitemap_urls),
            "home_fallback_urls_discovered": len(home_fallback_urls),
            "manual_seed_urls_added": len(manual_seed_urls),
            "skipped_by_robots": int(skipped_by_robots),
            "page_download_errors": int(page_download_errors),
            "page_extraction_errors": int(page_extraction_errors),
            "initial_queue_size": int(initial_queue_size),
            "remaining_queue_size": int(len(to_visit)),
            "crawl_stop_code": crawl_stop_code,
        }

        if pages_processed == 0:
            reasons = []
            if crawl_site and not sitemap_urls:
                no_crawl_reason_codes.append("SITEMAP_EMPTY")
                reasons.append("no se descubrieron URLs desde sitemap")
                if home_fallback_urls:
                    no_crawl_reason_codes.append("HOME_FALLBACK_DISCOVERED")
                    reasons.append(f"fallback home encontró {len(home_fallback_urls)} URL(s)")
                else:
                    no_crawl_reason_codes.append("HOME_FALLBACK_EMPTY")
                    reasons.append("fallback home no encontró enlaces válidos")
            if crawl_site and manual_seed_urls:
                no_crawl_reason_codes.append("MANUAL_SEEDS_AVAILABLE")
                reasons.append(f"semillas manuales añadidas: {len(manual_seed_urls)} URL(s)")
            if skipped_by_robots > 0:
                no_crawl_reason_codes.append("ROBOTS_BLOCK_ALL")
                reasons.append(f"robots.txt bloqueó {skipped_by_robots} URL(s)")
            if page_download_errors > 0:
                no_crawl_reason_codes.append("PAGE_FETCH_ERROR")
                reasons.append(f"hubo {page_download_errors} error(es) al descargar páginas")
            if page_extraction_errors > 0:
                no_crawl_reason_codes.append("HTML_EXTRACTION_ERROR")
                reasons.append(f"hubo {page_extraction_errors} error(es) al extraer HTML")
            if not reasons:
                no_crawl_reason_codes.append("NO_ELIGIBLE_URLS")
                reasons.append("no hubo URLs válidas para procesar tras filtros de dominio/cola")

            logger.warning(
                "No se procesó ninguna página. Códigos: %s. Motivo(s): %s",
                ",".join(no_crawl_reason_codes),
                " | ".join(reasons),
            )
            no_crawl_reason = " | ".join(reasons)

        self._generate_reports()
        return {
            "total_images_found": len(self.downloaded_images),
            "total_failed": len(self.failed_urls),
            "pages_processed": pages_processed,
            "output_directory": str(self.output_dir),
            "crawl_stop_code": crawl_stop_code,
            "no_crawl_reason": no_crawl_reason,
            "no_crawl_reason_codes": no_crawl_reason_codes,
            "diagnostic_context": diagnostic_context,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Differential Image Scraper (best-practices: retries, robots, delays, headers)"
    )
    parser.add_argument("--url", required=False, default="", help="URL del sitio a scrapear")
    parser.add_argument("--output-dir", default="./image_scraper_results_differential", help="Directorio de salida")
    parser.add_argument("--max-pages", type=int, default=10, help="Número máximo de páginas")
    parser.add_argument("--crawl-site", action="store_true", help="Recorrer enlaces internos")
    parser.add_argument("--convert-to-png", action="store_true", help="Convertir imágenes a PNG")
    parser.add_argument("--perceptual-threshold", type=int, default=6, help="Umbral de deduplicación perceptual")
    parser.add_argument("--min-delay", type=float, default=0.8, help="Delay mínimo entre peticiones")
    parser.add_argument("--max-delay", type=float, default=2.2, help="Delay máximo entre peticiones")
    parser.add_argument("--max-retries", type=int, default=3, help="Reintentos HTTP automáticos")
    parser.add_argument("--backoff-factor", type=float, default=0.8, help="Factor de backoff exponencial")
    parser.add_argument("--ignore-robots", action="store_true", help="No respetar robots.txt")
    parser.add_argument("--interactive-batch", action="store_true", help="Pedir URLs por input y procesarlas en lote")
    parser.add_argument("--max-input-urls", type=int, default=20, help="Máximo de URLs válidas a guardar en lote")
    parser.add_argument(
        "--min-pages-success",
        type=int,
        default=10,
        help="Mínimo de páginas procesadas para aceptar una URL en modo lote",
    )
    parser.add_argument(
        "--seed-config",
        default="",
        help="Ruta a JSON con semillas por dominio {dominio: [rutas]} para fallback de descubrimiento",
    )

    args = parser.parse_args()

    if args.interactive_batch:
        max_inputs = max(1, int(args.max_input_urls))
        min_pages_success = max(0, int(args.min_pages_success))
        base_output = Path(args.output_dir)
        base_output.mkdir(parents=True, exist_ok=True)

        accepted_results: List[Dict] = []
        attempts = 0

        print("\n" + "=" * 60)
        print("MODO LOTE INTERACTIVO")
        print("=" * 60)
        print(f"Objetivo: guardar {max_inputs} URL(s) válidas")
        print(f"Criterio mínimo: páginas procesadas >= {min_pages_success}")
        print("Escribe 'fin' para terminar antes.")
        print("=" * 60)

        while len(accepted_results) < max_inputs:
            prompt = f"URL #{len(accepted_results) + 1} (válidas) > "
            user_input = input(prompt).strip()
            if not user_input:
                continue
            if user_input.lower() in {"fin", "exit", "quit", "q"}:
                break
            if not _is_valid_http_url(user_input):
                print("URL inválida. Debe empezar por http:// o https://")
                continue

            attempts += 1
            url_slug = _slug_from_url(user_input)
            run_output_dir = base_output / f"{len(accepted_results) + 1:02d}_{url_slug}"
            if run_output_dir.exists():
                shutil.rmtree(run_output_dir, ignore_errors=True)

            scraper = DifferentialImageScraper(
                output_dir=str(run_output_dir),
                convert_to_png=args.convert_to_png,
                perceptual_threshold=args.perceptual_threshold,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                max_retries=args.max_retries,
                backoff_factor=args.backoff_factor,
                respect_robots=not args.ignore_robots,
                seed_config_path=args.seed_config or None,
            )

            result = scraper.scrape(user_input, max_pages=args.max_pages, crawl_site=args.crawl_site)
            pages_processed = int(result.get("pages_processed", 0))

            if pages_processed < min_pages_success:
                reason_codes = result.get("no_crawl_reason_codes") or []
                reason_text = result.get("no_crawl_reason") or "No alcanzó el umbral mínimo"
                print(
                    "DESCARTADA: "
                    f"{user_input} | páginas={pages_processed} (< {min_pages_success}) | "
                    f"códigos={','.join(reason_codes) if reason_codes else 'N/A'} | motivo={reason_text}"
                )
                shutil.rmtree(run_output_dir, ignore_errors=True)
                print("Se vuelve a pedir una URL y no se guarda este resultado.")
                continue

            accepted_results.append(
                {
                    "url": user_input,
                    "pages_processed": pages_processed,
                    "total_images_found": int(result.get("total_images_found", 0)),
                    "output_directory": str(run_output_dir),
                    "crawl_stop_code": result.get("crawl_stop_code", "N/A"),
                }
            )
            print(
                "ACEPTADA: "
                f"{user_input} | páginas={pages_processed} | imágenes={result.get('total_images_found', 0)} | "
                f"salida={run_output_dir}"
            )

        print("\n" + "=" * 60)
        print("RESUMEN LOTE")
        print("=" * 60)
        print(f"Intentos totales: {attempts}")
        print(f"URLs aceptadas: {len(accepted_results)}")
        for idx, entry in enumerate(accepted_results, start=1):
            print(
                f"{idx:02d}. {entry['url']} | páginas={entry['pages_processed']} | "
                f"imágenes={entry['total_images_found']} | stop={entry['crawl_stop_code']}"
            )
        print(f"Resultados base en: {base_output}")
        print("=" * 60)
        return

    if not args.url:
        parser.error("Debes pasar --url en modo normal o usar --interactive-batch")

    scraper = DifferentialImageScraper(
        output_dir=args.output_dir,
        convert_to_png=args.convert_to_png,
        perceptual_threshold=args.perceptual_threshold,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        respect_robots=not args.ignore_robots,
        seed_config_path=args.seed_config or None,
    )

    result = scraper.scrape(args.url, max_pages=args.max_pages, crawl_site=args.crawl_site)

    print("\n" + "=" * 60)
    print("SCRAPING DIFERENCIAL COMPLETADO")
    print("=" * 60)
    print(f"Imágenes descargadas: {result['total_images_found']}")
    print(f"Fallos: {result['total_failed']}")
    print(f"Páginas procesadas: {result['pages_processed']}")
    print(f"Código de parada: {result.get('crawl_stop_code', 'N/A')}")
    if result.get("pages_processed", 0) == 0 and result.get("no_crawl_reason"):
        reason_codes = result.get("no_crawl_reason_codes") or []
        if reason_codes:
            print(f"Código(s) sin crawl: {', '.join(reason_codes)}")
        print(f"Motivo sin crawl: {result['no_crawl_reason']}")
        ctx = result.get("diagnostic_context") or {}
        print(
            "Contexto: "
            f"crawl_site={ctx.get('crawl_site')} | "
            f"max_pages_solicitado={ctx.get('requested_max_pages')} | "
            f"max_pages_efectivo={ctx.get('effective_page_limit')} | "
            f"sitemaps={ctx.get('sitemap_urls_discovered')} | "
            f"home_fallback={ctx.get('home_fallback_urls_discovered')} | "
            f"manual_seeds={ctx.get('manual_seed_urls_added')} | "
            f"robots_skips={ctx.get('skipped_by_robots')} | "
            f"fetch_errors={ctx.get('page_download_errors')} | "
            f"extract_errors={ctx.get('page_extraction_errors')}"
        )
    print(f"Resultados en: {result['output_directory']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
