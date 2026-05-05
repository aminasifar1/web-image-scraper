#!/usr/bin/env python3
"""Descargador de imágenes públicas centrado en arte.

Fuentes objetivo (enfocadas):
- Flickr
- Flickr Commons
- Behance
- DeviantArt
- Library of Congress Pictures

Mejoras principales:
- Crawling multipágina interno por dominio (no solo home).
- Filtro anti-logo/icono/avatar/sprite.
- Umbral de tamaño mínimo más estricto para priorizar obras reales.

Uso:
    python public_art_downloader.py --per-site 60 --max-pages 6 --output-dir public_art_images_focused
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import ssl
import time
from io import BytesIO
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from PIL import Image


TARGET_SITES: Sequence[Tuple[str, str]] = (
    ("flickr", "https://www.flickr.com"),
    ("flickr_commons", "https://www.flickr.com/commons"),
    ("behance", "https://www.behance.net"),
    ("deviantart", "https://www.deviantart.com"),
    ("loc_pictures", "https://www.loc.gov/pictures/"),
)

SITE_SEED_PATHS: Dict[str, Sequence[str]] = {
    "flickr": (
        "/explore",
        "/search/?text=art",
        "/photos/tags/art",
        "/photos/tags/painting",
    ),
    "flickr_commons": (
        "/commons",
        "/search/?text=commons",
        "/search/?text=historical+photos",
    ),
    "behance": (
        "/galleries/illustration",
        "/galleries/digital-art",
        "/galleries/character-design",
        "/search/projects?search=art",
    ),
    "deviantart": (
        "/tag/art",
        "/tag/digitalart",
        "/tag/illustration",
        "/daily-deviations",
    ),
    "loc_pictures": (
        "/pictures/",
        "/pictures/collection/",
        "/pictures/search/?q=art",
        "/pictures/search/?q=painting",
    ),
}

ALLOW_LINK_HINTS: Dict[str, Sequence[str]] = {
    "flickr": ("/photos/", "/explore", "/search", "/tags/", "/commons"),
    "flickr_commons": ("/commons", "/photos/", "/search", "/tags/"),
    "behance": ("/gallery/", "/galleries/", "/search/", "/projects/"),
    "deviantart": ("/art/", "/tag/", "/daily-deviations", "/watch/"),
    "loc_pictures": ("/pictures/collection/", "/pictures/item/", "/pictures/search/"),
}

BLOCK_IMAGE_KEYWORDS = {
    "logo",
    "icon",
    "avatar",
    "sprite",
    "favicon",
    "badge",
    "placeholder",
    "loader",
    "spinner",
    "pixel",
    "thumb-default",
    "profilepic",
}

MIN_IMAGE_BYTES = 2 * 1024
MIN_IMAGE_WIDTH = 900
MIN_IMAGE_HEIGHT = 600

SITE_IMAGE_HINTS: Dict[str, Sequence[str]] = {
    "behance": ("behance.net", "mir-s3-cdn-cf.behance.net", "project_modules", "projects"),
    "deviantart": ("images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/", "d"),
    "flickr": ("live.staticflickr.com/", "staticflickr.com/"),
    "flickr_commons": ("live.staticflickr.com/", "staticflickr.com/"),
    "loc_pictures": ("/pictures/", "iiif.", "/static/data/"),
}

SITE_MIN_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "behance": (320, 240),
    "deviantart": (280, 200),
    "flickr": (300, 200),
    "flickr_commons": (300, 200),
    "loc_pictures": (260, 180),
}

SITE_MIN_PAUSE: Dict[str, float] = {
    "loc_pictures": 1.2,
    "flickr_commons": 0.4,
}

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class PageExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.image_urls: List[str] = []
        self.links: List[str] = []
        self.page_title: str = ""
        self._in_title = False
        self._title_parts: List[str] = []
        self.image_meta: Dict[str, Dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        candidates: List[str] = []

        if tag.lower() == "title":
            self._in_title = True

        if tag.lower() == "img":
            for key in ("data-full", "data-original", "data-src", "data-lazy-src", "src"):
                value = attrs_dict.get(key, "").strip()
                if value:
                    candidates.append(value)

            alt_text = attrs_dict.get("alt", "").strip()
            title_text = attrs_dict.get("title", "").strip()
            aria_label = attrs_dict.get("aria-label", "").strip()
            meta = {
                "alt_text": alt_text,
                "title_text": title_text,
                "aria_label": aria_label,
                "loading": attrs_dict.get("loading", "").strip(),
                "width": attrs_dict.get("width", "").strip(),
                "height": attrs_dict.get("height", "").strip(),
                "srcset": attrs_dict.get("srcset", "").strip(),
            }

            for key in ("data-srcset", "srcset"):
                srcset = attrs_dict.get(key, "").strip()
                if srcset:
                    best = select_best_srcset(srcset)
                    if best:
                        candidates.append(best)

        if tag.lower() == "source":
            srcset = attrs_dict.get("srcset", "").strip()
            if srcset:
                best = select_best_srcset(srcset)
                if best:
                    candidates.append(best)

        if tag.lower() == "a":
            href = attrs_dict.get("href", "").strip()
            if href:
                self.links.append(urljoin(self.base_url, href))

        for c in candidates:
            abs_url = urljoin(self.base_url, c)
            if abs_url:
                self.image_urls.append(abs_url)
                if tag.lower() == "img":
                    self.image_meta.setdefault(abs_url, meta)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            self.page_title = clean_text(" ".join(self._title_parts))

    def handle_data(self, data: str) -> None:
        if self._in_title and data:
            self._title_parts.append(data)


def clean_text(value: str, max_len: int = 300) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip()
    if len(value) > max_len:
        return value[:max_len].rstrip() + "..."
    return value


def normalize_meta_text(value: str) -> str:
    """Normaliza texto de metadatos para comparar similitud."""
    normalized = clean_text(value, max_len=400).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def metadata_signature_candidates(site_key: str, image_meta: Dict[str, str], width: int, height: int) -> Set[str]:
    """Genera firmas de metadatos para detectar duplicados semánticos.

    Se basa en ALT/TITLE/ARIA con normalización + relación de aspecto.
    """
    signatures: Set[str] = set()
    ratio_bucket = round((width / height), 2) if width and height else 0.0
    dim_bucket = f"{int(round(width / 100.0) * 100)}x{int(round(height / 100.0) * 100)}"

    generic_tokens = {
        "image",
        "img",
        "photo",
        "picture",
        "thumbnail",
        "art",
        "untitled",
    }

    for field_name in ("alt_text", "title_text", "aria_label"):
        raw_value = image_meta.get(field_name, "")
        normalized = normalize_meta_text(raw_value)
        if not normalized:
            continue
        tokens = [t for t in normalized.split() if t]
        if len(tokens) < 2:
            continue
        if set(tokens).issubset(generic_tokens):
            continue
        if len(normalized) < 12:
            continue

        base = f"{site_key}|{field_name}|{normalized}"
        signatures.add(hashlib.sha1(base.encode("utf-8")).hexdigest())
        signatures.add(hashlib.sha1(f"{base}|r:{ratio_bucket}".encode("utf-8")).hexdigest())
        signatures.add(hashlib.sha1(f"{base}|d:{dim_bucket}".encode("utf-8")).hexdigest())

    return signatures


def select_best_srcset(srcset: str) -> Optional[str]:
    best_url: Optional[str] = None
    best_score = -1.0
    for part in srcset.split(","):
        item = part.strip()
        if not item:
            continue
        chunks = item.split()
        url = chunks[0]
        score = 0.0
        if len(chunks) > 1:
            descriptor = chunks[-1].lower()
            if descriptor.endswith("w"):
                try:
                    score = float(descriptor[:-1])
                except ValueError:
                    score = 0.0
            elif descriptor.endswith("x"):
                try:
                    score = float(descriptor[:-1]) * 1000
                except ValueError:
                    score = 0.0
        if score >= best_score:
            best_score = score
            best_url = url
    return best_url


def make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_html(url: str, timeout: int = 12) -> Tuple[str, int, str]:
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*;q=0.8"})
    with urlopen(req, timeout=timeout, context=make_ssl_context()) as resp:
        code = int(getattr(resp, "status", resp.getcode()))
        final_url = getattr(resp, "url", url)
        body = resp.read().decode("utf-8", errors="ignore")
        return final_url, code, body


def looks_like_image_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    if not path:
        return False
    if path.endswith(".svg"):
        return False
    lowered = url.lower()
    if any(keyword in lowered for keyword in BLOCK_IMAGE_KEYWORDS):
        return False
    if any(part in lowered for part in ("logo", "icon", "avatar", "badge", "sprite", "favicon", "placeholder")):
        return False
    if re.search(r"\.(jpg|jpeg|png|webp|bmp|tiff|tif|gif)(?:$|\?)", path):
        return True
    return any(host in parsed.netloc.lower() for host in ("staticflickr", "wixmp", "behance", "loc.gov"))


def normalize_link(url: str) -> str:
    if not url:
        return ""
    # Evita errores por espacios no escapados en query/path.
    url = url.strip().replace(" ", "%20")
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="", query=parsed.query)
    return cleaned.geturl()


def same_domain(url: str, root_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(root_url).netloc.lower()


def allowed_internal_link(site_key: str, url: str, root_url: str) -> bool:
    if not same_domain(url, root_url):
        return False
    hints = ALLOW_LINK_HINTS.get(site_key, ())
    lowered = url.lower()
    return any(hint in lowered for hint in hints)


def allowed_image_url(site_key: str, url: str) -> bool:
    lowered = url.lower()
    hints = SITE_IMAGE_HINTS.get(site_key, ())
    if hints and not any(hint in lowered for hint in hints):
        return False
    if site_key == "behance":
        if any(block in lowered for block in ("avatar", "page_banner", "/img/creator_pro/", "creator_pro")):
            return False
        if not any(block in lowered for block in ("projects", "project_modules", "mir-s3-cdn-cf.behance.net", "a5.behance.net")):
            return False
    if site_key == "loc_pictures":
        # Evita muchos assets de UI; prioriza colecciones, iiif y datos de imágenes
        if any(block in lowered for block in ("leftnav", "blog.gif", "search_arrow", "logo", "favicon", "sprite")):
            return False
    return True


def image_dimensions(payload: bytes) -> Tuple[int, int]:
    with Image.open(BytesIO(payload)) as img:
        return img.size


def meets_min_dimensions(site_key: str, width: int, height: int) -> bool:
    # Flickr/Flickr Commons: aceptar verticales útiles y no solo landscape.
    if site_key in {"flickr", "flickr_commons"}:
        if min(width, height) < 180:
            return False
        return (width * height) >= 70_000

    min_w, min_h = SITE_MIN_DIMENSIONS.get(site_key, (MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT))
    return width >= min_w and height >= min_h


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def infer_extension(url: str, content_type: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
        if path.endswith(ext):
            return ext
    ctype = (content_type or "").lower()
    if "png" in ctype:
        return ".png"
    if "webp" in ctype:
        return ".webp"
    if "gif" in ctype:
        return ".gif"
    return ".jpg"


def download_binary(url: str, timeout: int = 15) -> Tuple[bytes, str, int]:
    req = Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
    with urlopen(req, timeout=timeout, context=make_ssl_context()) as resp:
        code = int(getattr(resp, "status", resp.getcode()))
        content_type = str(resp.headers.get("Content-Type", ""))
        data = resp.read()
        return data, content_type, code


def site_folder(base_output: Path, site_key: str) -> Path:
    p = base_output / site_key
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_candidates_for_site(site_key: str, site_url: str, max_pages: int, pause: float) -> Tuple[List[Tuple[str, str, str, Dict[str, str]]], List[Dict[str, str]]]:
    """Devuelve candidatos de imagen y eventos de página.

    candidates: [(page_url, page_status, image_url), ...]
    """
    events: List[Dict[str, str]] = []
    seeds = [site_url]
    for rel in SITE_SEED_PATHS.get(site_key, ()):
        seeds.append(urljoin(site_url, rel))

    queue = unique_keep_order(seeds)
    visited: Set[str] = set()
    candidates: List[Tuple[str, str, str, Dict[str, str]]] = []
    seen_image_urls: Set[str] = set()
    effective_pause = max(0.0, pause, SITE_MIN_PAUSE.get(site_key, 0.0))

    while queue and len(visited) < max(1, int(max_pages)):
        current = normalize_link(queue.pop(0))
        if current in visited:
            continue
        visited.add(current)

        try:
            final_url, status, html = fetch_html(current)
        except Exception as exc:
            events.append(
                {
                    "site": site_key,
                    "site_url": site_url,
                    "page_url": current,
                    "page_status": "error",
                    "image_url": "",
                    "local_path": "",
                    "bytes": "0",
                    "status": "page_error",
                    "error": str(exc),
                }
            )
            continue

        extractor = PageExtractor(final_url)
        extractor.feed(html)
        extractor.close()
        page_title = clean_text(extractor.page_title)

        for img_url in extractor.image_urls:
            img_url = normalize_link(img_url)
            if img_url in seen_image_urls:
                continue
            if not looks_like_image_url(img_url):
                continue
            if not allowed_image_url(site_key, img_url):
                continue
            seen_image_urls.add(img_url)
            image_meta = extractor.image_meta.get(img_url, {})
            image_meta = {
                "alt_text": clean_text(image_meta.get("alt_text", "")),
                "title_text": clean_text(image_meta.get("title_text", "")),
                "aria_label": clean_text(image_meta.get("aria_label", "")),
                "loading": clean_text(image_meta.get("loading", "")),
                "width_hint": clean_text(image_meta.get("width", "")),
                "height_hint": clean_text(image_meta.get("height", "")),
                "srcset": clean_text(image_meta.get("srcset", ""), 500),
                "page_title": page_title,
            }
            candidates.append((final_url, str(status), img_url, image_meta))

        for link in extractor.links:
            link = normalize_link(link)
            if link in visited:
                continue
            if not allowed_internal_link(site_key, link, site_url):
                continue
            if link not in queue and len(queue) < max(250, max_pages * 40):
                queue.append(link)

        time.sleep(effective_pause)

    return candidates, events


def download_site_images(site_key: str, site_url: str, output_dir: Path, per_site: int, pause: float, max_pages: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    candidates, events = collect_candidates_for_site(
        site_key=site_key,
        site_url=site_url,
        max_pages=max(1, int(max_pages)),
        pause=max(0.0, pause),
    )
    rows.extend(events)
    candidates = candidates[: max(0, int(per_site))]
    effective_pause = max(0.0, pause, SITE_MIN_PAUSE.get(site_key, 0.0))

    site_out = site_folder(output_dir, site_key)
    idx = 0
    seen_hashes: Set[str] = set()
    seen_canonical: Set[str] = set()
    seen_metadata_signatures: Dict[str, str] = {}
    for page_url, page_status, img_url, image_meta in candidates:
        idx += 1
        canonical = re.sub(r"[?].*$", "", img_url)
        canonical = re.sub(r"/\d+(?:x\d+)?(?=/|$)", "/SIZE", canonical)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        row = {
            "site": site_key,
            "site_url": site_url,
            "page_url": page_url,
            "page_status": str(page_status),
            "image_url": img_url,
            "local_path": "",
            "bytes": "0",
            "status": "error",
            "error": "",
            "page_title": image_meta.get("page_title", ""),
            "alt_text": image_meta.get("alt_text", ""),
            "title_text": image_meta.get("title_text", ""),
            "aria_label": image_meta.get("aria_label", ""),
            "loading": image_meta.get("loading", ""),
            "width_hint": image_meta.get("width_hint", ""),
            "height_hint": image_meta.get("height_hint", ""),
            "srcset": image_meta.get("srcset", ""),
            "duplicate_rule": "",
            "matched_with_url": "",
        }
        try:
            payload, ctype, code = download_binary(img_url)
            if code >= 400:
                row["status"] = "http_error"
                row["error"] = f"HTTP {code}"
            elif not (ctype.lower().startswith("image/") or payload.startswith(b"\x89PNG") or payload.startswith(b"\xff\xd8")):
                row["status"] = "not_image"
                row["error"] = f"content-type={ctype}"
            elif len(payload) < MIN_IMAGE_BYTES:
                row["status"] = "too_small"
                row["error"] = f"payload < {MIN_IMAGE_BYTES} bytes"
            else:
                try:
                    width, height = image_dimensions(payload)
                except Exception as exc:
                    row["status"] = "bad_image"
                    row["error"] = f"cannot read image: {exc}"
                    rows.append(row)
                    time.sleep(effective_pause)
                    continue

                row["page_status"] = f"{page_status};{width}x{height}"
                if not meets_min_dimensions(site_key, width, height):
                    row["status"] = "too_small"
                    row["error"] = f"dimensions {width}x{height} below minimum"
                    rows.append(row)
                    time.sleep(effective_pause)
                    continue

                # Dedupe semántico por metadatos similares (ALT/TITLE/ARIA).
                # Solo aplica cuando hay texto útil; evita guardar variantes repetidas.
                meta_signatures = metadata_signature_candidates(site_key, image_meta, width, height)
                if meta_signatures:
                    matched = None
                    for sig in meta_signatures:
                        if sig in seen_metadata_signatures:
                            matched = seen_metadata_signatures[sig]
                            break
                    if matched:
                        row["status"] = "duplicate_metadata"
                        row["duplicate_rule"] = "alt_title_aria_similarity"
                        row["matched_with_url"] = matched
                        row["error"] = "duplicate by similar metadata signature"
                        rows.append(row)
                        time.sleep(effective_pause)
                        continue
                    for sig in meta_signatures:
                        seen_metadata_signatures[sig] = img_url

                payload_hash = hashlib.sha1(payload).hexdigest()
                if payload_hash in seen_hashes:
                    row["status"] = "duplicate_payload"
                    row["duplicate_rule"] = "payload_sha1"
                    row["error"] = "duplicate by payload hash"
                    rows.append(row)
                    time.sleep(effective_pause)
                    continue
                seen_hashes.add(payload_hash)

                ext = infer_extension(img_url, ctype)
                digest = hashlib.sha1(img_url.encode("utf-8")).hexdigest()[:10]
                filename = f"{idx:04d}_{digest}{ext}"
                out_path = site_out / filename
                out_path.write_bytes(payload)
                row["local_path"] = str(out_path)
                row["bytes"] = str(len(payload))
                row["status"] = "downloaded"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)
        time.sleep(effective_pause)

    if not candidates:
        rows.append(
            {
                "site": site_key,
                "site_url": site_url,
                "page_url": site_url,
                "page_status": "n/a",
                "image_url": "",
                "local_path": "",
                "bytes": "0",
                "status": "no_candidates",
                "error": "No image candidates found in crawled pages",
                "page_title": "",
                "alt_text": "",
                "title_text": "",
                "aria_label": "",
                "loading": "",
                "width_hint": "",
                "height_hint": "",
                "srcset": "",
                "duplicate_rule": "",
                "matched_with_url": "",
            }
        )
    return rows


def write_report(rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["site", "site_url", "page_url", "page_title", "page_status", "image_url", "alt_text", "title_text", "aria_label", "loading", "width_hint", "height_hint", "srcset", "local_path", "bytes", "status", "duplicate_rule", "matched_with_url", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga imágenes públicas de sitios de arte seleccionados")
    parser.add_argument("--output-dir", default="public_art_images", help="Directorio de salida")
    parser.add_argument("--report", default="public_art_images/download_report.csv", help="CSV de reporte")
    parser.add_argument("--per-site", type=int, default=80, help="Máximo de imágenes por sitio")
    parser.add_argument("--max-pages", type=int, default=10, help="Máximo de páginas a rastrear por sitio")
    parser.add_argument("--pause", type=float, default=0.35, help="Pausa entre requests (segundos)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    for site_key, site_url in TARGET_SITES:
        print(f"[site] {site_key}: {site_url}")
        rows = download_site_images(
            site_key=site_key,
            site_url=site_url,
            output_dir=output_dir,
            per_site=max(0, int(args.per_site)),
            pause=max(0.0, float(args.pause)),
            max_pages=max(1, int(args.max_pages)),
        )
        downloaded_count = sum(1 for r in rows if r.get("status") == "downloaded")
        print(f"  -> downloaded: {downloaded_count}")
        all_rows.extend(rows)

    report_path = Path(args.report)
    write_report(all_rows, report_path)
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
