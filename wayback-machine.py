#!/usr/bin/env python3
"""wayback-machine.py

Consulta la Wayback Machine (CDX API) para snapshots por año
e integra con `AdvancedImageScraper` para ejecutar el scrapper
sobre las URLs archivadas.

Primer uso de prueba: años 2020-2025.

Mejoras:
- Extrae URLs de imágenes del HTML histórico (img, picture, srcset, backgrounds)
- Detecta y valida URLs relativas y absolutas
- Crawl automático opcional con Scrapy
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import List, Set, Dict, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from advanced_image_scraper import AdvancedImageScraper

logger = logging.getLogger("wayback")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_images_from_html(html_content: str, base_url: str) -> Set[str]:
    """Extrae todas las URLs de imágenes del HTML histórico.
    
    Detecta:
    - <img src="...">
    - <picture><source srcset="...">
    - srcset con múltiples URLs
    - background-image en estilos CSS
    - data-* atributos comunes
    
    Args:
        html_content: contenido HTML del snapshot
        base_url: URL base para resolver rutas relativas
        
    Returns:
        Set de URLs de imágenes absolutas
    """
    image_urls = set()
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.warning(f"Error parsing HTML: {e}")
        return image_urls
    
    # 1. <img> tags con src
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
        if src:
            img_url = urljoin(base_url, src.strip())
            if _is_valid_image_url(img_url):
                image_urls.add(img_url)
        
        # srcset en <img>
        srcset = img.get('srcset')
        if srcset:
            image_urls.update(_parse_srcset(srcset, base_url))
    
    # 2. <picture> tags
    for picture in soup.find_all('picture'):
        for source in picture.find_all('source'):
            srcset = source.get('srcset')
            if srcset:
                image_urls.update(_parse_srcset(srcset, base_url))
        # Fallback a <img> dentro de <picture>
        img = picture.find('img')
        if img:
            src = img.get('src')
            if src:
                img_url = urljoin(base_url, src.strip())
                if _is_valid_image_url(img_url):
                    image_urls.add(img_url)
    
    # 3. Background images en <style> o atributo style
    # Buscar en style tags
    for style in soup.find_all('style'):
        urls = _extract_urls_from_css(style.string, base_url)
        image_urls.update(urls)
    
    # Buscar en atributos style
    for element in soup.find_all(style=True):
        urls = _extract_urls_from_css(element.get('style'), base_url)
        image_urls.update(urls)
    
    # 4. <link rel="preload"> para imágenes
    for link in soup.find_all('link', rel=['preload', 'prefetch']):
        href = link.get('href')
        if href and link.get('as') == 'image':
            img_url = urljoin(base_url, href.strip())
            if _is_valid_image_url(img_url):
                image_urls.add(img_url)
    
    # 5. AMP images
    for amp_img in soup.find_all('amp-img'):
        src = amp_img.get('src')
        if src:
            img_url = urljoin(base_url, src.strip())
            if _is_valid_image_url(img_url):
                image_urls.add(img_url)
    
    logger.info(f"Extraídas {len(image_urls)} imágenes del HTML")
    return image_urls


def _parse_srcset(srcset: str, base_url: str) -> Set[str]:
    """Parsea srcset y extrae URLs válidas.
    
    Maneja formato: "url1.jpg 1x, url2.jpg 2x" o "url1.jpg 480w, url2.jpg 1024w"
    """
    urls = set()
    if not srcset:
        return urls
    
    entries = [e.strip() for e in srcset.split(',')]
    for entry in entries:
        # Dividir por espacios, primera parte es la URL
        parts = entry.split()
        if parts:
            url = parts[0].strip()
            abs_url = urljoin(base_url, url)
            if _is_valid_image_url(abs_url):
                urls.add(abs_url)
    return urls


def _extract_urls_from_css(css_content: str, base_url: str) -> Set[str]:
    """Extrae URLs de background-image y similares en CSS."""
    urls = set()
    if not css_content:
        return urls
    
    # Buscar url(...) en CSS
    url_pattern = r'url\([\'"]?([^\)\'\"]+)[\'"]?\)'
    matches = re.findall(url_pattern, css_content)
    for match in matches:
        url = match.strip()
        abs_url = urljoin(base_url, url)
        if _is_valid_image_url(abs_url):
            urls.add(abs_url)
    return urls


def _is_valid_image_url(url: str) -> bool:
    """Valida que sea una URL de imagen válida.
    
    Filtra:
    - data:// URLs
    - blobs
    - URLs malformadas
    """
    if not url or url.startswith('data:'):
        return False
    if url.startswith('blob:'):
        return False
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme in ('http', 'https'):
            return False
        
        # Validar extensión o content-type
        path = parsed.path.lower()
        image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.ico')
        if path and not any(path.endswith(ext) for ext in image_exts):
            # Podría ser una URL sin extensión clara
            # Por ahora la aceptamos, AdvancedImageScraper decidirá
            pass
        return True
    except Exception:
        return False


def get_snapshots(target_url: str, year: int, limit: int = 5, timeout: int = 30) -> List[str]:
    """Consulta la CDX API y devuelve una lista de URLs archivadas para `year`.

    Args:
        target_url: dominio o URL a consultar (ej. example.com)
        year: año (p.ej. 2020)
        limit: máximo de snapshots a devolver
    Returns:
        Lista de URLs tipo https://web.archive.org/web/{timestamp}/{original}
    """
    cdx_endpoint = "http://web.archive.org/cdx/search/cdx"
    params = {
        "url": target_url,
        "from": f"{year}0101",
        "to": f"{year}1231",
        "output": "json",
        "filter": "statuscode:200",
        "collapse": "digest",
        "limit": str(limit),
    }

    try:
        resp = requests.get(cdx_endpoint, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"CDX request failed for {target_url} {year}: {e}")
        return []

    if not data:
        return []

    # CDX JSON returns a header row as first element (fields), detectlo
    header = None
    rows = data
    if isinstance(data[0], list) and any(isinstance(x, str) for x in data[0]):
        header = data[0]
        rows = data[1:]

    # Indices conocidos en CDX: urlkey, timestamp, original, mimetype, statuscode, digest, length
    ts_idx = 1
    orig_idx = 2
    snapshots = []
    for row in rows:
        try:
            timestamp = row[ts_idx]
            original = row[orig_idx]
            snapshots.append(f"https://web.archive.org/web/{timestamp}/{original}")
        except Exception:
            continue

    # mantener orden y unicidad
    seen = set()
    unique = []
    for s in snapshots:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def fetch_snapshot_html(snapshot_url: str, timeout: int = 30) -> Tuple[str, bool]:
    """Descarga el HTML de un snapshot.
    
    Args:
        snapshot_url: URL del snapshot (ej. https://web.archive.org/web/20200101120000/example.com)
        timeout: timeout en segundos
        
    Returns:
        Tupla (contenido_html, éxito)
    """
    try:
        resp = requests.get(snapshot_url, timeout=timeout)
        resp.raise_for_status()
        return resp.text, True
    except Exception as e:
        logger.warning(f"Failed to fetch snapshot {snapshot_url}: {e}")
        return "", False


def get_historical_images(
    target_url: str, 
    year: int, 
    limit: int = 5,
    fetch_html: bool = True
) -> Dict[str, Set[str]]:
    """Obtiene snapshots y extrae URLs de imágenes del HTML histórico.
    
    Args:
        target_url: dominio o URL
        year: año
        limit: máximo snapshots
        fetch_html: si True, descarga y parsea HTML; si False, solo retorna snapshot URLs
        
    Returns:
        Diccionario {snapshot_url: set(image_urls)}
    """
    snapshot_urls = get_snapshots(target_url, year, limit=limit)
    result = {}
    
    if not fetch_html:
        # Solo retornar las URLs de snapshots sin extraer imágenes
        return {url: set() for url in snapshot_urls}
    
    for snapshot_url in snapshot_urls:
        logger.info(f"Extrayendo imágenes de {snapshot_url}")
        html, success = fetch_snapshot_html(snapshot_url)
        
        if success:
            # Extraer URL base (sin timestamp)
            # snapshot_url es: https://web.archive.org/web/20200101120000/example.com/page
            # base_url debe ser: https://example.com/page (sin el /web/timestamp)
            base_url = _reconstruct_base_url(snapshot_url)
            images = extract_images_from_html(html, base_url)
            result[snapshot_url] = images
        else:
            result[snapshot_url] = set()
    
    return result


def _reconstruct_base_url(snapshot_url: str) -> str:
    """Reconstruye la URL base desde un snapshot de Wayback.
    
    De: https://web.archive.org/web/20200101120000/https://example.com/page
    A: https://example.com/page
    """
    pattern = r'https://web\.archive\.org/web/\d+/(.+)'
    match = re.match(pattern, snapshot_url)
    if match:
        return match.group(1)
    return snapshot_url


def load_sites_from_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    sites = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # Saltar encabezado
        for row in reader:
            if not row:
                continue
            sites.append(row[0].strip())
    return sites


def run_wayback_scrape(
    sites: List[str],
    years: List[int],
    output_dir: Path,
    limit_per_year: int = 3,
    max_pages: int = 1,
    delay_between_snapshots: float = 1.0,
    extract_images: bool = True,
):
    """Corre el scraper de Wayback para múltiples sitios y años.
    
    Args:
        sites: lista de sitios/URLs
        years: lista de años
        output_dir: directorio raíz de salida
        limit_per_year: máximo snapshots por año
        max_pages: máximo páginas por snapshot
        delay_between_snapshots: delay entre snapshots en segundos
        extract_images: si True, extrae imágenes del HTML antes de scrapear
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for site in sites:
        site_slug = site.replace("https://", "").replace("http://", "").rstrip("/").replace("/", "_")
        for year in years:
            logger.info(f"Buscando snapshots de {site} en {year}")
            
            if extract_images:
                # Modo mejorado: extrae imágenes del HTML histórico
                snapshot_images = get_historical_images(site, year, limit=limit_per_year)
                
                for snapshot_url, image_urls in snapshot_images.items():
                    logger.info(f"Snapshot {snapshot_url} contiene {len(image_urls)} imágenes")
                    
                    year_dir = output_dir / site_slug / str(year)
                    year_dir.mkdir(parents=True, exist_ok=True)
                    
                    snap_safe = snapshot_url.replace("https://", "").replace("/", "_")
                    scraper_out = year_dir / snap_safe
                    
                    # Guardar lista de imágenes detectadas
                    if image_urls:
                        images_file = scraper_out / "detected_images.json"
                        images_file.parent.mkdir(parents=True, exist_ok=True)
                        with images_file.open("w", encoding="utf-8") as fh:
                            json.dump(list(image_urls), fh, indent=2, ensure_ascii=False)
                        logger.info(f"Guardadas {len(image_urls)} URLs de imágenes en {images_file}")
                    
                    # Scrapear snapshot
                    scraper = AdvancedImageScraper(output_dir=str(scraper_out))
                    try:
                        result = scraper.scrape(snapshot_url, max_pages=max_pages, crawl_site=False)
                        with (scraper_out / "result.json").open("w", encoding="utf-8") as fh:
                            json.dump(result, fh, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.exception(f"Error al scrapear {snapshot_url}: {e}")
                    
                    time.sleep(delay_between_snapshots)
            
            else:
                # Modo original: solo snapshots sin extracción de imágenes
                snaps = get_snapshots(site, year, limit=limit_per_year)
                if not snaps:
                    logger.info(f"No snapshots para {site} {year}")
                    continue

                year_dir = output_dir / site_slug / str(year)
                year_dir.mkdir(parents=True, exist_ok=True)

                with (year_dir / "snapshots.json").open("w", encoding="utf-8") as fh:
                    json.dump(snaps, fh, indent=2, ensure_ascii=False)

                for snap in snaps:
                    logger.info(f"Scrapeando snapshot {snap}")
                    snap_safe = snap.replace("https://", "").replace("/", "_")
                    scraper_out = year_dir / snap_safe
                    scraper = AdvancedImageScraper(output_dir=str(scraper_out))
                    try:
                        result = scraper.scrape(snap, max_pages=max_pages, crawl_site=False)
                        with (scraper_out / "result.json").open("w", encoding="utf-8") as fh:
                            json.dump(result, fh, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.exception(f"Error al scrapear {snap}: {e}")

                    time.sleep(delay_between_snapshots)


def parse_years_arg(years_arg: str) -> List[int]:
    parts = []
    for token in years_arg.split(','):
        token = token.strip()
        if '-' in token:
            a, b = token.split('-', 1)
            parts.extend(range(int(a), int(b) + 1))
        else:
            parts.append(int(token))
    return sorted(set(parts))


def main():
    parser = argparse.ArgumentParser(description="Scrapea snapshots de Wayback Machine usando AdvancedImageScraper")
    parser.add_argument('--sites-file', default='websites-list.csv', help='CSV con sitios (columna 0)')
    parser.add_argument('--sites', nargs='*', help='Lista de sitios/URLs (alternativa a --sites-file)')
    parser.add_argument('--years', default='2020-2025', help='Años o rango de años, p.ej. 2020-2025 o 2020,2022')
    parser.add_argument('--limit', type=int, default=3, help='Máx snapshots por año')
    parser.add_argument('--output-dir', default='./wayback_results', help='Directorio raíz de salida')
    parser.add_argument('--max-pages', type=int, default=1, help='Máx páginas para el scraper (por snapshot)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay entre snapshots')
    parser.add_argument('--no-extract-images', action='store_true', help='No extraer URLs de imágenes del HTML (modo legacy)')
    args = parser.parse_args()

    years = parse_years_arg(args.years)
    sites = []
    if args.sites:
        sites = args.sites
    else:
        csv_sites = load_sites_from_csv(Path(args.sites_file))
        if csv_sites:
            sites = csv_sites

    if not sites:
        logger.error("No sites provided. Use --sites or --sites-file")
        return

    run_wayback_scrape(
        sites=sites, 
        years=years, 
        output_dir=Path(args.output_dir), 
        limit_per_year=args.limit, 
        max_pages=args.max_pages, 
        delay_between_snapshots=args.delay,
        extract_images=not args.no_extract_images
    )


if __name__ == '__main__':
    main()
