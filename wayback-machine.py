#!/usr/bin/env python3
"""wayback-machine.py

Consulta la Wayback Machine (CDX API) para snapshots por año
e integra con `AdvancedImageScraper` para ejecutar el scrapper
sobre las URLs archivadas.

Primer uso de prueba: años 2020-2025.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import List

import requests

from advanced_image_scraper import AdvancedImageScraper

logger = logging.getLogger("wayback")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for site in sites:
        site_slug = site.replace("https://", "").replace("http://", "").rstrip("/").replace("/", "_")
        for year in years:
            logger.info(f"Buscando snapshots de {site} en {year}")
            snaps = get_snapshots(site, year, limit=limit_per_year)
            if not snaps:
                logger.info(f"No snapshots para {site} {year}")
                continue

            year_dir = output_dir / site_slug / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)

            # Guardar lista de snapshots
            with (year_dir / "snapshots.json").open("w", encoding="utf-8") as fh:
                json.dump(snaps, fh, indent=2, ensure_ascii=False)

            for snap in snaps:
                logger.info(f"Scrapeando snapshot {snap}")
                # cada snapshot tendrá su propio directorio de salida
                snap_safe = snap.replace("https://", "").replace("/", "_")
                scraper_out = year_dir / snap_safe
                scraper = AdvancedImageScraper(output_dir=str(scraper_out))
                try:
                    result = scraper.scrape(snap, max_pages=max_pages, crawl_site=False)
                    # Guardar metadata resumen
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

    run_wayback_scrape(sites=sites, years=years, output_dir=Path(args.output_dir), limit_per_year=args.limit, max_pages=args.max_pages, delay_between_snapshots=args.delay)


if __name__ == '__main__':
    main()
