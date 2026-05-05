#!/usr/bin/env python3
"""Ejecuta scraping masivo desde un CSV de websites.

Usa `advanced_image_scraper.py` por cada URL del CSV y guarda:
- resultados por sitio (carpetas separadas)
- logs por sitio
- resumen global `batch_summary.csv`

Ejemplo:
  python run_csv_full_scrape.py \
    --csv websites-list.csv \
    --base-output batch_scrape_results \
    --max-pages 0 \
    --delay 0.8 \
    --crawl-site
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "site"


def site_slug(url: str, org_name: str) -> str:
    host = urlparse(url).netloc.lower().replace(".", "-")
    org = slugify(org_name)
    return f"{host}__{org}"


def read_sites(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if not url:
                continue
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scrape desde CSV")
    parser.add_argument("--csv", default="websites-list.csv", help="CSV con columna url")
    parser.add_argument("--base-output", default="batch_scrape_results", help="Directorio base de resultados")
    parser.add_argument("--max-pages", type=int, default=0, help="Máx páginas por sitio (0 usa límite seguro del scraper)")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay entre requests del scraper")
    parser.add_argument("--crawl-site", action="store_true", help="Activar crawling del sitio completo")
    parser.add_argument("--stop-on-error", action="store_true", help="Detener batch al primer error")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV no encontrado: {csv_path}")

    rows = read_sites(csv_path)
    if not rows:
        raise SystemExit("No hay URLs en el CSV")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.base_output) / f"run_{ts}"
    logs_dir = base_output / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base_output / "batch_summary.csv"
    summary_fields = [
        "idx",
        "organization_name",
        "sector",
        "subsector",
        "url",
        "site_output_dir",
        "log_path",
        "exit_code",
        "elapsed_sec",
        "status",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()

        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            url = (row.get("url") or "").strip()
            org = (row.get("organization_name") or "").strip()
            sector = (row.get("sector") or "").strip()
            subsector = (row.get("subsector") or "").strip()

            slug = site_slug(url, org)
            site_out = base_output / "sites" / f"{idx:03d}_{slug}"
            site_out.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{idx:03d}_{slug}.log"

            cmd = [
                sys.executable,
                "advanced_image_scraper.py",
                "--url",
                url,
                "--output-dir",
                str(site_out),
                "--max-pages",
                str(args.max_pages),
                "--delay",
                str(args.delay),
            ]
            if args.crawl_site:
                cmd.append("--crawl-site")

            print(f"[{idx}/{total}] {org or url}")
            started = time.time()
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write("COMMAND: " + " ".join(cmd) + "\n\n")
                lf.flush()
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

            elapsed = round(time.time() - started, 2)
            status = "ok" if proc.returncode == 0 else "error"
            writer.writerow(
                {
                    "idx": idx,
                    "organization_name": org,
                    "sector": sector,
                    "subsector": subsector,
                    "url": url,
                    "site_output_dir": str(site_out),
                    "log_path": str(log_path),
                    "exit_code": proc.returncode,
                    "elapsed_sec": elapsed,
                    "status": status,
                }
            )
            summary_file.flush()

            print(f"  -> {status} ({elapsed}s)")
            if proc.returncode != 0 and args.stop_on_error:
                print("Se detiene por --stop-on-error")
                break

    print(f"\nBatch finalizado. Resumen: {summary_path}")


if __name__ == "__main__":
    main()
