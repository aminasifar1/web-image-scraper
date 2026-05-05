#!/usr/bin/env python3
"""Exporta metadatos por imagen desde un reporte de descargas.

Sirve para comparar imágenes descargadas por tamaño, dimensiones, hash y huella
canónica, sin tocar el scraper principal.

Uso:
  python export_image_metadata.py \
    --report public_art_images_focused/download_report.csv \
    --output public_art_images_focused/image_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


def read_download_rows(report_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with report_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip() != "downloaded":
                continue
            local_path = (row.get("local_path") or "").strip()
            if not local_path:
                continue
            rows.append(row)
    return rows


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_image_stats(path: Path) -> Tuple[int, int, int, str, str, str]:
    data = path.read_bytes()
    width = height = 0
    image_mode = ""
    image_format = ""
    with Image.open(BytesIO(data)) as img:
        width, height = img.size
        image_mode = img.mode
        image_format = (img.format or "").upper()
    return len(data), width, height, image_mode, image_format, sha1_bytes(data)


def canonical_url_key(url: str) -> str:
    """Clave ligera para agrupar variantes de una misma imagen.

    - Quita querystring.
    - Normaliza secuencias de tamaño comunes.
    - Mantiene el host y la ruta base.
    """
    url = (url or "").strip()
    if not url:
        return ""
    url = url.split("?", 1)[0]
    url = url.split("#", 1)[0]
    url = url.replace("/thumb/", "/thumb/")
    url = url.replace("/normal/", "/normal/")
    return sha1_text(url)


def aspect_ratio(width: int, height: int) -> float:
    return round(width / height, 4) if width and height else 0.0


def size_bucket(bytes_count: int) -> str:
    if bytes_count < 10_000:
        return "tiny"
    if bytes_count < 50_000:
        return "small"
    if bytes_count < 200_000:
        return "medium"
    return "large"


def duplicate_groups(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        counter[row["sha1_bytes"]] += 1
    return {digest: count for digest, count in counter.items() if count > 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta metadatos por imagen desde un CSV de descargas")
    parser.add_argument("--report", required=True, help="CSV de descargas (download_report.csv)")
    parser.add_argument("--output", required=True, help="CSV de metadatos de salida")
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)

    rows = read_download_rows(report_path)
    if not rows:
        raise SystemExit(f"No hay filas descargadas en {report_path}")

    enriched: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, 1):
        local_path = Path(row["local_path"])
        if not local_path.exists():
            continue

        bytes_count, width, height, mode, image_format, sha1_digest = load_image_stats(local_path)
        image_url = (row.get("image_url") or "").strip()
        page_url = (row.get("page_url") or row.get("site_url") or "").strip()

        enriched.append(
            {
                "row_id": str(idx),
                "site": (row.get("site") or "").strip(),
                "site_url": (row.get("site_url") or "").strip(),
                "page_url": page_url,
                "image_url": image_url,
                "local_path": str(local_path),
                "file_name": local_path.name,
                "page_title": (row.get("page_title") or "").strip(),
                "alt_text": (row.get("alt_text") or "").strip(),
                "title_text": (row.get("title_text") or "").strip(),
                "aria_label": (row.get("aria_label") or "").strip(),
                "loading": (row.get("loading") or "").strip(),
                "width_hint": (row.get("width_hint") or "").strip(),
                "height_hint": (row.get("height_hint") or "").strip(),
                "srcset": (row.get("srcset") or "").strip(),
                "duplicate_rule": (row.get("duplicate_rule") or "").strip(),
                "matched_with_url": (row.get("matched_with_url") or "").strip(),
                "bytes": str(bytes_count),
                "width": str(width),
                "height": str(height),
                "aspect_ratio": str(aspect_ratio(width, height)),
                "size_bucket": size_bucket(bytes_count),
                "mode": mode,
                "format": image_format,
                "sha1_bytes": sha1_digest,
                "sha1_url": sha1_text(image_url),
                "canonical_url_key": canonical_url_key(image_url),
                "page_status": (row.get("page_status") or "").strip(),
                "source_status": (row.get("status") or "").strip(),
                "error": (row.get("error") or "").strip(),
            }
        )

    dups = duplicate_groups(enriched)
    for row in enriched:
        row["duplicate_group_size"] = str(dups.get(row["sha1_bytes"], 1))

    fieldnames = [
        "row_id",
        "site",
        "site_url",
        "page_url",
        "page_title",
        "image_url",
        "alt_text",
        "title_text",
        "aria_label",
        "loading",
        "width_hint",
        "height_hint",
        "srcset",
        "duplicate_rule",
        "matched_with_url",
        "local_path",
        "file_name",
        "bytes",
        "width",
        "height",
        "aspect_ratio",
        "size_bucket",
        "mode",
        "format",
        "sha1_bytes",
        "sha1_url",
        "canonical_url_key",
        "duplicate_group_size",
        "page_status",
        "source_status",
        "error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)

    print(f"Metadatos exportados: {len(enriched)} filas -> {output_path}")
    print(f"Duplicados por hash de bytes: {sum(1 for v in dups.values() if v > 1)} grupos")


if __name__ == "__main__":
    main()
