#!/usr/bin/env python3
"""Simple image crawler - collect all images with metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import mimetypes
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".svg",
}


@dataclass
class CrawlConfig:
    output_dir: Path
    max_pages: int
    max_depth: int
    delay_seconds: float
    same_domain_only: bool
    timeout_seconds: float


@dataclass
class SeedEntry:
    url: str
    category: str
    subcategory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple image crawler - collect all images with metadata"
    )
    parser.add_argument(
        "--categorized-url-csv",
        type=Path,
        default=None,
        help="CSV with category, subcategory, url columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for images and metadata",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum pages per seed URL",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum crawl depth",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between requests",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP timeout",
    )
    parser.add_argument(
        "--same-domain-only",
        action="store_true",
        help="Only follow same domain links",
    )
    return parser.parse_args()


def load_categorized_seed_entries(
    csv_path: Path,
) -> list[SeedEntry]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    entries: list[SeedEntry] = []
    for row in rows:
        url = row.get("url", "").strip()
        if not url:
            continue

        category = (row.get("category") or "uncategorized").strip() or "uncategorized"
        subcategory = (row.get("subcategory") or "").strip() or "general"
        entries.append(SeedEntry(url=url, category=category, subcategory=subcategory))

    return entries


def ensure_dirs(base: Path) -> dict[str, Path]:
    images_dir = base / "images"
    metadata_dir = base / "metadata"

    for directory in (images_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "images": images_dir,
        "metadata": metadata_dir,
    }


def get_root_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def should_follow_link(
    candidate_url: str,
    current_depth: int,
    config: CrawlConfig,
    allowed_domains: set[str],
) -> bool:
    if current_depth > config.max_depth:
        return False

    parsed = urlparse(candidate_url)
    if parsed.scheme not in {"http", "https"}:
        return False

    if config.same_domain_only and parsed.netloc.lower() not in allowed_domains:
        return False

    return True


def parse_srcset(srcset: str) -> str | None:
    if not srcset:
        return None
    first_entry = srcset.split(",")[0].strip()
    if not first_entry:
        return None
    return first_entry.split(" ")[0].strip()


def extract_images_from_page(page_url: str, soup: BeautifulSoup) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []

    for idx, tag in enumerate(soup.find_all("img")):
        src = (
            tag.get("src")
            or tag.get("data-src")
            or tag.get("data-original")
            or parse_srcset(tag.get("srcset", ""))
        )
        if not src:
            continue

        abs_src = urljoin(page_url, src)
        parsed = urlparse(abs_src)
        if parsed.scheme not in {"http", "https"}:
            continue

        images.append(
            {
                "page_url": page_url,
                "image_url": abs_src,
                "alt": tag.get("alt", ""),
                "title": tag.get("title", ""),
                "img_index": str(idx),
            }
        )

    return images


def extract_links(page_url: str, soup: BeautifulSoup) -> Iterable[str]:
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue
        resolved = urljoin(page_url, href)
        parsed = urlparse(resolved)
        if parsed.scheme in {"http", "https"}:
            yield resolved


def guess_extension(content_type: str, image_url: str) -> str:
    path_ext = Path(urlparse(image_url).path).suffix.lower()
    if path_ext in IMAGE_EXTENSIONS:
        return path_ext

    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    if guessed in IMAGE_EXTENSIONS:
        return guessed

    return ".jpg"


def safe_file_stem(page_url: str, image_url: str, img_index: str) -> str:
    unique = f"{page_url}|{image_url}|{img_index}"
    return hashlib.sha1(unique.encode("utf-8")).hexdigest()


def download_image(
    session: requests.Session,
    image_meta: dict[str, str],
    images_dir: Path,
    timeout_seconds: float,
) -> dict[str, str] | None:
    image_url = image_meta["image_url"]
    try:
        response = session.get(image_url, timeout=timeout_seconds)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    content_type = response.headers.get("Content-Type", "")
    if content_type and not content_type.lower().startswith("image/"):
        return None

    suffix = guess_extension(content_type, image_url)
    file_stem = safe_file_stem(
        page_url=image_meta["page_url"],
        image_url=image_url,
        img_index=image_meta["img_index"],
    )
    file_name = f"{file_stem}{suffix}"
    image_path = images_dir / file_name

    image_path.write_bytes(response.content)

    width, height = "", ""
    try:
        with Image.open(image_path) as img:
            width, height = str(img.width), str(img.height)
    except Exception:
        image_path.unlink(missing_ok=True)
        return None

    image_hash = hashlib.sha256(response.content).hexdigest()

    stored = dict(image_meta)
    stored.update(
        {
            "sha256": image_hash,
            "http_status": str(response.status_code),
            "content_type": content_type,
            "content_length": str(len(response.content)),
            "file_path": str(image_path),
            "filename": file_name,
            "width": width,
            "height": height,
        }
    )
    return stored


def crawl_and_collect(
    seed_urls: list[str],
    config: CrawlConfig,
    session: requests.Session,
) -> list[dict[str, str]]:
    queue: deque[tuple[str, int]] = deque((url, 0) for url in seed_urls)
    visited_pages: set[str] = set()
    collected: list[dict[str, str]] = []

    allowed_domains = {get_root_domain(url) for url in seed_urls}

    while queue and len(visited_pages) < config.max_pages:
        page_url, depth = queue.popleft()
        if page_url in visited_pages:
            continue

        visited_pages.add(page_url)

        try:
            response = session.get(page_url, timeout=config.timeout_seconds)
        except requests.RequestException:
            time.sleep(config.delay_seconds)
            continue

        if response.status_code != 200:
            time.sleep(config.delay_seconds)
            continue

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            time.sleep(config.delay_seconds)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        page_images = extract_images_from_page(page_url, soup)
        collected.extend(page_images)

        next_depth = depth + 1
        if next_depth <= config.max_depth:
            for link_url in extract_links(page_url, soup):
                if should_follow_link(link_url, next_depth, config, allowed_domains):
                    queue.append((link_url, next_depth))

        time.sleep(config.delay_seconds)

    return collected


def main() -> None:
    args = parse_args()

    seed_entries: list[SeedEntry] = []
    if args.categorized_url_csv:
        seed_entries.extend(load_categorized_seed_entries(args.categorized_url_csv))

    if not seed_entries:
        raise RuntimeError("No seed URLs provided via --categorized-url-csv")

    dirs = ensure_dirs(args.output_dir)

    config = CrawlConfig(
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=max(args.delay_seconds, 0.0),
        same_domain_only=args.same_domain_only,
        timeout_seconds=max(args.timeout_seconds, 1.0),
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ImageCrawler/1.0)"})

    print(f"[1/2] Crawling {len(seed_entries)} seed URL(s)...")
    print(f"[2/2] Downloading images...")

    all_images: list[dict[str, str]] = []

    for seed in seed_entries:
        collected_images = crawl_and_collect([seed.url], config, session)

        for image_meta in collected_images:
            image_meta["seed_url"] = seed.url
            image_meta["category"] = seed.category
            image_meta["subcategory"] = seed.subcategory

            stored = download_image(
                session=session,
                image_meta=image_meta,
                images_dir=dirs["images"],
                timeout_seconds=config.timeout_seconds,
            )
            if stored:
                all_images.append(stored)

    # Save metadata
    metadata_csv = dirs["metadata"] / "images.csv"
    if all_images:
        fieldnames = sorted({key for row in all_images for key in row.keys()})
        with metadata_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_images)

    print()
    print("=" * 70)
    print(f"✓ Downloaded: {len(all_images)} images")
    print(f"✓ Storage: {dirs['images']}")
    print(f"✓ Metadata: {metadata_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
