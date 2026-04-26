#!/usr/bin/env python3
"""Crawl webpages, download images, store metadata, and filter ad images."""

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
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

try:
    import clip
    import torch

    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False


AD_TEXT_KEYWORDS = {
    "ad",
    "ads",
    "advert",
    "advertise",
    "advertisement",
    "banner",
    "sponsor",
    "sponsored",
    "promoted",
    "doubleclick",
    "googlesyndication",
    "adservice",
    "taboola",
    "outbrain",
    "criteo",
    "affiliate",
    "tracking",
}

AD_HOST_KEYWORDS = {
    "doubleclick.net",
    "googlesyndication.com",
    "adservice.google.com",
    "googletagmanager.com",
    "taboola.com",
    "outbrain.com",
    "criteo.com",
    "adsrvr.org",
    "adnxs.com",
}

COMMON_BANNER_SIZES = {
    (300, 250),
    (320, 50),
    (336, 280),
    (468, 60),
    (728, 90),
    (970, 90),
    (970, 250),
    (160, 600),
    (300, 600),
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
    ".svg",
}

CLIP_AD_TEXTS = [
    "a web advertisement banner",
    "a sponsored ad image",
    "a marketing display ad",
    "an affiliate promotion image",
]

CLIP_CONTEXT_TEXTS = [
    "a contextual image inside an article",
    "a photo that illustrates webpage content",
    "an editorial image on a website",
    "a product or news image that is part of the content",
]


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
        description="Crawl webpages, download images, and keep only contextual images"
    )
    parser.add_argument(
        "--start-url",
        action="append",
        default=[],
        help="Seed URL. Repeat this argument to pass multiple URLs.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="Optional text file with one URL per line.",
    )
    parser.add_argument(
        "--categorized-url-csv",
        type=Path,
        default=None,
        help="CSV with at least URL + category columns.",
    )
    parser.add_argument(
        "--url-column",
        type=str,
        default="url",
        help="URL column name in --categorized-url-csv.",
    )
    parser.add_argument(
        "--category-column",
        type=str,
        default="category",
        help="Category column name in --categorized-url-csv.",
    )
    parser.add_argument(
        "--subcategory-column",
        type=str,
        default="subcategory",
        help="Subcategory column name in --categorized-url-csv (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for downloaded images and metadata.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum number of pages to visit.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum crawl depth from seed pages.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between page requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--same-domain-only",
        action="store_true",
        help="Only follow links from the same domain as the seed pages.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (compatible; ContextImageCrawler/1.0)",
        help="User agent string for HTTP requests.",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["heuristic", "clip", "hybrid"],
        default="hybrid",
        help="Filtering strategy. hybrid combines heuristics with CLIP when available.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.62,
        help="Ad probability threshold used when filter mode is clip.",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.55,
        help="Combined ad confidence threshold used when filter mode is hybrid.",
    )
    parser.add_argument(
        "--clip-device",
        type=str,
        default="auto",
        help="Device for CLIP model: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--min-contextual-per-category",
        type=int,
        default=0,
        help="Minimum contextual images to collect for each category.",
    )
    parser.add_argument(
        "--dataloader-csv",
        type=Path,
        default=None,
        help="Output CSV path ready for SPAI dataloader. Defaults to metadata/dataloader_contextual.csv",
    )
    return parser.parse_args()


class ClipAdClassifier:
    def __init__(self, device: str = "auto") -> None:
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP dependencies are not available.")

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in {"cpu", "cuda"}:
            resolved_device = device
        else:
            raise ValueError("--clip-device must be one of: auto, cpu, cuda")

        self.device = resolved_device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        with torch.no_grad():
            ad_tokens = clip.tokenize(CLIP_AD_TEXTS).to(self.device)
            contextual_tokens = clip.tokenize(CLIP_CONTEXT_TEXTS).to(self.device)
            self.ad_text_features = self.model.encode_text(ad_tokens)
            self.ctx_text_features = self.model.encode_text(contextual_tokens)
            self.ad_text_features /= self.ad_text_features.norm(dim=-1, keepdim=True)
            self.ctx_text_features /= self.ctx_text_features.norm(dim=-1, keepdim=True)

    def ad_probability(self, image_path: Path) -> float:
        with Image.open(image_path).convert("RGB") as img:
            image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            ad_sim = (img_feat @ self.ad_text_features.T).mean(dim=1)
            ctx_sim = (img_feat @ self.ctx_text_features.T).mean(dim=1)
            logits = torch.stack([ctx_sim, ad_sim], dim=1)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())


def normalize_url(url: str) -> str:
    return url.strip()


def load_seed_urls(url_args: list[str], url_file: Path | None) -> list[str]:
    urls = [normalize_url(u) for u in url_args if normalize_url(u)]

    if url_file is not None:
        lines = url_file.read_text(encoding="utf-8").splitlines()
        urls.extend(normalize_url(line) for line in lines if normalize_url(line))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def load_categorized_seed_entries(
    csv_path: Path,
    url_column: str,
    category_column: str,
    subcategory_column: str,
) -> list[SeedEntry]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    entries: list[SeedEntry] = []
    for idx, row in enumerate(rows):
        if url_column not in row:
            raise RuntimeError(
                f"Missing '{url_column}' column in categorized CSV: {csv_path}"
            )
        if category_column not in row:
            raise RuntimeError(
                f"Missing '{category_column}' column in categorized CSV: {csv_path}"
            )

        url = normalize_url(row.get(url_column, ""))
        if not url:
            continue

        category = (row.get(category_column) or "uncategorized").strip() or "uncategorized"
        subcategory = (row.get(subcategory_column) or "").strip()
        entries.append(SeedEntry(url=url, category=category, subcategory=subcategory))

        if idx == 0 and not urlparse(url).scheme:
            raise RuntimeError(
                f"Invalid URL in categorized CSV ({url_column}): '{url}'. Include http:// or https://"
            )

    return entries


def ensure_dirs(base: Path) -> dict[str, Path]:
    raw_dir = base / "raw_images"
    contextual_dir = base / "contextual_images"
    ads_dir = base / "ads_images"
    metadata_dir = base / "metadata"
    by_category_dir = metadata_dir / "by_category"
    by_subcategory_dir = metadata_dir / "by_subcategory"

    for directory in (raw_dir, contextual_dir, ads_dir, metadata_dir, by_category_dir, by_subcategory_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "raw": raw_dir,
        "contextual": contextual_dir,
        "ads": ads_dir,
        "metadata": metadata_dir,
        "by_category": by_category_dir,
        "by_subcategory": by_subcategory_dir,
    }


def slugify(value: str, fallback: str = "uncategorized") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


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


def text_contains_ad_keywords(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in AD_TEXT_KEYWORDS)


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

        parent = tag.parent
        parent_classes = ""
        parent_id = ""
        if parent is not None:
            parent_classes = " ".join(parent.get("class", []))
            parent_id = parent.get("id", "")

        width = tag.get("width", "")
        height = tag.get("height", "")

        images.append(
            {
                "page_url": page_url,
                "image_url": abs_src,
                "source_attr": src,
                "img_index": str(idx),
                "alt": tag.get("alt", ""),
                "title": tag.get("title", ""),
                "img_class": " ".join(tag.get("class", [])),
                "img_id": tag.get("id", ""),
                "parent_class": parent_classes,
                "parent_id": parent_id,
                "declared_width": width,
                "declared_height": height,
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
    raw_dir: Path,
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
    raw_path = raw_dir / file_name

    raw_path.write_bytes(response.content)

    width, height = "", ""
    try:
        with Image.open(raw_path) as img:
            width, height = str(img.width), str(img.height)
    except Exception:
        raw_path.unlink(missing_ok=True)
        return None

    image_hash = hashlib.sha256(response.content).hexdigest()

    stored = dict(image_meta)
    stored.update(
        {
            "image_sha256": image_hash,
            "http_status": str(response.status_code),
            "content_type": content_type,
            "content_length": str(len(response.content)),
            "stored_path": str(raw_path),
            "stored_filename": file_name,
            "detected_width": width,
            "detected_height": height,
        }
    )
    return stored


def int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def classify_as_ad(image_row: dict[str, str]) -> tuple[bool, int, list[str]]:
    score = 0
    reasons: list[str] = []

    combined_text = " ".join(
        [
            image_row.get("image_url", ""),
            image_row.get("alt", ""),
            image_row.get("title", ""),
            image_row.get("img_class", ""),
            image_row.get("img_id", ""),
            image_row.get("parent_class", ""),
            image_row.get("parent_id", ""),
        ]
    ).lower()

    if text_contains_ad_keywords(combined_text):
        score += 2
        reasons.append("ad_keyword_match")

    host = urlparse(image_row.get("image_url", "")).netloc.lower()
    if any(keyword in host for keyword in AD_HOST_KEYWORDS):
        score += 3
        reasons.append("ad_host_match")

    width = int_or_none(image_row.get("detected_width", ""))
    height = int_or_none(image_row.get("detected_height", ""))
    if width and height:
        if (width, height) in COMMON_BANNER_SIZES:
            score += 3
            reasons.append("common_banner_size")

        ratio = width / max(height, 1)
        if ratio >= 4.0 or ratio <= 0.25:
            score += 1
            reasons.append("extreme_aspect_ratio")

        if width <= 180 and height <= 120:
            score += 1
            reasons.append("tiny_image")

    if re.search(r"([?&](utm_|gclid|fbclid|adid|campaign)=)", image_row.get("image_url", ""), re.IGNORECASE):
        score += 1
        reasons.append("tracking_query_param")

    is_ad = score >= 3
    return is_ad, score, reasons


def classify_with_mode(
    image_row: dict[str, str],
    filter_mode: str,
    clip_classifier: ClipAdClassifier | None,
    clip_threshold: float,
    hybrid_threshold: float,
) -> tuple[bool, float, list[str], float | None]:
    heur_is_ad, heur_score, heur_reasons = classify_as_ad(image_row)
    heur_conf = min(1.0, heur_score / 6.0)

    clip_prob: float | None = None
    if clip_classifier is not None:
        try:
            clip_prob = clip_classifier.ad_probability(Path(image_row["stored_path"]))
        except Exception:
            clip_prob = None

    if filter_mode == "heuristic":
        return heur_is_ad, heur_conf, heur_reasons, clip_prob

    if filter_mode == "clip":
        if clip_prob is None:
            # Hard fallback so the pipeline can still finish if CLIP fails at runtime.
            return heur_is_ad, heur_conf, heur_reasons + ["clip_unavailable_fallback"], clip_prob
        is_ad = clip_prob >= clip_threshold
        reasons = [f"clip_ad_prob={clip_prob:.4f}"]
        return is_ad, clip_prob, reasons, clip_prob

    # hybrid mode
    if clip_prob is None:
        return heur_is_ad, heur_conf, heur_reasons + ["clip_unavailable_fallback"], clip_prob

    combined = 0.6 * heur_conf + 0.4 * clip_prob
    if "ad_host_match" in heur_reasons or "common_banner_size" in heur_reasons:
        combined = max(combined, 0.85)
    is_ad = combined >= hybrid_threshold
    reasons = list(heur_reasons)
    reasons.append(f"clip_ad_prob={clip_prob:.4f}")
    reasons.append(f"combined_ad_conf={combined:.4f}")
    return is_ad, combined, reasons, clip_prob


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataloader_csv(
    contextual_rows: list[dict[str, str]],
    output_dir: Path,
    output_csv: Path,
) -> None:
    entries: list[dict[str, str]] = []
    for row in contextual_rows:
        rel_image = str(Path(row["filtered_path"]).relative_to(output_dir))
        entries.append(
            {
                "image": rel_image,
                "split": "test",
                "class": "1",
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", ""),
                "page_url": row.get("page_url", ""),
                "image_url": row.get("image_url", ""),
            }
        )

    fieldnames = ["image", "split", "class", "category", "subcategory", "page_url", "image_url"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def write_grouped_csvs(rows: list[dict[str, str]], output_dir: Path, prefix: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_name = row.get(prefix, "")
        groups[group_name].append(row)

    for group_name, group_rows in groups.items():
        if prefix == "category":
            out_name = f"{slugify(group_name, 'uncategorized')}.csv"
        else:
            category = slugify(group_rows[0].get("category", "uncategorized"), "uncategorized")
            subcategory = slugify(group_name, "general")
            out_name = f"{category}__{subcategory}.csv"
        write_csv(group_rows, output_dir / out_name)


def crawl_and_collect(seed_urls: list[str], config: CrawlConfig, session: requests.Session) -> list[dict[str, str]]:
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


def split_and_store_images(
    rows: list[dict[str, str]],
    dirs: dict[str, Path],
    filter_mode: str,
    clip_classifier: ClipAdClassifier | None,
    clip_threshold: float,
    hybrid_threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    contextual_rows: list[dict[str, str]] = []
    ad_rows: list[dict[str, str]] = []

    for row in rows:
        is_ad, score, reasons, clip_prob = classify_with_mode(
            image_row=row,
            filter_mode=filter_mode,
            clip_classifier=clip_classifier,
            clip_threshold=clip_threshold,
            hybrid_threshold=hybrid_threshold,
        )
        row["is_ad"] = "1" if is_ad else "0"
        row["ad_score"] = f"{score:.4f}"
        row["filter_mode"] = filter_mode
        row["ad_reasons"] = "|".join(reasons)
        row["clip_ad_probability"] = "" if clip_prob is None else f"{clip_prob:.4f}"

        raw_path = Path(row["stored_path"])
        target_dir = dirs["ads"] if is_ad else dirs["contextual"]
        category = slugify(row.get("category", "uncategorized"), "uncategorized")
        subcategory = slugify(row.get("subcategory", "general"), "general")
        target_dir = target_dir / category / subcategory
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / raw_path.name

        shutil.copy2(raw_path, target_path)
        row["filtered_path"] = str(target_path)

        if is_ad:
            ad_rows.append(row)
        else:
            contextual_rows.append(row)

    return contextual_rows, ad_rows


def collect_downloaded_rows(
    seed_entries: list[SeedEntry],
    config: CrawlConfig,
    session: requests.Session,
    dirs: dict[str, Path],
    filter_mode: str,
    clip_classifier: ClipAdClassifier | None,
    clip_threshold: float,
    hybrid_threshold: float,
    min_contextual_per_category: int,
) -> list[dict[str, str]]:
    downloaded_rows: list[dict[str, str]] = []
    seen_image_urls: set[str] = set()
    contextual_per_category: dict[str, int] = {}

    for seed in seed_entries:
        if min_contextual_per_category > 0 and contextual_per_category.get(seed.category, 0) >= min_contextual_per_category:
            continue

        collected_images = crawl_and_collect([seed.url], config, session)

        for image_meta in collected_images:
            image_url = image_meta["image_url"]
            if image_url in seen_image_urls:
                continue
            seen_image_urls.add(image_url)

            image_meta["seed_url"] = seed.url
            image_meta["category"] = seed.category
            image_meta["subcategory"] = seed.subcategory

            stored = download_image(
                session=session,
                image_meta=image_meta,
                raw_dir=dirs["raw"],
                timeout_seconds=config.timeout_seconds,
            )
            if stored is None:
                continue

            downloaded_rows.append(stored)
            is_ad, _, _, _ = classify_with_mode(
                image_row=stored,
                filter_mode=filter_mode,
                clip_classifier=clip_classifier,
                clip_threshold=clip_threshold,
                hybrid_threshold=hybrid_threshold,
            )
            if not is_ad:
                contextual_per_category[seed.category] = contextual_per_category.get(seed.category, 0) + 1

                if min_contextual_per_category > 0 and contextual_per_category[seed.category] >= min_contextual_per_category:
                    break

    if min_contextual_per_category > 0:
        categories = sorted({s.category for s in seed_entries})
        for category in categories:
            count = contextual_per_category.get(category, 0)
            if count < min_contextual_per_category:
                print(
                    f"Warning: category '{category}' reached {count} contextual images "
                    f"(target={min_contextual_per_category})."
                )

    return downloaded_rows


def main() -> None:
    args = parse_args()

    seed_entries: list[SeedEntry] = []
    if args.categorized_url_csv is not None:
        seed_entries.extend(
            load_categorized_seed_entries(
                csv_path=args.categorized_url_csv,
                url_column=args.url_column,
                category_column=args.category_column,
                subcategory_column=args.subcategory_column,
            )
        )

    seed_urls = load_seed_urls(args.start_url, args.url_file)
    seed_entries.extend(
        [SeedEntry(url=u, category="manual", subcategory="") for u in seed_urls]
    )

    if not seed_entries:
        raise RuntimeError(
            "No seed URLs provided. Use --start-url / --url-file / --categorized-url-csv."
        )

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
    session.headers.update({"User-Agent": args.user_agent})

    clip_classifier: ClipAdClassifier | None = None
    filter_mode = args.filter_mode
    if filter_mode in {"clip", "hybrid"}:
        if CLIP_AVAILABLE:
            print("Initializing CLIP model for visual filtering...")
            clip_classifier = ClipAdClassifier(device=args.clip_device)
        else:
            print("CLIP is not available in this environment. Falling back to heuristic filtering.")
            filter_mode = "heuristic"

    print(f"[1/3] Crawling pages and collecting image references from {len(seed_entries)} seed URL(s)...")
    print("[2/3] Downloading images and storing metadata...")
    downloaded_rows = collect_downloaded_rows(
        seed_entries=seed_entries,
        config=config,
        session=session,
        dirs=dirs,
        filter_mode=filter_mode,
        clip_classifier=clip_classifier,
        clip_threshold=min(max(args.clip_threshold, 0.0), 1.0),
        hybrid_threshold=min(max(args.hybrid_threshold, 0.0), 1.0),
        min_contextual_per_category=max(args.min_contextual_per_category, 0),
    )

    print(f"Stored {len(downloaded_rows)} valid images in {dirs['raw']}")

    print(f"[3/3] Filtering ad images and keeping contextual ones (mode={filter_mode})...")
    contextual_rows, ad_rows = split_and_store_images(
        rows=downloaded_rows,
        dirs=dirs,
        filter_mode=filter_mode,
        clip_classifier=clip_classifier,
        clip_threshold=min(max(args.clip_threshold, 0.0), 1.0),
        hybrid_threshold=min(max(args.hybrid_threshold, 0.0), 1.0),
    )

    all_csv = dirs["metadata"] / "all_images.csv"
    contextual_csv = dirs["metadata"] / "contextual_images.csv"
    ads_csv = dirs["metadata"] / "ads_images.csv"

    write_csv(downloaded_rows, all_csv)
    write_csv(contextual_rows, contextual_csv)
    write_csv(ad_rows, ads_csv)
    write_grouped_csvs(contextual_rows, dirs["by_category"], prefix="category")
    write_grouped_csvs(contextual_rows, dirs["by_subcategory"], prefix="subcategory")

    dataloader_csv = args.dataloader_csv or (dirs["metadata"] / "dataloader_contextual.csv")
    write_dataloader_csv(contextual_rows, args.output_dir, dataloader_csv)

    print("Done.")
    print(f"All images metadata: {all_csv}")
    print(f"Contextual images metadata: {contextual_csv}")
    print(f"Ad images metadata: {ads_csv}")
    print(f"Contextual images folder: {dirs['contextual']}")
    print(f"Ad images folder: {dirs['ads']}")
    print(f"Grouped CSVs by category: {dirs['by_category']}")
    print(f"Grouped CSVs by subcategory: {dirs['by_subcategory']}")
    print(f"Dataloader CSV: {dataloader_csv}")


if __name__ == "__main__":
    main()
