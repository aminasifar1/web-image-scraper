#!/usr/bin/env python3
"""
Corporate Website Image Crawler for AI-Gen Detection Research
Optimized for crawling and analyzing images from corporate/institutional websites
with sector classification, Wayback Machine historical crawling, and daily reporting.

Usage:
    # Live crawl only
    python corporate_crawler.py \
        --websites-csv websites.csv \
        --output-dir crawl_results \
        --max-images-per-site 50 \
        --delay-seconds 2.0

    # Wayback Machine historical crawl
    python corporate_crawler.py \
        --websites-csv websites.csv \
        --output-dir crawl_results \
        --wayback \
        --wayback-years 2020 2021 2022 2023 2024

CSV format (websites.csv):
    url,sector,subsector,organization_name
    https://example.com,tech,cloud,Example Corp
    https://news.com,media,news,News Outlet
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging with both file and console output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class WebsiteEntry:
    url: str
    sector: str
    subsector: str
    organization_name: str
    crawl_date: str = ""

    @property
    def domain(self) -> str:
        parsed = urlparse(self.url)
        return parsed.netloc.lower()

    @property
    def domain_slug(self) -> str:
        """Safe filename-friendly domain identifier."""
        return self.domain.replace(".", "_").replace("-", "_")


@dataclass
class ImageMetadata:
    """Metadata for each downloaded image."""
    image_url: str
    website_url: str
    organization_name: str
    sector: str
    subsector: str
    stored_filename: str
    stored_path: str
    img_alt_text: str
    img_title: str
    detected_width: int
    detected_height: int
    file_hash: str
    content_type: str
    download_time: str
    crawl_date: str
    # Wayback-specific fields
    source: str = "live"          # "live" | "wayback"
    wayback_year: Optional[int] = None
    wayback_timestamp: str = ""   # e.g. "20220315123045"
    original_url: str = ""        # original URL before Wayback rewrite
    page_url: str = ""            # page from which this image was extracted

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# WAYBACK MACHINE CLIENT
# ============================================================================

class WaybackClient:
    """
    Minimal client for the Wayback Machine CDX and Availability APIs.

    Availability API  → get the closest snapshot for a URL + timestamp
    CDX API           → list all snapshots in a year for a URL
    """

    CDX_API = "http://web.archive.org/cdx/search/cdx"
    AVAIL_API = "https://archive.org/wayback/available"
    WB_PREFIX = "https://web.archive.org/web/"

    def __init__(self, session: requests.Session, logger: logging.Logger,
                 timeout: float = 20.0, delay: float = 1.5):
        self.session = session
        self.logger = logger
        self.timeout = timeout
        self.delay = delay

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_snapshot_url(self, url: str, timestamp: str) -> Optional[str]:
        """
        Return the closest Wayback snapshot URL for *url* near *timestamp*
        (format: YYYYMMDD or YYYYMMDDHHmmss).
        Returns None if no snapshot exists.
        """
        params = {"url": url, "timestamp": timestamp}
        try:
            r = self.session.get(
                self.AVAIL_API,
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            snapshot = data.get("archived_snapshots", {}).get("closest", {})
            if snapshot.get("available"):
                return snapshot["url"]
        except Exception as e:
            self.logger.debug(f"Wayback availability check failed for {url}: {e}")
        return None

    def get_year_snapshots(
        self,
        url: str,
        year: int,
        max_snapshots: int = 3,
    ) -> list[dict]:
        """
        Return up to *max_snapshots* CDX records for *url* within *year*,
        spread across the year (first, middle, last available).

        Each record: {"timestamp": str, "wayback_url": str, "original": str}
        """
        params = {
            "url": url,
            "output": "json",
            "from": f"{year}0101",
            "to": f"{year}1231",
            "fl": "timestamp,original",
            "collapse": "timestamp:6",   # one per month at most
            "limit": "100",
        }
        try:
            r = self.session.get(self.CDX_API, params=params, timeout=self.timeout)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            self.logger.debug(f"CDX query failed for {url} ({year}): {e}")
            return []

        if not rows or len(rows) < 2:
            # rows[0] is the header line ["timestamp", "original"]
            return []

        records = rows[1:]  # skip header
        if not records:
            return []

        # Pick spread: first, middle, last
        indices = self._spread_indices(len(records), max_snapshots)
        result = []
        for i in indices:
            ts, orig = records[i][0], records[i][1]
            result.append({
                "timestamp": ts,
                "wayback_url": f"{self.WB_PREFIX}{ts}/{orig}",
                "original": orig,
            })
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spread_indices(n: int, k: int) -> list[int]:
        """Return k evenly-spread indices in range [0, n)."""
        if n <= 0 or k <= 0:
            return []
        if n <= k:
            return list(range(n))

        # Include both ends (0 and n-1) and spread interior points.
        if k == 1:
            return [0]
        step = (n - 1) / (k - 1)
        idxs = [int(round(step * i)) for i in range(k)]

        # Keep order and uniqueness if rounding collides.
        ordered_unique = list(dict.fromkeys(idxs))
        if len(ordered_unique) < k:
            for i in range(n):
                if i not in ordered_unique:
                    ordered_unique.append(i)
                if len(ordered_unique) == k:
                    break
        return sorted(ordered_unique[:k])

    def rewrite_image_url(self, wayback_page_url: str, img_src: str) -> str:
        """
        Given a Wayback page URL and a relative or absolute img src,
        return a Wayback-rewritten URL so the image is served from
        the archive rather than the live site.

        Wayback rewrites relative URLs automatically in HTML, but when
        we reconstruct URLs ourselves we need to do it manually.
        """
        if img_src.startswith(self.WB_PREFIX):
            return img_src

        # Extract timestamp from the page URL
        # e.g. https://web.archive.org/web/20220315123045/https://example.com/
        ts = ""
        if "/web/" in wayback_page_url:
            try:
                after_web = wayback_page_url.split("/web/", 1)[1]
                ts = after_web.split("/")[0]
            except IndexError:
                pass

        if not ts:
            return img_src  # fallback: try live URL

        # Resolve img_src against the original (non-Wayback) base
        try:
            original_base = wayback_page_url.split("/web/", 1)[1]
            original_base = original_base[len(ts):].lstrip("/")
            if not original_base.startswith("http"):
                original_base = "https://" + original_base
            resolved = urljoin(original_base, img_src)
        except Exception:
            resolved = img_src

        return f"{self.WB_PREFIX}{ts}if_/{resolved}"


# ============================================================================
# CRAWLER OPERATIONS
# ============================================================================

class CorporateCrawler:
    """Crawler optimized for corporate website image collection."""

    DEFAULT_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    ]

    # Common ad keywords to filter out
    AD_KEYWORDS = {
        "ad", "ads", "advert", "banner", "sponsored", "promotion",
        "doubleclick", "googlesyndication", "taboola", "outbrain",
        "adsystem", "adserver", "adclick", "adslot",
    }

    # Banner sizes (width, height)
    BANNER_SIZES = {
        (300, 250), (320, 50), (468, 60), (728, 90),
        (970, 90), (970, 250), (160, 600), (300, 600),
    }

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    NON_HTML_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico",
        ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz",
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".json", ".xml", ".rss", ".atom",
    }
    ENUM_FILENAME_PATTERN = re.compile(r"^image_(\d+)\.[a-z0-9]+$", re.IGNORECASE)

    UI_ASSET_KEYWORDS = {
        "icon", "favicon", "sprite", "avatar",
        "emoji", "badge", "loader", "placeholder",
        "sticker", "emote", "reaction", "stamp",
        "logo", "menu", "share", "newsletter",
        "cookie", "consent", "close", "search",
        "header", "footer", "nav", "button",
    }

    NON_INFORMATIVE_ALT = {
        "image", "img", "photo", "picture", "thumbnail", "thumb", "logo",
        "icon", "banner", "advertisement", "ad",
    }

    VIDEO_THUMBNAIL_KEYWORDS = {
        "video", "thumbnail", "thumb", "poster", "trailer",
        "preview", "playbutton", "play-button", "cover_video",
        "hqdefault", "mqdefault", "maxresdefault", "sddefault",
        "youtube", "ytimg", "vimeo", "dailymotion", "wistia",
        "jwplayer", "brightcove",
    }

    # Page priority keywords: prefer content-rich sections
    PRIORITY_PATH_KEYWORDS = {
        "news", "blog", "press", "media", "gallery", "about",
        "story", "article", "report", "publication", "insight",
    }

    @staticmethod
    def _parse_srcset_first(srcset: str) -> str:
        if not srcset:
            return ""
        first = srcset.split(",")[0].strip()
        if not first:
            return ""
        return first.split(" ")[0].strip()

    @staticmethod
    def _parse_srcset_best(srcset: str) -> str:
        """Return the highest-resolution URL from a srcset attribute."""
        if not srcset:
            return ""
        best_url = ""
        best_w = -1
        for part in srcset.split(","):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            url = tokens[0]
            w = 0
            if len(tokens) > 1:
                descriptor = tokens[1]
                try:
                    if descriptor.endswith("w"):
                        w = int(descriptor[:-1])
                    elif descriptor.endswith("x"):
                        w = int(float(descriptor[:-1]) * 500)
                except ValueError:
                    pass
            if w > best_w:
                best_w = w
                best_url = url
        return best_url

    def __init__(
        self,
        output_dir: Path,
        max_images_per_site: int = 50,
        max_pages_per_site: int = 10,
        delay_seconds: float = 2.0,
        timeout_seconds: float = 15.0,
        min_width: int = 180,
        min_height: int = 180,
        min_area: int = 50000,
        min_bytes: int = 4000,
        min_aspect_ratio: float = 0.33,
        max_aspect_ratio: float = 3.0,
        organize_by_sector_site: bool = False,
        reject_sticker_like: bool = False,
        sticker_max_transparent_ratio: float = 0.35,
        sticker_max_content_box_ratio: float = 0.70,
        reject_video_thumbnails: bool = True,
        drop_duplicate_content: bool = True,
        drop_near_duplicates: bool = False,
        near_duplicate_hash_size: int = 16,
        near_duplicate_hamming_threshold: int = 8,
        # Wayback options
        wayback_enabled: bool = False,
        wayback_years: Optional[list[int]] = None,
        wayback_snapshots_per_year: int = 6,
        wayback_delay: float = 2.0,
        render_js: bool = False,
        scroll_steps: int = 3,
        scroll_delay: float = 0.6,
        playwright_timeout: float = 20.0,
        ignore_robots: bool = False,
        rotate_user_agent: bool = False,
        user_agent: str = "",
        cookie_json: Optional[Path] = None,
        require_alt_text: bool = False,
        min_alt_text_chars: int = 8,
        logger: logging.Logger = None,
    ):
        self.output_dir = output_dir
        self.max_images_per_site = max_images_per_site
        self.max_pages_per_site = max_pages_per_site
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.min_bytes = min_bytes
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.organize_by_sector_site = organize_by_sector_site
        self.reject_sticker_like = reject_sticker_like
        self.sticker_max_transparent_ratio = sticker_max_transparent_ratio
        self.sticker_max_content_box_ratio = sticker_max_content_box_ratio
        self.reject_video_thumbnails = reject_video_thumbnails
        self.drop_duplicate_content = drop_duplicate_content
        self.drop_near_duplicates = drop_near_duplicates
        self.near_duplicate_hash_size = max(8, near_duplicate_hash_size)
        self.near_duplicate_hamming_threshold = max(0, near_duplicate_hamming_threshold)
        self.wayback_enabled = wayback_enabled
        self.wayback_years = wayback_years or []
        self.wayback_snapshots_per_year = wayback_snapshots_per_year
        self.render_js = render_js
        self.scroll_steps = max(0, int(scroll_steps))
        self.scroll_delay = max(0.0, float(scroll_delay))
        self.playwright_timeout = max(1.0, float(playwright_timeout))
        self.ignore_robots = ignore_robots
        self.rotate_user_agent = rotate_user_agent
        self.user_agent = user_agent.strip() or self.DEFAULT_USER_AGENTS[0]
        self.cookie_json = cookie_json
        self.require_alt_text = require_alt_text
        self.min_alt_text_chars = max(0, int(min_alt_text_chars))
        self.logger = logger or logging.getLogger(__name__)

        # Setup directories
        self.images_dir = output_dir / "images"
        self.metadata_dir = output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Session with user agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        if self.cookie_json is not None:
            self._load_cookies(self.cookie_json)

        # Wayback client
        self.wayback = WaybackClient(
            session=self.session,
            logger=self.logger,
            timeout=timeout_seconds,
            delay=wayback_delay,
        )

        # Track seen byte hashes for statistics; do not drop occurrences
        # because repeated images across sites/years are useful evidence.
        self._downloaded_hashes: set[str] = set()
        self._near_duplicate_hashes: list[list[int]] = []
        self._next_image_index = 1
        self._robots_cache: dict[str, robotparser.RobotFileParser] = {}

        self._playwright = None
        self._pw_browser = None
        self._pw_context = None
        self._pw_page = None

        # Load prior state so repeated executions on the same output dir only add new images.
        self._load_existing_state()

        # Statistics
        self.stats = {
            "websites_attempted": 0,
            "websites_successful": 0,
            "pages_crawled": 0,
            "images_found": 0,
            "images_downloaded": 0,
            "images_skipped_duplicate": 0,
            "images_filtered_ads": 0,
            "images_filtered_ui": 0,
            "images_filtered_quality": 0,
            "images_filtered_sticker": 0,
            "images_filtered_video": 0,
            "images_filtered_near_duplicate": 0,
            "images_filtered_missing_alt": 0,
            "pages_skipped_robots": 0,
            "wayback_snapshots_crawled": 0,
            "start_time": datetime.now(),
        }

    def _pick_user_agent(self) -> str:
        if self.rotate_user_agent:
            return random.choice(self.DEFAULT_USER_AGENTS)
        return self.user_agent

    def _load_cookies(self, cookie_path: Path) -> None:
        try:
            payload = json.loads(cookie_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning(f"Failed to load cookie JSON '{cookie_path}': {exc}")
            return

        if isinstance(payload, dict):
            items = [{"name": k, "value": str(v)} for k, v in payload.items()]
            # Dict payload is treated as global cookies and attached to all requests.
            cookie_header = "; ".join(f"{k}={v}" for k, v in payload.items())
            if cookie_header:
                self.session.headers.update({"Cookie": cookie_header})
                self.logger.info(
                    "Applied global Cookie header from JSON dict to all requests"
                )
        elif isinstance(payload, list):
            items = payload
        else:
            self.logger.warning("Cookie JSON must be a dict or list. Ignoring cookies.")
            return

        loaded = 0
        for cookie in items:
            if not isinstance(cookie, dict):
                continue
            name = cookie.get("name")
            value = cookie.get("value")
            if not name or value is None:
                continue
            set_kwargs = {
                "name": str(name),
                "value": str(value),
                "path": str(cookie.get("path") or "/"),
            }
            domain = cookie.get("domain")
            if domain:
                set_kwargs["domain"] = str(domain)
            self.session.cookies.set(**set_kwargs)
            loaded += 1
        self.logger.info(f"Loaded {loaded} cookies from {cookie_path}")

    def _is_allowed_by_robots(self, url: str) -> bool:
        if self.ignore_robots:
            return True

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            return True

        if domain not in self._robots_cache:
            robots_url = f"{parsed.scheme or 'https'}://{domain}/robots.txt"
            rp = robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                return True
            self._robots_cache[domain] = rp

        try:
            return bool(self._robots_cache[domain].can_fetch(self._pick_user_agent(), url))
        except Exception:
            return True

    def _ensure_playwright(self) -> bool:
        if self._pw_page is not None:
            return True
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            self.logger.warning(f"Playwright not available; falling back to requests HTML fetch: {exc}")
            return False

        try:
            self._playwright = sync_playwright().start()
            self._pw_browser = self._playwright.chromium.launch(headless=True)
            self._pw_context = self._pw_browser.new_context(user_agent=self._pick_user_agent())
            self._pw_page = self._pw_context.new_page()
            return True
        except Exception as exc:
            self.logger.warning(f"Failed to initialize Playwright; falling back to requests: {exc}")
            return False

    def _close_playwright(self) -> None:
        for handle in (self._pw_page, self._pw_context, self._pw_browser):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._pw_page = None
        self._pw_context = None
        self._pw_browser = None
        self._playwright = None

    def _fetch_page_html(self, page_url: str, source: str) -> str:
        if source == "live" and self.render_js and self._ensure_playwright():
            try:
                timeout_ms = int(self.playwright_timeout * 1000)
                self._pw_page.set_extra_http_headers({"User-Agent": self._pick_user_agent()})
                self._pw_page.goto(page_url, timeout=timeout_ms, wait_until="domcontentloaded")
                for _ in range(self.scroll_steps):
                    self._pw_page.evaluate("window.scrollBy(0, window.innerHeight)")
                    self._pw_page.wait_for_timeout(int(self.scroll_delay * 1000))
                return self._pw_page.content()
            except Exception as exc:
                self.logger.debug(f"Playwright fetch failed for {page_url}: {exc}")

        headers = {"User-Agent": self._pick_user_agent(), "Referer": page_url}
        response = self.session.get(page_url, timeout=self.timeout_seconds, headers=headers)
        response.raise_for_status()
        return response.text

    def _load_existing_state(self) -> None:
        """Load existing hashes and image index to support incremental crawls."""
        max_idx = 0

        for img_path in self.images_dir.rglob("*"):
            if not img_path.is_file():
                continue
            match = self.ENUM_FILENAME_PATTERN.match(img_path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

        metadata_files = list(self.metadata_dir.glob("*.json"))
        for metadata_file in metadata_files:
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    row = json.load(f)
            except Exception:
                continue

            file_hash = row.get("file_hash", "")
            if file_hash:
                self._downloaded_hashes.add(file_hash)

        self._next_image_index = max_idx + 1

        if self._downloaded_hashes:
            self.logger.info(
                "Loaded %d existing images from previous runs. Next image index: %d",
                len(self._downloaded_hashes),
                self._next_image_index,
            )

    def _next_enumerated_filename(self, ext: str) -> str:
        """Return next sequential image filename, e.g., image_00000001.jpg."""
        filename = f"image_{self._next_image_index:08d}{ext}"
        self._next_image_index += 1
        return filename

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contains_ui_keyword(text: str) -> bool:
        lowered = text.lower()
        return any(k in lowered for k in CorporateCrawler.UI_ASSET_KEYWORDS)

    def is_likely_ui_asset(self, image_row: dict) -> bool:
        """Filter obvious UI assets such as icons/sprites based on metadata."""
        url_text = image_row.get("url", "")
        combined = " ".join([
            url_text,
            image_row.get("alt", ""),
            image_row.get("title", ""),
            image_row.get("img_class", ""),
            image_row.get("img_id", ""),
            image_row.get("parent_class", ""),
            image_row.get("parent_id", ""),
        ])
        if self._contains_ui_keyword(combined):
            return True
        path = urlparse(url_text).path.lower()
        if path.endswith(".svg"):
            return True
        if any(token in path for token in ["/icons/", "/sprites/", "/logos/"]):
            return True
        return False

    def _text_is_meaningful(self, text: str) -> bool:
        cleaned = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if len(cleaned) < self.min_alt_text_chars:
            return False
        if cleaned in self.NON_INFORMATIVE_ALT:
            return False
        return True

    def _extract_image_semantics(self, img_tag) -> tuple[str, str]:
        alt = (img_tag.get("alt") or "").strip()
        title = (img_tag.get("title") or "").strip()

        if self._text_is_meaningful(alt):
            return alt, title

        aria = (img_tag.get("aria-label") or "").strip()
        if self._text_is_meaningful(aria):
            return aria, title

        figure = img_tag.find_parent("figure")
        if figure is not None:
            caption = figure.find("figcaption")
            if caption is not None:
                cap_text = " ".join(caption.get_text(" ", strip=True).split())
                if self._text_is_meaningful(cap_text):
                    return cap_text, title

        parent = img_tag.parent
        if parent is not None:
            parent_text = " ".join(parent.get_text(" ", strip=True).split())
            if self._text_is_meaningful(parent_text):
                return parent_text[:220], title

        return alt, title

    def passes_quality_filters(
        self,
        width: int,
        height: int,
        content_type: str,
        content_len: int,
    ) -> bool:
        if content_type.startswith("image/svg"):
            return False
        if width < self.min_width or height < self.min_height:
            return False
        if width * height < self.min_area:
            return False
        ratio = width / max(height, 1)
        if ratio < self.min_aspect_ratio or ratio > self.max_aspect_ratio:
            return False
        if content_len < self.min_bytes:
            return False
        return True

    @staticmethod
    def _slugify_path_component(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip().lower())
        cleaned = cleaned.strip("._-")
        return cleaned or fallback

    def _build_image_output_path(
        self,
        website: WebsiteEntry,
        source: str,
        wayback_year: Optional[int],
        filename: str,
    ) -> Path:
        if not self.organize_by_sector_site:
            return self.images_dir / filename

        sector = self._slugify_path_component(website.sector, "unknown_sector")
        site = self._slugify_path_component(website.domain, "unknown_site")

        if source == "wayback" and wayback_year is not None:
            return self.images_dir / sector / site / "wayback" / str(wayback_year) / filename
        return self.images_dir / sector / site / "live" / filename

    def is_sticker_like(self, pil_img: Image.Image, content_type: str) -> bool:
        """
        Heuristic for sticker-like assets: mostly transparent PNG/WebP with
        small opaque content box relative to full canvas.
        """
        ct = (content_type or "").lower()
        if "png" not in ct and "webp" not in ct:
            return False
        bands = pil_img.getbands()
        if "A" not in bands:
            return False

        alpha = pil_img.getchannel("A")
        hist = alpha.histogram()
        total = sum(hist)
        if total <= 0:
            return True

        transparent = sum(hist[:8])
        transparent_ratio = transparent / total

        mask = alpha.point(lambda p: 255 if p > 20 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return True

        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        box_ratio = (bw * bh) / max(1, pil_img.width * pil_img.height)

        touches_edge = (
            bbox[0] <= 2
            or bbox[1] <= 2
            or bbox[2] >= pil_img.width - 2
            or bbox[3] >= pil_img.height - 2
        )

        return (
            transparent_ratio >= self.sticker_max_transparent_ratio
            and box_ratio <= self.sticker_max_content_box_ratio
            and not touches_edge
        )

    @staticmethod
    def _contains_video_keyword(text: str) -> bool:
        lowered = text.lower()
        return any(k in lowered for k in CorporateCrawler.VIDEO_THUMBNAIL_KEYWORDS)

    def is_likely_video_thumbnail(self, image_row: dict) -> bool:
        combined = " ".join(
            [
                image_row.get("url", ""),
                image_row.get("alt", ""),
                image_row.get("title", ""),
                image_row.get("img_class", ""),
                image_row.get("img_id", ""),
                image_row.get("parent_class", ""),
                image_row.get("parent_id", ""),
                image_row.get("page_url", ""),
            ]
        )
        if self._contains_video_keyword(combined):
            return True
        path = urlparse(image_row.get("url", "")).path.lower()
        if re.search(r"(^|[/_-])(hqdefault|maxresdefault|mqdefault|sddefault|poster|thumbnail|thumb)($|[._-])", path):
            return True
        return False

    def is_video_page(self, page_url: str, soup: BeautifulSoup) -> bool:
        if self._contains_video_keyword(page_url):
            return True
        for tag in soup.find_all("meta"):
            prop = (tag.get("property") or "").lower()
            name = (tag.get("name") or "").lower()
            content = (tag.get("content") or "").lower()
            if prop == "og:type" and "video" in content:
                return True
            if name in {"twitter:card", "twitter:player"} and "player" in content:
                return True
        return False

    def _compute_average_hash(self, pil_img: Image.Image) -> int:
        """Compute perceptual average hash as an integer bitset."""
        size = self.near_duplicate_hash_size
        gray = pil_img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
        avg = sum(pixels) / max(1, len(pixels))
        bits = 0
        for i, px in enumerate(pixels):
            if px >= avg:
                bits |= (1 << i)
        return bits

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _trim_uniform_border(self, pil_img: Image.Image) -> Image.Image:
        """Trim near-uniform background border to normalize padded logos/crops."""
        gray = pil_img.convert("L")
        w, h = gray.size
        if w < 8 or h < 8:
            return pil_img

        px = gray.load()
        border_samples = []
        # top/bottom rows
        for x in range(w):
            border_samples.append(px[x, 0])
            border_samples.append(px[x, h - 1])
        # left/right cols
        for y in range(h):
            border_samples.append(px[0, y])
            border_samples.append(px[w - 1, y])

        border_samples.sort()
        bg = border_samples[len(border_samples) // 2]
        tol = 18

        # Build foreground mask by deviation from border color.
        mask = gray.point(lambda v: 255 if abs(v - bg) > tol else 0)
        bbox = mask.getbbox()
        if not bbox:
            return pil_img

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw < 8 or bh < 8:
            return pil_img

        return pil_img.crop(bbox)

    def _compute_crop_aware_hashes(self, pil_img: Image.Image) -> list[int]:
        """Compute multiple perceptual hashes to detect near-duplicates with crops."""
        trimmed = self._trim_uniform_border(pil_img)
        w, h = pil_img.size
        if w < 8 or h < 8:
            return [self._compute_average_hash(pil_img)]

        # Full image + overlapping crops to tolerate webpage recrops/reframes.
        w75 = (3 * w) // 4
        h75 = (3 * h) // 4
        x25 = w // 4
        y25 = h // 4
        boxes = [
            (0, 0, w, h),  # full
            (0, 0, w // 2, h),  # left half
            (w // 2, 0, w, h),  # right half
            (0, 0, w, h // 2),  # top half
            (0, h // 2, w, h),  # bottom half
            (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4),  # center crop
            (0, 0, w75, h),  # left 75%
            (x25, 0, w, h),  # right 75%
            (0, 0, w, h75),  # top 75%
            (0, y25, w, h),  # bottom 75%
            (0, 0, w75, h75),  # top-left 75%
            (x25, 0, w, h75),  # top-right 75%
            (0, y25, w75, h),  # bottom-left 75%
            (x25, y25, w, h),  # bottom-right 75%
        ]

        hashes: list[int] = [self._compute_average_hash(trimmed)]
        for x1, y1, x2, y2 in boxes:
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            hashes.append(self._compute_average_hash(crop))
            hashes.append(self._compute_average_hash(self._trim_uniform_border(crop)))

        # Keep order but remove duplicates.
        return list(dict.fromkeys(hashes)) or [self._compute_average_hash(pil_img)]

    def is_near_duplicate(self, pil_img: Image.Image) -> bool:
        """Check if image is perceptually near-duplicate of previous accepted images."""
        candidate_hashes = self._compute_crop_aware_hashes(pil_img)
        thr = self.near_duplicate_hamming_threshold

        for prev_hashes in self._near_duplicate_hashes:
            for h1 in candidate_hashes:
                for h2 in prev_hashes:
                    if self._hamming_distance(h1, h2) <= thr:
                        return True

        self._near_duplicate_hashes.append(candidate_hashes)
        return False

    def is_likely_ad(self, img_tag) -> bool:
        """Quick heuristic to filter obvious ads."""
        text_fields = [
            img_tag.get("alt", ""),
            img_tag.get("title", ""),
            " ".join(img_tag.get("class", [])),
            img_tag.get("id", ""),
        ]
        combined = " ".join(text_fields).lower()
        if any(kw in combined for kw in self.AD_KEYWORDS):
            return True
        # Also check parent element
        parent = img_tag.parent
        if parent:
            parent_combined = " ".join([
                " ".join(parent.get("class", [])),
                parent.get("id", ""),
            ]).lower()
            if any(kw in parent_combined for kw in self.AD_KEYWORDS):
                return True
        # Size check
        try:
            w = int(img_tag.get("width", 0))
            h = int(img_tag.get("height", 0))
            if (w, h) in self.BANNER_SIZES:
                return True
        except (ValueError, TypeError):
            pass
        return False

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def load_websites(self, csv_path: Path) -> list[WebsiteEntry]:
        """Load website list from CSV."""
        websites = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = WebsiteEntry(
                    url=row["url"].strip(),
                    sector=row.get("sector", "unknown").strip(),
                    subsector=row.get("subsector", "unknown").strip(),
                    organization_name=row.get("organization_name", "").strip(),
                    crawl_date=datetime.now().strftime("%Y-%m-%d"),
                )
                websites.append(entry)
        self.logger.info(f"Loaded {len(websites)} websites from {csv_path}")
        return websites

    # ------------------------------------------------------------------
    # HTML image extraction
    # ------------------------------------------------------------------

    def extract_image_urls(self, page_url: str, html: str,
                           wayback_mode: bool = False) -> list[dict]:
        """
        Extract image URLs and metadata from HTML.

        When *wayback_mode* is True, resolves URLs through the Wayback
        rewriter so images are fetched from the archive.
        """
        images = []
        seen_urls: set[str] = set()
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            self.logger.warning(f"Failed to parse HTML from {page_url}: {e}")
            return images

        page_is_video = self.is_video_page(page_url, soup)

        # ── 1. Standard <img> tags ──────────────────────────────────────
        for idx, img_tag in enumerate(soup.find_all("img")):
            if self.is_likely_ad(img_tag):
                self.stats["images_filtered_ads"] += 1
                continue

            # Prefer highest-resolution source
            src = (
                self._parse_srcset_best(img_tag.get("srcset", ""))
                or img_tag.get("src")
                or img_tag.get("data-src")
                or img_tag.get("data-original")
                or img_tag.get("data-lazy-src")
                or img_tag.get("data-lazy")
                or ""
            )
            if not src:
                continue

            img_url = self._resolve_url(page_url, src, wayback_mode)
            if not img_url or not img_url.startswith(("http://", "https://")):
                continue
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            semantic_alt, semantic_title = self._extract_image_semantics(img_tag)

            image_row = {
                "url": img_url,
                "alt": semantic_alt,
                "title": semantic_title,
                "img_class": " ".join(img_tag.get("class", [])),
                "img_id": img_tag.get("id", ""),
                "parent_class": " ".join(
                    (img_tag.parent.get("class", []) if img_tag.parent else [])
                ),
                "parent_id": (
                    img_tag.parent.get("id", "") if img_tag.parent else ""
                ),
                "page_url": page_url,
                "index": idx,
            }

            if self.is_likely_ui_asset(image_row):
                self.stats["images_filtered_ui"] += 1
                continue

            if self.require_alt_text and not self._text_is_meaningful(image_row.get("alt", "")):
                self.stats["images_filtered_missing_alt"] += 1
                continue

            if self.reject_video_thumbnails and self.is_likely_video_thumbnail(image_row):
                self.stats["images_filtered_video"] += 1
                continue

            images.append(image_row)

        # ── 2. CSS background-image in inline style attributes ──────────
        for tag in soup.find_all(style=True):
            style_val = tag.get("style", "")
            bg_url = self._extract_css_bg_url(style_val)
            if not bg_url:
                continue
            img_url = self._resolve_url(page_url, bg_url, wayback_mode)
            if not img_url or not img_url.startswith(("http://", "https://")):
                continue
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)
            image_row = {
                "url": img_url,
                "alt": tag.get("aria-label", ""),
                "title": "",
                "img_class": " ".join(tag.get("class", [])),
                "img_id": tag.get("id", ""),
                "parent_class": "",
                "parent_id": "",
                "page_url": page_url,
                "index": len(images),
            }
            if self.is_likely_ui_asset(image_row):
                self.stats["images_filtered_ui"] += 1
                continue
            if self.reject_video_thumbnails and self.is_likely_video_thumbnail(image_row):
                self.stats["images_filtered_video"] += 1
                continue
            images.append(image_row)

        # ── 3. Open Graph / Twitter meta tags ───────────────────────────
        meta_selectors = [
            ("property", "og:image"),
            ("property", "og:image:url"),
            ("name", "twitter:image"),
            ("name", "twitter:image:src"),
        ]
        for attr_name, attr_value in meta_selectors:
            for tag in soup.find_all("meta", attrs={attr_name: attr_value}):
                if self.reject_video_thumbnails and page_is_video:
                    self.stats["images_filtered_video"] += 1
                    continue
                content = (tag.get("content") or "").strip()
                if not content:
                    continue
                img_url = self._resolve_url(page_url, content, wayback_mode)
                if not img_url or not img_url.startswith(("http://", "https://")):
                    continue
                if img_url in seen_urls:
                    continue
                seen_urls.add(img_url)
                image_row = {
                    "url": img_url,
                    "alt": "",
                    "title": f"meta:{attr_value}",
                    "img_class": "",
                    "img_id": "",
                    "parent_class": "",
                    "parent_id": "",
                    "page_url": page_url,
                    "index": len(images),
                }
                if self.is_likely_ui_asset(image_row):
                    self.stats["images_filtered_ui"] += 1
                    continue
                if self.reject_video_thumbnails and self.is_likely_video_thumbnail(image_row):
                    self.stats["images_filtered_video"] += 1
                    continue
                images.append(image_row)

        # ── 4. <picture> / <source> tags ────────────────────────────────
        for source_tag in soup.find_all("source"):
            src = (
                self._parse_srcset_best(source_tag.get("srcset", ""))
                or source_tag.get("src", "")
            )
            if not src:
                continue
            img_url = self._resolve_url(page_url, src, wayback_mode)
            if not img_url or not img_url.startswith(("http://", "https://")):
                continue
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)
            image_row = {
                "url": img_url,
                "alt": "",
                "title": "picture:source",
                "img_class": "",
                "img_id": "",
                "parent_class": "",
                "parent_id": "",
                "page_url": page_url,
                "index": len(images),
            }
            if self.is_likely_ui_asset(image_row):
                self.stats["images_filtered_ui"] += 1
                continue
            if self.reject_video_thumbnails and self.is_likely_video_thumbnail(image_row):
                self.stats["images_filtered_video"] += 1
                continue
            images.append(image_row)

        self.stats["images_found"] += len(images)
        return images

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _resolve_url(self, page_url: str, src: str, wayback_mode: bool) -> str:
        if wayback_mode:
            return self.wayback.rewrite_image_url(page_url, src)
        return urljoin(page_url, src)

    @staticmethod
    def _extract_css_bg_url(style: str) -> str:
        """Extract URL from a CSS background-image inline style."""
        import re
        m = re.search(r"background(?:-image)?\s*:[^;]*url\(['\"]?([^'\")\s]+)['\"]?\)", style, re.IGNORECASE)
        if m:
            return m.group(1)
        return ""

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    def download_image(
        self,
        img_url: str,
        website: WebsiteEntry,
        img_metadata: dict,
        source: str = "live",
        wayback_year: Optional[int] = None,
        wayback_timestamp: str = "",
        page_url: str = "",
    ) -> Optional[ImageMetadata]:
        """Download and validate an image."""
        try:
            response = self.session.get(
                img_url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self._pick_user_agent(), "Referer": page_url},
            )
            response.raise_for_status()
        except Exception as e:
            self.logger.debug(f"Failed to download {img_url}: {e}")
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if not content_type.startswith("image/"):
            return None

        try:
            img = Image.open(io.BytesIO(response.content))
            width, height = img.size
        except Exception as e:
            self.logger.debug(f"Invalid image at {img_url}: {e}")
            return None

        if self.reject_sticker_like and self.is_sticker_like(img, content_type):
            self.stats["images_filtered_sticker"] += 1
            return None

        if self.drop_near_duplicates and self.is_near_duplicate(img):
            self.stats["images_filtered_near_duplicate"] += 1
            return None

        if not self.passes_quality_filters(
            width=width,
            height=height,
            content_type=content_type,
            content_len=len(response.content),
        ):
            self.stats["images_filtered_quality"] += 1
            return None

        file_hash = hashlib.sha256(response.content).hexdigest()

        # Track duplicates for reporting, but keep metadata occurrences.
        if file_hash in self._downloaded_hashes:
            self.stats["images_skipped_duplicate"] += 1
            if self.drop_duplicate_content:
                return None
        else:
            self._downloaded_hashes.add(file_hash)

        ext = self._guess_extension(content_type, img_url)
        filepath: Path | None = None
        filename = ""
        for _ in range(10000):
            candidate = self._next_enumerated_filename(ext)
            candidate_path = self._build_image_output_path(
                website=website,
                source=source,
                wayback_year=wayback_year,
                filename=candidate,
            )
            if not candidate_path.exists():
                filename = candidate
                filepath = candidate_path
                break
        if filepath is None:
            raise RuntimeError("Could not allocate a unique enumerated filename")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(response.content)

        # Derive original URL (strip Wayback prefix if present)
        original_url = img_url
        if "/web/" in img_url and "web.archive.org" in img_url:
            try:
                after = img_url.split("/web/", 1)[1]
                first_segment, _, remainder = after.partition("/")
                if first_segment.endswith("if_"):
                    first_segment = first_segment[:-3]
                original_url = remainder.lstrip("/")
            except Exception:
                pass

        metadata = ImageMetadata(
            image_url=img_url,
            website_url=website.url,
            organization_name=website.organization_name,
            sector=website.sector,
            subsector=website.subsector,
            stored_filename=filename,
            stored_path=str(filepath),
            img_alt_text=img_metadata.get("alt", ""),
            img_title=img_metadata.get("title", ""),
            detected_width=width,
            detected_height=height,
            file_hash=file_hash,
            content_type=content_type,
            download_time=datetime.now().isoformat(),
            crawl_date=website.crawl_date,
            source=source,
            wayback_year=wayback_year,
            wayback_timestamp=wayback_timestamp,
            original_url=original_url,
            page_url=page_url,
        )

        self.stats["images_downloaded"] += 1
        return metadata

    def _guess_extension(self, content_type: str, img_url: str) -> str:
        path = urlparse(img_url).path.lower()
        for ext in self.IMAGE_EXTENSIONS:
            if path.endswith(ext):
                return ext
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        elif "png" in content_type:
            return ".png"
        elif "webp" in content_type:
            return ".webp"
        elif "gif" in content_type:
            return ".gif"
        return ".jpg"

    # ------------------------------------------------------------------
    # Page link extraction with priority ordering
    # ------------------------------------------------------------------

    def extract_links(self, page_url: str, html: str, domain: str) -> list[str]:
        """
        Extract same-domain links, prioritising content-rich paths.
        Returns a list where priority pages come first.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return []

        priority = []
        normal = []

        for link_tag in soup.find_all("a", href=True):
            href = link_tag.get("href", "")
            if not href:
                continue
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full_url = urljoin(page_url, href)
            parsed = urlparse(full_url)
            if parsed.scheme not in {"http", "https"}:
                continue
            if not self._is_same_site_host(parsed.netloc.lower(), domain):
                continue
            if any(parsed.path.lower().endswith(ext) for ext in self.NON_HTML_EXTENSIONS):
                continue

            clean = self._canonicalize_page_url(full_url)
            path_lower = parsed.path.lower()
            if any(kw in path_lower for kw in self.PRIORITY_PATH_KEYWORDS):
                priority.append(clean)
            else:
                normal.append(clean)

        return priority + normal

    @staticmethod
    def _is_same_site_host(host: str, base_host: str) -> bool:
        host = (host or "").lower()
        base_host = (base_host or "").lower()
        if host == base_host:
            return True

        # Treat www-prefixed and non-www hostnames as the same site.
        host_no_www = host[4:] if host.startswith("www.") else host
        base_no_www = base_host[4:] if base_host.startswith("www.") else base_host
        return host_no_www == base_no_www

    @staticmethod
    def _canonicalize_page_url(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        # Remove query params/fragments to avoid crawl-budget waste on tracking variants.
        cleaned = parsed._replace(path=path, query="", fragment="")
        return cleaned.geturl()

    # ------------------------------------------------------------------
    # Live crawl
    # ------------------------------------------------------------------

    def crawl_website(self, website: WebsiteEntry) -> int:
        """Crawl a single live website and download images."""
        self.logger.info(
            f"[LIVE] Starting crawl: {website.organization_name} ({website.url})"
        )
        self.stats["websites_attempted"] += 1
        return self._crawl_pages(
            website=website,
            start_url=website.url,
            source="live",
        )

    def _crawl_pages(
        self,
        website: WebsiteEntry,
        start_url: str,
        source: str,
        wayback_year: Optional[int] = None,
        wayback_timestamp: str = "",
    ) -> int:
        """Generic BFS page crawler shared by live and Wayback modes."""
        images_collected = 0
        pages_visited: set[str] = set()
        start_clean = self._canonicalize_page_url(start_url)
        to_visit = [start_clean]
        queued: set[str] = {start_clean}

        while (
            to_visit
            and len(pages_visited) < self.max_pages_per_site
            and images_collected < self.max_images_per_site
        ):
            page_url = self._canonicalize_page_url(to_visit.pop(0))
            if page_url in pages_visited:
                continue
            pages_visited.add(page_url)

            # Domain check (skip for Wayback URLs, which go through archive.org)
            if source == "live":
                if not self._is_same_site_host(urlparse(page_url).netloc.lower(), website.domain):
                    continue
                if not self._is_allowed_by_robots(page_url):
                    self.logger.debug(f"  Skipping by robots.txt policy: {page_url}")
                    self.stats["pages_skipped_robots"] += 1
                    continue

            self.logger.debug(
                f"  [{source.upper()}] Page ({len(pages_visited)}/{self.max_pages_per_site}): {page_url}"
            )

            try:
                page_html = self._fetch_page_html(page_url, source=source)
            except Exception as e:
                self.logger.warning(f"  Failed to fetch {page_url}: {e}")
                continue

            self.stats["pages_crawled"] += 1
            wayback_mode = source == "wayback"

            img_urls = self.extract_image_urls(page_url, page_html, wayback_mode=wayback_mode)

            for img_meta in img_urls:
                if images_collected >= self.max_images_per_site:
                    break
                meta = self.download_image(
                    img_url=img_meta["url"],
                    website=website,
                    img_metadata=img_meta,
                    source=source,
                    wayback_year=wayback_year,
                    wayback_timestamp=wayback_timestamp,
                    page_url=page_url,
                )
                if meta:
                    images_collected += 1
                    self._save_image_metadata(meta)

            # Follow links — use priority ordering
            new_links = self.extract_links(page_url, page_html, website.domain)
            for lnk in new_links:
                if lnk not in pages_visited and lnk not in queued:
                    to_visit.append(lnk)
                    queued.add(lnk)

            time.sleep(self.delay_seconds)

        return images_collected

    # ------------------------------------------------------------------
    # Wayback Machine crawl
    # ------------------------------------------------------------------

    def crawl_website_wayback(self, website: WebsiteEntry) -> dict[int, int]:
        """
        Crawl Wayback Machine snapshots of *website* across all configured years.
        Returns {year: images_collected}.
        """
        results: dict[int, int] = {}

        for year in self.wayback_years:
            self.logger.info(
                f"[WAYBACK {year}] {website.organization_name} ({website.url})"
            )
            snapshots = self.wayback.get_year_snapshots(
                website.url, year, max_snapshots=self.wayback_snapshots_per_year
            )

            if not snapshots:
                self.logger.warning(
                    f"  [WAYBACK {year}] No snapshots found for {website.url}"
                )
                results[year] = 0
                time.sleep(self.wayback.delay)
                continue

            year_images = 0
            for snap in snapshots:
                self.logger.debug(
                    f"  [WAYBACK {year}] Snapshot {snap['timestamp']}: {snap['wayback_url']}"
                )
                self.stats["wayback_snapshots_crawled"] += 1

                # Use a temporary WebsiteEntry pointing at the snapshot
                snap_website = WebsiteEntry(
                    url=website.url,
                    sector=website.sector,
                    subsector=website.subsector,
                    organization_name=website.organization_name,
                    crawl_date=f"{year}-01-01",
                )

                n = self._crawl_pages(
                    website=snap_website,
                    start_url=snap["wayback_url"],
                    source="wayback",
                    wayback_year=year,
                    wayback_timestamp=snap["timestamp"],
                )
                year_images += n
                time.sleep(self.wayback.delay)

            results[year] = year_images
            self.logger.info(
                f"  [WAYBACK {year}] {website.organization_name}: {year_images} images"
            )

        return results

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def crawl_all(self, websites: list[WebsiteEntry]) -> dict:
        """Crawl all websites (live + Wayback if enabled) and generate reports."""
        self.logger.info(f"Starting crawl of {len(websites)} websites")
        self.logger.info(f"Wayback enabled: {self.wayback_enabled}, years: {self.wayback_years}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(
            "Incremental mode: existing hashes=%d, next filename index=%d",
            len(self._downloaded_hashes),
            self._next_image_index,
        )

        total_images = 0

        try:
            for website in websites:
                # ── Live crawl ──────────────────────────────────────────────
                try:
                    n = self.crawl_website(website)
                    total_images += n
                    if n > 0:
                        self.stats["websites_successful"] += 1
                        self.logger.info(f"✓ [LIVE] {website.organization_name}: {n} images")
                    else:
                        self.logger.warning(f"✗ [LIVE] {website.organization_name}: No images")
                except Exception as e:
                    self.logger.error(f"Unexpected error (live) {website.url}: {e}")

                # ── Wayback crawl ───────────────────────────────────────────
                if self.wayback_enabled and self.wayback_years:
                    try:
                        wb_results = self.crawl_website_wayback(website)
                        for yr, n in wb_results.items():
                            total_images += n
                    except Exception as e:
                        self.logger.error(f"Unexpected error (wayback) {website.url}: {e}")

            self.stats["end_time"] = datetime.now()
            self.stats["total_images_collected"] = total_images

            self._generate_reports()
            return self.stats
        finally:
            self._close_playwright()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_image_metadata(self, metadata: ImageMetadata):
        """Save individual image metadata to JSON."""
        record_id = f"{metadata.file_hash[:12]}_{uuid.uuid4().hex[:10]}"
        metadata_file = self.metadata_dir / f"{record_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _generate_reports(self):
        """Generate summary reports (stats JSON + metadata CSV)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Statistics JSON
        stats_file = self.output_dir / f"stats_{ts}.json"
        stats_copy = dict(self.stats)
        stats_copy["start_time"] = stats_copy["start_time"].isoformat()
        if "end_time" in stats_copy:
            stats_copy["end_time"] = stats_copy["end_time"].isoformat()
        with open(stats_file, "w") as f:
            json.dump(stats_copy, f, indent=2)

        # 2. Metadata CSV (all images)
        csv_file = self.output_dir / f"images_metadata_{ts}.csv"
        metadata_files = list(self.metadata_dir.glob("*.json"))
        if metadata_files:
            all_metadata = []
            for mf in metadata_files:
                try:
                    with open(mf) as f:
                        all_metadata.append(json.load(f))
                except Exception as e:
                    self.logger.warning(f"Failed to read {mf}: {e}")

            if all_metadata:
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
                    writer.writeheader()
                    writer.writerows(all_metadata)
                self.logger.info(f"✓ Generated metadata CSV: {csv_file}")

            # 3. Per-year Wayback summary (useful for temporal analysis)
            if self.wayback_enabled:
                wayback_summary = defaultdict(lambda: defaultdict(int))
                for record in all_metadata:
                    if record.get("source") == "wayback" and record.get("wayback_year"):
                        yr = record["wayback_year"]
                        org = record["organization_name"]
                        wayback_summary[org][yr] += 1

                wb_csv = self.output_dir / f"wayback_yearly_summary_{ts}.csv"
                with open(wb_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["organization_name", "year", "images_collected"])
                    for org, years in sorted(wayback_summary.items()):
                        for yr, count in sorted(years.items()):
                            writer.writerow([org, yr, count])
                self.logger.info(f"✓ Generated Wayback yearly summary: {wb_csv}")

        self.logger.info("=" * 80)
        self.logger.info("CRAWL SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"  Websites attempted:          {self.stats['websites_attempted']}")
        self.logger.info(f"  Websites successful:         {self.stats['websites_successful']}")
        self.logger.info(f"  Pages crawled:               {self.stats['pages_crawled']}")
        self.logger.info(f"  Images found (post-filters): {self.stats['images_found']}")
        self.logger.info(f"  Images filtered (ads):       {self.stats['images_filtered_ads']}")
        self.logger.info(f"  Images filtered (ui/icon):   {self.stats['images_filtered_ui']}")
        self.logger.info(f"  Images filtered (quality):   {self.stats['images_filtered_quality']}")
        self.logger.info(f"  Images filtered (sticker):   {self.stats['images_filtered_sticker']}")
        self.logger.info(f"  Images filtered (video):     {self.stats['images_filtered_video']}")
        self.logger.info(f"  Images filtered (near-dup):  {self.stats['images_filtered_near_duplicate']}")
        self.logger.info(f"  Images filtered (no alt):    {self.stats['images_filtered_missing_alt']}")
        self.logger.info(f"  Images skipped (duplicate):  {self.stats['images_skipped_duplicate']}")
        self.logger.info(f"  Images downloaded:           {self.stats['images_downloaded']}")
        self.logger.info(f"  Pages skipped (robots):      {self.stats['pages_skipped_robots']}")
        self.logger.info(f"  Wayback snapshots crawled:   {self.stats['wayback_snapshots_crawled']}")
        if "end_time" in self.stats:
            self.logger.info(
                f"  Total time:                  {self.stats['end_time'] - self.stats['start_time']}"
            )
        self.logger.info("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crawl corporate/institutional websites and download images "
            "for AI-generation detection research"
        )
    )
    parser.add_argument(
        "--websites-csv", type=Path, required=True,
        help="CSV file (columns: url, sector, subsector, organization_name)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("crawl_results"),
        help="Output directory (default: crawl_results)"
    )
    parser.add_argument(
        "--max-images-per-site", type=int, default=50,
        help="Max images per website (default: 50)"
    )
    parser.add_argument(
        "--max-pages-per-site", type=int, default=10,
        help="Max pages per website (default: 10)"
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=15.0,
        help="HTTP request timeout in seconds (default: 15.0)"
    )
    parser.add_argument(
        "--render-js", action="store_true",
        help="Use Playwright rendering for live pages (handles JS/lazy loading)"
    )
    parser.add_argument(
        "--scroll-steps", type=int, default=3,
        help="Scroll steps used in JS rendering mode (default: 3)"
    )
    parser.add_argument(
        "--scroll-delay", type=float, default=0.6,
        help="Seconds to wait after each scroll in JS rendering mode (default: 0.6)"
    )
    parser.add_argument(
        "--playwright-timeout", type=float, default=20.0,
        help="Playwright page navigation timeout in seconds (default: 20)"
    )
    parser.add_argument(
        "--ignore-robots", action="store_true",
        help="Ignore robots.txt restrictions (not recommended)"
    )
    parser.add_argument(
        "--rotate-user-agent", action="store_true",
        help="Rotate User-Agent across requests"
    )
    parser.add_argument(
        "--user-agent", type=str, default="",
        help="Custom User-Agent string for HTTP requests"
    )
    parser.add_argument(
        "--cookie-json", type=Path, default=None,
        help="Optional cookie JSON (dict or list of cookies) for authenticated crawling"
    )
    parser.add_argument(
        "--require-alt-text", action="store_true",
        help="Keep only images with meaningful alt/caption text"
    )
    parser.add_argument(
        "--min-alt-text-chars", type=int, default=8,
        help="Minimum length for semantic alt/caption text (default: 8)"
    )
    parser.add_argument(
        "--min-width", type=int, default=180,
        help="Minimum image width (default: 180)"
    )
    parser.add_argument(
        "--min-height", type=int, default=180,
        help="Minimum image height (default: 180)"
    )
    parser.add_argument(
        "--min-area", type=int, default=50000,
        help="Minimum image area width×height (default: 50000)"
    )
    parser.add_argument(
        "--min-bytes", type=int, default=4000,
        help="Minimum image size in bytes (default: 4000)"
    )
    parser.add_argument(
        "--min-aspect-ratio", type=float, default=0.33,
        help="Minimum width/height ratio (default: 0.33)"
    )
    parser.add_argument(
        "--max-aspect-ratio", type=float, default=3.0,
        help="Maximum width/height ratio (default: 3.0)"
    )
    parser.add_argument(
        "--organize-by-sector-site", action="store_true",
        help="Store images under images/<sector>/<site>/<live|wayback>/..."
    )
    parser.add_argument(
        "--reject-sticker-like", action="store_true",
        help="Filter sticker-like transparent PNG/WebP assets"
    )
    parser.add_argument(
        "--sticker-max-transparent-ratio", type=float, default=0.35,
        help="Sticker filter: min transparent ratio threshold (default: 0.35)"
    )
    parser.add_argument(
        "--sticker-max-content-box-ratio", type=float, default=0.70,
        help="Sticker filter: max opaque bbox ratio threshold (default: 0.70)"
    )
    parser.add_argument(
        "--reject-video-thumbnails", action="store_true",
        help="Filter video thumbnails/covers/posters"
    )
    parser.add_argument(
        "--drop-duplicate-content", action="store_true",
        help="Drop identical images by content hash even if URL/filename differs"
    )
    parser.add_argument(
        "--drop-near-duplicates", action="store_true",
        help="Drop visually near-duplicate images using perceptual hash"
    )
    parser.add_argument(
        "--near-duplicate-hash-size", type=int, default=16,
        help="Perceptual hash grid size for near-duplicate filtering (default: 16)"
    )
    parser.add_argument(
        "--near-duplicate-hamming-threshold", type=int, default=8,
        help="Max Hamming distance to consider near-duplicate (default: 8)"
    )
    # Wayback Machine options
    parser.add_argument(
        "--wayback", action="store_true",
        help="Enable Wayback Machine historical crawl"
    )
    parser.add_argument(
        "--wayback-years", type=int, nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="Years to crawl via Wayback Machine (default: 2020–2024)"
    )
    parser.add_argument(
        "--wayback-snapshots-per-year", type=int, default=3,
        help="Snapshots per year per site for Wayback (default: 3)"
    )
    parser.add_argument(
        "--wayback-delay", type=float, default=2.0,
        help="Delay between Wayback API requests in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    logger.info("Corporate Crawler starting")

    if not args.websites_csv.exists():
        logger.error(f"Website CSV not found: {args.websites_csv}")
        sys.exit(1)

    crawler = CorporateCrawler(
        output_dir=args.output_dir,
        max_images_per_site=args.max_images_per_site,
        max_pages_per_site=args.max_pages_per_site,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        min_bytes=args.min_bytes,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        organize_by_sector_site=args.organize_by_sector_site,
        reject_sticker_like=args.reject_sticker_like,
        sticker_max_transparent_ratio=args.sticker_max_transparent_ratio,
        sticker_max_content_box_ratio=args.sticker_max_content_box_ratio,
        reject_video_thumbnails=args.reject_video_thumbnails,
        drop_duplicate_content=args.drop_duplicate_content,
        drop_near_duplicates=args.drop_near_duplicates,
        near_duplicate_hash_size=args.near_duplicate_hash_size,
        near_duplicate_hamming_threshold=args.near_duplicate_hamming_threshold,
        wayback_enabled=args.wayback,
        wayback_years=args.wayback_years,
        wayback_snapshots_per_year=args.wayback_snapshots_per_year,
        wayback_delay=args.wayback_delay,
        render_js=args.render_js,
        scroll_steps=args.scroll_steps,
        scroll_delay=args.scroll_delay,
        playwright_timeout=args.playwright_timeout,
        ignore_robots=args.ignore_robots,
        rotate_user_agent=args.rotate_user_agent,
        user_agent=args.user_agent,
        cookie_json=args.cookie_json,
        require_alt_text=args.require_alt_text,
        min_alt_text_chars=args.min_alt_text_chars,
        logger=logger,
    )

    websites = crawler.load_websites(args.websites_csv)
    crawler.crawl_all(websites)
    logger.info("Crawl completed successfully")


if __name__ == "__main__":
    main()