#!/usr/bin/env python3
"""
Generic Web Image Classifier using SPAI
Crawl images from any website and run AI-generation inference on them.

Usage:
    # Basic usage with a website URL
    python infer_web_classifier.py \
        --url https://www.example.com \
        --output-csv results_example.csv
    
    # With custom settings
    python infer_web_classifier.py \
        --url https://www.bbcnews.com \
        --max-images 100 \
        --threshold 0.5 \
        --output-csv results_bbc.csv \
        --crawl-depth 2
    
    # Multiple websites
    python infer_web_classifier.py \
        --urls https://www.bbc.com https://www.cnn.com \
        --max-images-per-url 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

from inference import EndpointHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebImageCrawler:
    """Generic crawler for images from any website."""
    
    def __init__(
        self, 
        max_images: int = 50,
        delay_seconds: float = 1.0,
        crawl_depth: int = 1,
        timeout: int = 10
    ):
        self.max_images = max_images
        self.delay_seconds = delay_seconds
        self.crawl_depth = crawl_depth
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.visited_urls = set()
    
    def crawl_images(self, base_urls: list[str]) -> list[dict[str, Any]]:
        """Crawl images from the given URLs."""
        images = []
        to_visit = [(url, 0) for url in base_urls]  # (url, depth)
        
        while to_visit and len(images) < self.max_images:
            current_url, depth = to_visit.pop(0)
            
            # Skip if already visited or depth exceeded
            if current_url in self.visited_urls or depth > self.crawl_depth:
                continue
            
            self.visited_urls.add(current_url)
            
            try:
                logger.info(f"Crawling [{depth+1}/{self.crawl_depth}]: {current_url}")
                response = self.session.get(current_url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, "html.parser")
                base_domain = urlparse(current_url).netloc
                
                # Find all img tags
                img_tags = soup.find_all("img", limit=(self.max_images - len(images)) * 2)
                logger.info(f"  Found {len(img_tags)} img tags")
                
                for img_tag in img_tags:
                    if len(images) >= self.max_images:
                        break
                    
                    # Get image URL
                    img_url = img_tag.get("src") or img_tag.get("data-src")
                    if not img_url:
                        continue
                    
                    # Resolve relative URLs
                    img_url = self._resolve_url(img_url, current_url)
                    if not img_url:
                        continue
                    
                    # Try to download image
                    if self._download_and_verify_image(img_url):
                        alt_text = img_tag.get("alt", "")
                        title = img_tag.get("title", "")
                        description = alt_text or title or f"Image from {base_domain}"
                        
                        images.append({
                            "url": img_url,
                            "description": description,
                            "source": base_domain,
                            "source_page": current_url
                        })
                        
                        logger.info(
                            f"  ✓ Image {len(images)}/{self.max_images}: "
                            f"{description[:40]}"
                        )
                        time.sleep(self.delay_seconds)
                
                # Find links to crawl (if depth allows)
                if depth < self.crawl_depth:
                    links = soup.find_all("a", href=True, limit=10)
                    for link in links:
                        link_url = self._resolve_url(link["href"], current_url)
                        if link_url and link_url not in self.visited_urls:
                            # Only visit same domain
                            if urlparse(link_url).netloc == base_domain:
                                to_visit.append((link_url, depth + 1))
                
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                continue
        
        logger.info(f"\n✓ Successfully crawled {len(images)} images")
        return images
    
    def _resolve_url(self, url: str, base_url: str) -> str | None:
        """Resolve relative URLs to absolute URLs."""
        if not url or not isinstance(url, str):
            return None
        
        # Handle protocol-relative URLs
        if url.startswith("//"):
            parsed_base = urlparse(base_url)
            return f"{parsed_base.scheme}:{url}"
        
        # Handle relative URLs
        if url.startswith("/"):
            return urljoin(base_url, url)
        
        # Handle absolute URLs
        if url.startswith("http"):
            return url
        
        # Handle implicit relative URLs
        if not url.startswith("http") and not url.startswith("/"):
            return urljoin(base_url, url)
        
        return None
    
    def _download_and_verify_image(self, img_url: str) -> bool:
        """Download and verify an image."""
        try:
            response = self.session.get(img_url, timeout=5)
            response.raise_for_status()
            
            # Verify it's a valid image
            img = Image.open(BytesIO(response.content))
            img.verify()
            
            return True
        except Exception:
            return False


def infer_on_images(
    images: list[dict[str, Any]],
    output_csv: Path,
    output_jsonl: Path,
    threshold: float = 0.6,
    model_dir: str = "/fhome/aaasidar/spai-hf",
) -> dict[str, Any]:
    """Run SPAI inference on images and save results."""
    
    logger.info(f"\nLoading SPAI model from {model_dir}...")
    handler = EndpointHandler(path=model_dir)
    
    results = []
    errors = []
    
    for idx, image_data in enumerate(images, 1):
        try:
            logger.info(
                f"[{idx}/{len(images)}] Inferring: {image_data['description'][:40]}"
            )
            
            # Run inference
            result = handler({"inputs": image_data["url"]})
            
            # Add metadata
            result.update({
                "image_url": image_data["url"],
                "description": image_data["description"],
                "source": image_data["source"],
                "source_page": image_data.get("source_page", "")
            })
            
            results.append(result)
            
            # Append to JSONL
            with output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=True) + "\n")
            
            label_str = "🤖 AI" if result["predicted_label"] == 1 else "📸 REAL"
            logger.info(
                f"  {label_str} | Score: {result['score']:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to infer on {image_data['url']}: {e}")
            errors.append(str(image_data['url']))
            continue
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"\n✓ Saved {len(results)} results to {output_csv}")
    
    # Print statistics
    stats = _compute_statistics(results)
    stats['errors'] = len(errors)
    
    return stats


def _compute_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics from results."""
    if not results:
        return {}
    
    scores = [r["score"] for r in results]
    ai_count = sum(1 for r in results if r["predicted_label"] == 1)
    real_count = len(results) - ai_count
    
    return {
        "total": len(results),
        "ai_generated": ai_count,
        "real": real_count,
        "ai_percentage": 100 * ai_count / len(results),
        "real_percentage": 100 * real_count / len(results),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "median_score": sorted(scores)[len(scores) // 2]
    }


def print_statistics(stats: dict[str, Any]) -> None:
    """Print statistics in a nice format."""
    if not stats:
        logger.info("No results to display")
        return
    
    logger.info("\n" + "="*70)
    logger.info("📊 CLASSIFICATION RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"Total images analyzed:  {stats['total']}")
    logger.info(f"")
    logger.info(f"🤖 AI-Generated:  {stats['ai_generated']:>3}  ({stats['ai_percentage']:>5.1f}%)")
    logger.info(f"📸 Real:          {stats['real']:>3}  ({stats['real_percentage']:>5.1f}%)")
    logger.info(f"")
    logger.info(f"Score Statistics:")
    logger.info(f"  Average:  {stats['avg_score']:.3f}")
    logger.info(f"  Median:   {stats['median_score']:.3f}")
    logger.info(f"  Min:      {stats['min_score']:.3f}")
    logger.info(f"  Max:      {stats['max_score']:.3f}")
    if stats.get('errors', 0) > 0:
        logger.info(f"")
        logger.info(f"⚠️  Processing errors: {stats['errors']}")
    logger.info("="*70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic web image classifier using SPAI AI-generation detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single website
  python infer_web_classifier.py --url https://www.bbc.com/news

  # Multiple websites
  python infer_web_classifier.py --urls https://www.bbc.com https://www.cnn.com

  # Custom output and parameters
  python infer_web_classifier.py \\
    --url https://example.com \\
    --max-images 100 \\
    --threshold 0.5 \\
    --output-csv results.csv

  # Deeper crawl
  python infer_web_classifier.py \\
    --url https://example.com \\
    --crawl-depth 2 \\
    --max-images 200
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url",
        type=str,
        help="Single website URL to crawl"
    )
    input_group.add_argument(
        "--urls",
        type=str,
        nargs="+",
        help="Multiple website URLs to crawl"
    )
    
    # Output options
    parser.add_argument(
        "--output-csv",
        type=str,
        default="web_classifier_results.csv",
        help="Output CSV file path (default: web_classifier_results.csv)"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Output JSONL file path (default: same as CSV but .jsonl)"
    )
    
    # Crawling options
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum total images to crawl (default: 50)"
    )
    parser.add_argument(
        "--max-images-per-url",
        type=int,
        default=None,
        help="Maximum images per URL (overrides --max-images for each URL)"
    )
    parser.add_argument(
        "--crawl-depth",
        type=int,
        default=1,
        help="How deep to crawl (1=homepage only, 2=homepage + links, etc.) (default: 1)"
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    
    # Classification options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Classification threshold 0-1 (default: 0.6)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/fhome/aaasidar/spai-hf",
        help="Directory containing SPAI model and config"
    )
    
    args = parser.parse_args()
    
    # Determine URLs
    urls = args.urls if args.urls else [args.url]
    
    # Determine output paths
    output_csv = Path(args.output_csv)
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
    else:
        output_jsonl = output_csv.with_suffix(".jsonl")
    
    # Clear output files
    output_csv.unlink(missing_ok=True)
    output_jsonl.unlink(missing_ok=True)
    
    # Crawl images
    logger.info(f"🕷️  Starting web crawler for {len(urls)} website(s)...")
    logger.info(f"   URLs: {', '.join(urls)}")
    logger.info(f"   Target images: {args.max_images}")
    logger.info(f"   Crawl depth: {args.crawl_depth}\n")
    
    crawler = WebImageCrawler(
        max_images=args.max_images,
        delay_seconds=args.delay_seconds,
        crawl_depth=args.crawl_depth
    )
    images = crawler.crawl_images(urls)
    
    if not images:
        logger.error("❌ No images crawled from the given URLs")
        sys.exit(1)
    
    # Run inference
    logger.info(f"\n🤖 Running SPAI AI-generation classifier...")
    logger.info(f"   Threshold: {args.threshold}\n")
    
    stats = infer_on_images(
        images=images,
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        threshold=args.threshold,
        model_dir=args.model_dir
    )
    
    print_statistics(stats)


if __name__ == "__main__":
    main()
