#!/usr/bin/env python3
"""Post-process crawler output: validate quality, deduplicate, generate reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


@dataclass
class QualityConfig:
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    min_megapixels: float
    max_megapixels: float
    duplicate_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image quality, deduplicate, and generate reports"
    )
    parser.add_argument(
        "--crawler-output-dir",
        type=Path,
        required=True,
        help="Output directory from web_image_crawler.py",
    )
    parser.add_argument(
        "--quality-output-dir",
        type=Path,
        required=True,
        help="Output directory for quality-filtered results",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="Minimum image width in pixels",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=256,
        help="Minimum image height in pixels",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=8192,
        help="Maximum image width in pixels",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=8192,
        help="Maximum image height in pixels",
    )
    parser.add_argument(
        "--min-megapixels",
        type=float,
        default=0.065,
        help="Minimum megapixels (width*height/1M)",
    )
    parser.add_argument(
        "--max-megapixels",
        type=float,
        default=50.0,
        help="Maximum megapixels",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Perceptual hash similarity threshold (0-1) for duplicates",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    contextual_dir = base / "contextual_images"
    ads_dir = base / "ads_images"
    metadata_dir = base / "metadata"
    reports_dir = base / "reports"

    for directory in (contextual_dir, ads_dir, metadata_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "contextual": contextual_dir,
        "ads": ads_dir,
        "metadata": metadata_dir,
        "reports": reports_dir,
    }


def validate_image_quality(
    image_path: Path, config: QualityConfig
) -> tuple[bool, list[str]]:
    """Validate image dimensions and quality. Returns (is_valid, reasons)."""
    reasons: list[str] = []

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        return False, [f"corrupt_or_unreadable: {type(e).__name__}"]

    if width < config.min_width or height < config.min_height:
        reasons.append(f"too_small: {width}x{height} < {config.min_width}x{config.min_height}")

    if width > config.max_width or height > config.max_height:
        reasons.append(f"too_large: {width}x{height} > {config.max_width}x{config.max_height}")

    megapixels = (width * height) / 1e6
    if megapixels < config.min_megapixels:
        reasons.append(f"insufficient_megapixels: {megapixels:.3f}M < {config.min_megapixels}M")

    if megapixels > config.max_megapixels:
        reasons.append(f"excessive_megapixels: {megapixels:.3f}M > {config.max_megapixels}M")

    is_valid = len(reasons) == 0
    return is_valid, reasons


def compute_perceptual_hash(image_path: Path) -> str | None:
    """Compute dhash for perceptual deduplication."""
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        with Image.open(image_path) as img:
            return str(imagehash.dhash(img))
    except Exception:
        return None


def hash_similarity(hash1: str, hash2: str) -> float:
    """Hamming similarity between two hashes (0-1)."""
    if not hash1 or not hash2:
        return 0.0
    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    max_distance = max(len(hash1), len(hash2)) * 4
    return 1.0 - (distance / max_distance)


def load_metadata_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load crawler output CSV."""
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_metadata_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write metadata CSV."""
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_images(
    input_csv: Path,
    input_img_base: Path,
    output_img_base: Path,
    config: QualityConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate and deduplicate images. Returns (kept, rejected, stats)."""
    rows = load_metadata_csv(input_csv)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: dict[str, Path] = {}
    stats: dict[str, Any] = {
        "total_input": len(rows),
        "valid_quality": 0,
        "invalid_quality": 0,
        "duplicates_removed": 0,
        "quality_reasons": defaultdict(int),
        "size_distribution": defaultdict(int),
        "category_counts": defaultdict(int),
    }

    for row in rows:
        # Use filtered_path as-is (already contains full relative path from crawler output)
        filtered_path = Path(row.get("filtered_path", ""))
        
        # Ensure it's absolute or resolve from cwd
        if not filtered_path.is_absolute():
            filtered_path = Path.cwd() / filtered_path
        is_valid, reasons = validate_image_quality(filtered_path, config)
        if not is_valid:
            for reason in reasons:
                stats["quality_reasons"][reason] += 1
            row["rejection_reason"] = "|".join(reasons)
            rejected.append(row)
            stats["invalid_quality"] += 1
            continue

        stats["valid_quality"] += 1

        # Compute perceptual hash
        phash = compute_perceptual_hash(filtered_path)
        is_duplicate = False
        if phash and IMAGEHASH_AVAILABLE:
            for existing_hash, existing_path in seen_hashes.items():
                similarity = hash_similarity(phash, existing_hash)
                if similarity >= config.duplicate_threshold:
                    is_duplicate = True
                    row["duplicate_of"] = str(existing_path)
                    rejected.append(row)
                    stats["duplicates_removed"] += 1
                    break

        if not is_duplicate:
            # Get image dimensions for stats
            try:
                with Image.open(filtered_path) as img:
                    w, h = img.size
                    size_bucket = f"{(w*h)//1e6:.1f}M"
                    stats["size_distribution"][size_bucket] += 1
            except Exception:
                pass

            category = row.get("category", "unknown")
            stats["category_counts"][category] += 1

            row["perceptual_hash"] = phash or ""
            row["quality_validation"] = "pass"
            kept.append(row)
            if phash:
                seen_hashes[phash] = filtered_path

    return kept, rejected, stats


def copy_valid_images(kept_rows: list[dict[str, Any]], output_dir: Path, input_base: Path) -> None:
    """Copy valid images to output directory, preserving category structure."""
    import shutil

    for row in kept_rows:
        input_path = Path(row.get("filtered_path", ""))
        if not input_path.is_absolute():
            input_path = input_base / input_path

        category = row.get("category", "uncategorized")
        subcategory = row.get("subcategory", "general")
        output_subdir = output_dir / category / subcategory
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_path = output_subdir / input_path.name
        if input_path.exists():
            shutil.copy2(input_path, output_path)
            row["quality_filtered_path"] = str(output_path)


def generate_html_report(
    stats: dict[str, Any],
    kept_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate HTML summary report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Image Quality Processing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #0066cc; margin-top: 30px; }
        .stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .stat-box { background: #f0f0f0; padding: 15px; border-radius: 5px; border-left: 4px solid #0066cc; }
        .stat-value { font-size: 28px; font-weight: bold; color: #0066cc; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th { background: #0066cc; color: white; padding: 10px; text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f9f9f9; }
        .warning { color: #d9534f; }
        .success { color: #5cb85c; }
    </style>
</head>
<body>
<div class="container">
    <h1>🖼️ Image Quality Processing Report</h1>
"""

    html += f"""
    <h2>Summary Statistics</h2>
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">{stats.get('total_input', 0)}</div>
            <div class="stat-label">Total Input Images</div>
        </div>
        <div class="stat-box">
            <div class="stat-value success">{len(kept_rows)}</div>
            <div class="stat-label">Passed Quality Check</div>
        </div>
        <div class="stat-box">
            <div class="stat-value warning">{len(rejected_rows)}</div>
            <div class="stat-label">Rejected</div>
        </div>
    </div>

    <h2>Quality Validation</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Valid Quality</td><td class="success">{stats.get('valid_quality', 0)}</td></tr>
        <tr><td>Invalid Quality</td><td class="warning">{stats.get('invalid_quality', 0)}</td></tr>
        <tr><td>Duplicates Removed</td><td class="warning">{stats.get('duplicates_removed', 0)}</td></tr>
        <tr><td>Kept Images (Final)</td><td class="success">{len(kept_rows)}</td></tr>
    </table>

    <h2>Rejection Reasons</h2>
    <table>
        <tr><th>Reason</th><th>Count</th></tr>
"""

    for reason, count in sorted(
        stats.get("quality_reasons", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        html += f"        <tr><td>{reason}</td><td>{count}</td></tr>\n"

    html += """    </table>

    <h2>Category Distribution</h2>
    <table>
        <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
"""

    total_kept = len(kept_rows)
    for category, count in sorted(
        stats.get("category_counts", {}).items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = 100 * count / total_kept if total_kept > 0 else 0
        html += f"        <tr><td>{category}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"

    html += """    </table>

    <h2>Size Distribution</h2>
    <table>
        <tr><th>Megapixels</th><th>Count</th></tr>
"""

    for size_bucket, count in sorted(stats.get("size_distribution", {}).items()):
        html += f"        <tr><td>{size_bucket}</td><td>{count}</td></tr>\n"

    html += """    </table>
</div>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()

    config = QualityConfig(
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        min_megapixels=args.min_megapixels,
        max_megapixels=args.max_megapixels,
        duplicate_threshold=args.duplicate_threshold,
    )

    print(f"Image Quality Processor")
    print(f"Input: {args.crawler_output_dir}")
    print(f"Output: {args.quality_output_dir}")
    if not IMAGEHASH_AVAILABLE:
        print("⚠️  imagehash not available. Deduplication will be skipped.")

    input_dirs = {
        "contextual": args.crawler_output_dir / "contextual_images",
        "ads": args.crawler_output_dir / "ads_images",
    }
    output_dirs = ensure_dirs(args.quality_output_dir)

    # Process contextual images
    print("\n[1/2] Processing contextual images...")
    contextual_csv = args.crawler_output_dir / "metadata" / "contextual_images.csv"
    kept_ctx, rejected_ctx, stats_ctx = process_images(
        contextual_csv, args.crawler_output_dir, output_dirs["contextual"], config
    )
    copy_valid_images(kept_ctx, output_dirs["contextual"], args.crawler_output_dir)
    write_metadata_csv(kept_ctx, output_dirs["metadata"] / "contextual_images_filtered.csv")
    write_metadata_csv(rejected_ctx, output_dirs["metadata"] / "contextual_images_rejected.csv")
    print(f"  Kept: {len(kept_ctx)}, Rejected: {len(rejected_ctx)}")

    # Process ads images
    print("[2/2] Processing ads images...")
    ads_csv = args.crawler_output_dir / "metadata" / "ads_images.csv"
    kept_ads, rejected_ads, stats_ads = process_images(
        ads_csv, args.crawler_output_dir, output_dirs["ads"], config
    )
    copy_valid_images(kept_ads, output_dirs["ads"], args.crawler_output_dir)
    write_metadata_csv(kept_ads, output_dirs["metadata"] / "ads_images_filtered.csv")
    write_metadata_csv(rejected_ads, output_dirs["metadata"] / "ads_images_rejected.csv")
    print(f"  Kept: {len(kept_ads)}, Rejected: {len(rejected_ads)}")

    # Merge stats
    merged_quality_reasons = defaultdict(int, stats_ctx["quality_reasons"])
    merged_quality_reasons.update(stats_ads["quality_reasons"])
    
    merged_category_counts = defaultdict(int, stats_ctx["category_counts"])
    merged_category_counts.update(stats_ads["category_counts"])
    
    merged_size_distribution = defaultdict(int, stats_ctx["size_distribution"])
    merged_size_distribution.update(stats_ads["size_distribution"])
    
    all_stats = {
        "total_input": stats_ctx["total_input"] + stats_ads["total_input"],
        "valid_quality": stats_ctx["valid_quality"] + stats_ads["valid_quality"],
        "invalid_quality": stats_ctx["invalid_quality"] + stats_ads["invalid_quality"],
        "duplicates_removed": stats_ctx["duplicates_removed"] + stats_ads["duplicates_removed"],
        "quality_reasons": dict(merged_quality_reasons),
        "category_counts": dict(merged_category_counts),
        "size_distribution": dict(merged_size_distribution),
    }

    # Generate reports
    print("\nGenerating reports...")
    generate_html_report(
        all_stats,
        kept_ctx + kept_ads,
        rejected_ctx + rejected_ads,
        output_dirs["reports"] / "quality_report.html",
    )

    summary_json = {
        "total_input": all_stats["total_input"],
        "valid_quality": all_stats["valid_quality"],
        "invalid_quality": all_stats["invalid_quality"],
        "duplicates_removed": all_stats["duplicates_removed"],
        "final_kept": len(kept_ctx) + len(kept_ads),
        "quality_reasons": dict(all_stats["quality_reasons"]),
        "category_distribution": dict(all_stats["category_counts"]),
    }
    (output_dirs["metadata"] / "quality_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    print("\nDone!")
    print(f"✓ Quality report: {output_dirs['reports'] / 'quality_report.html'}")
    print(f"✓ Filtered metadata: {output_dirs['metadata']}")
    print(f"✓ Filtered images: {output_dirs['contextual']} & {output_dirs['ads']}")
    print(f"\nFinal count: {len(kept_ctx) + len(kept_ads)} images (from {all_stats['total_input']} input)")


if __name__ == "__main__":
    main()
