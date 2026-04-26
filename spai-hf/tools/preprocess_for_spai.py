#!/usr/bin/env python3
"""Pre-process images for SPAI: normalize dimensions and quality assessment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-process images for SPAI model input"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with images (organized by category/subcategory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for normalized images",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Input metadata CSV (from quality_filtered)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Target dimension for square resize (SPAI default: 224)",
    )
    parser.add_argument(
        "--min-upscale-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio before upscaling (0.5 = don't upscale if < 50% of target)",
    )
    parser.add_argument(
        "--filter-small",
        action="store_true",
        help="Exclude images smaller than target_size after analysis",
    )
    parser.add_argument(
        "--quality-loss-threshold",
        type=float,
        default=0.3,
        help="Warn if quality loss > N (0-1)",
    )
    return parser.parse_args()


def assess_upscale_quality(original_size: tuple[int, int], target_size: int) -> dict[str, Any]:
    """Assess how much quality will be lost by upscaling."""
    orig_w, orig_h = original_size
    orig_mp = (orig_w * orig_h) / 1e6
    target_mp = (target_size * target_size) / 1e6
    
    # For square resize, use smaller dimension as reference
    smaller_dim = min(orig_w, orig_h)
    upscale_ratio = target_size / smaller_dim if smaller_dim > 0 else 1.0
    
    # Quality loss metric: how much we're enlarging
    # 1.0x = no upscale, 2.0x = 2x larger, etc.
    quality_score = 1.0 / upscale_ratio  # 0-1, lower = more upscaling
    
    # Interpolation method recommendation
    if upscale_ratio <= 1.0:
        method = "LANCZOS"
        quality_concern = False
    elif upscale_ratio <= 1.5:
        method = "LANCZOS"
        quality_concern = False
    elif upscale_ratio <= 2.0:
        method = "BICUBIC"
        quality_concern = False
    elif upscale_ratio <= 3.0:
        method = "BICUBIC"
        quality_concern = True
    else:
        method = "BILINEAR"
        quality_concern = True
    
    return {
        "original_size": f"{orig_w}x{orig_h}",
        "original_mp": round(orig_mp, 3),
        "upscale_ratio": round(upscale_ratio, 2),
        "quality_score": round(quality_score, 2),  # 0-1, higher is better
        "interpolation": method,
        "quality_concern": quality_concern,
    }


def normalize_image(
    input_path: Path,
    output_path: Path,
    target_size: int = 224,
) -> dict[str, Any]:
    """Normalize image to target size. Returns assessment + status."""
    
    try:
        with Image.open(input_path) as img:
            orig_size = img.size
            orig_format = img.format
            
            # Assess quality impact
            assessment = assess_upscale_quality(orig_size, target_size)
            
            # Resize with aspect ratio preservation
            # Use LANCZOS for high quality
            img_resized = img.copy()
            img_resized.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Pad to exact square size (224x224)
            img_padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            
            # Center the resized image
            paste_x = (target_size - img_resized.width) // 2
            paste_y = (target_size - img_resized.height) // 2
            
            if img_resized.mode == "RGBA":
                img_padded.paste(img_resized, (paste_x, paste_y), img_resized)
            else:
                img_padded.paste(img_resized, (paste_x, paste_y))
            
            # Save normalized image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_padded.save(output_path, "JPEG", quality=95)
            
            assessment["status"] = "success"
            assessment["final_size"] = "224x224"
            assessment["output_path"] = str(output_path)
            
            return assessment
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(type(e).__name__),
            "message": str(e),
        }


def process_images(
    input_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    target_size: int = 224,
    min_upscale_ratio: float = 0.5,
    filter_small: bool = False,
    quality_loss_threshold: float = 0.3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all images. Returns (success, failed/filtered)."""
    
    # Load metadata
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    
    success_rows = []
    failed_rows = []
    
    for row in rows:
        # Find image file
        image_path = Path(row.get("quality_filtered_path", ""))
        if not image_path.is_absolute():
            image_path = input_dir.parent / image_path  # Go up one level to root
        
        if not image_path.exists():
            # Try alternate location
            alt_path = input_dir / Path(row.get("quality_filtered_path", "")).name
            if alt_path.exists():
                image_path = alt_path
            else:
                failed_rows.append({**row, "reason": f"File not found: {image_path}"})
                continue
        
        # Assess and normalize
        assessment = normalize_image(image_path, output_dir / image_path.name, target_size)
        
        if assessment["status"] != "success":
            failed_rows.append({**row, "reason": assessment.get("message", "Unknown error")})
            continue
        
        # Check quality concerns
        upscale_ratio = assessment.get("upscale_ratio", 1.0)
        quality_score = assessment.get("quality_score", 1.0)
        
        if upscale_ratio > 1.0 / min_upscale_ratio:  # More upscaling than threshold
            if filter_small:
                failed_rows.append({
                    **row,
                    "reason": f"Too much upscaling required ({upscale_ratio}x) - filtered",
                    **assessment,
                })
                continue
        
        # Include assessment in output row
        row["normalized_path"] = assessment["output_path"]
        row["upscale_ratio"] = assessment["upscale_ratio"]
        row["quality_score"] = assessment["quality_score"]
        row["interpolation"] = assessment["interpolation"]
        row["quality_concern"] = assessment["quality_concern"]
        
        success_rows.append(row)
    
    return success_rows, failed_rows


def main() -> None:
    args = parse_args()
    
    print(f"Pre-processing images for SPAI model")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Min upscale ratio: {args.min_upscale_ratio}")
    print(f"Filter small images: {args.filter_small}")
    print()
    
    success_rows, failed_rows = process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        target_size=args.target_size,
        min_upscale_ratio=args.min_upscale_ratio,
        filter_small=args.filter_small,
        quality_loss_threshold=args.quality_loss_threshold,
    )
    
    # Save results
    output_metadata = args.output_dir / "normalized_metadata.csv"
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    
    if success_rows:
        fieldnames = sorted({key for row in success_rows for key in row.keys()})
        with output_metadata.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(success_rows)
    
    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"✓ Processed: {len(success_rows)}")
    print(f"✗ Failed/Filtered: {len(failed_rows)}")
    
    # Quality statistics
    if success_rows:
        upscales = [float(r.get("upscale_ratio", 1)) for r in success_rows if "upscale_ratio" in r]
        quality_scores = [float(r.get("quality_score", 1)) for r in success_rows]
        have_concerns = sum(1 for r in success_rows if r.get("quality_concern"))
        
        print(f"\nQuality Metrics:")
        print(f"  Avg upscale ratio: {sum(upscales)/len(upscales):.2f}x")
        print(f"  Avg quality score: {sum(quality_scores)/len(quality_scores):.2f}/1.0")
        print(f"  Images with concerns: {have_concerns} ({100*have_concerns/len(success_rows):.1f}%)")
        
        print(f"\nNormalized images: {args.output_dir}")
        print(f"Updated metadata: {output_metadata}")
    
    if failed_rows:
        print(f"\nFailed/Filtered reasons:")
        reasons = {}
        for row in failed_rows:
            reason = row.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
