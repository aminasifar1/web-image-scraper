#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image as PILImage

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    if path.suffix.lower() in IMAGE_EXTS:
        return True
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def export_hf_images(ds_name: str, out_dir: Path, max_images: int | None = None) -> list[dict]:
    ds = load_dataset(ds_name)
    rows = []
    count = 0

    for split_name, split_ds in ds.items():
        image_cols = [
            c
            for c, f in split_ds.features.items()
            if getattr(f, "dtype", None) == "PIL.Image.Image" or f.__class__.__name__ == "Image"
        ]
        if not image_cols:
            continue

        for idx, sample in enumerate(split_ds):
            if max_images is not None and count >= max_images:
                return rows

            for img_col in image_cols:
                image_obj = sample.get(img_col)
                if image_obj is None:
                    continue

                if not isinstance(image_obj, PILImage.Image):
                    pil_img = image_obj if hasattr(image_obj, "save") else None
                else:
                    pil_img = image_obj

                if pil_img is None:
                    continue

                file_stem = f"hf_{safe_name(split_name)}_{idx:06d}_{safe_name(img_col)}"
                out_path = out_dir / f"{file_stem}.png"
                pil_img.convert("RGB").save(out_path, format="PNG")

                label = None
                for cand in ("label", "binary_label", "class", "target"):
                    if cand in sample:
                        label = sample[cand]
                        break

                rows.append(
                    {
                        "filename": out_path.name,
                        "source": "hf_genai_bench",
                        "split": split_name,
                        "source_relpath": f"{split_name}/{idx}",
                        "label_raw": label,
                    }
                )
                count += 1
                if max_images is not None and count >= max_images:
                    return rows

    return rows


def copy_local_images(local_dir: Path, out_dir: Path, max_images: int | None = None) -> list[dict]:
    rows = []
    idx = 0
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file() or not is_image_file(p):
            continue
        if max_images is not None and idx >= max_images:
            break

        rel = p.relative_to(local_dir)
        stem = safe_name(rel.as_posix().replace("/", "_"))
        ext = p.suffix.lower() if p.suffix else ".jpg"
        out_name = f"local_{idx:06d}_{stem}{ext}"
        out_path = out_dir / out_name
        shutil.copy2(p, out_path)

        rows.append(
            {
                "filename": out_path.name,
                "source": "local_siuuu",
                "split": "na",
                "source_relpath": rel.as_posix(),
                "label_raw": "",
            }
        )
        idx += 1

    return rows


def write_manifest(rows: Iterable[dict], manifest_csv: Path) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "source", "split", "source_relpath", "label_raw"]
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge GenAI-Bench and local images into one folder")
    ap.add_argument("--hf-dataset", default="BaiqiL/GenAI-Bench")
    ap.add_argument("--local-dir", default="/fhome/aaasidar/spai-hf/siuuu")
    ap.add_argument("--out-dir", default="/fhome/aaasidar/spai-hf/merged_genaibench_siuuu/images")
    ap.add_argument("--manifest", default="/fhome/aaasidar/spai-hf/merged_genaibench_siuuu/manifest.csv")
    ap.add_argument("--max-hf-images", type=int, default=None)
    ap.add_argument("--max-local-images", type=int, default=None)
    ap.add_argument("--skip-local", action="store_true", help="Export only HF dataset images")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_dir = Path(args.local_dir)

    print(f"[INFO] exporting HF dataset: {args.hf_dataset}")
    hf_rows = export_hf_images(args.hf_dataset, out_dir, args.max_hf_images)
    print(f"[INFO] HF images exported: {len(hf_rows)}")

    local_rows = []
    if args.skip_local:
        print("[INFO] skipping local image copy (--skip-local)")
    elif local_dir.exists():
        print(f"[INFO] copying local images from: {local_dir}")
        local_rows = copy_local_images(local_dir, out_dir, args.max_local_images)
        print(f"[INFO] local images copied: {len(local_rows)}")
    else:
        print(f"[WARN] local directory not found, skipping: {local_dir}")

    all_rows = hf_rows + local_rows
    write_manifest(all_rows, Path(args.manifest))
    print(f"[OK] total images in merged folder: {len(all_rows)}")
    print(f"[OK] merged images: {out_dir}")
    print(f"[OK] manifest: {args.manifest}")


if __name__ == "__main__":
    main()
