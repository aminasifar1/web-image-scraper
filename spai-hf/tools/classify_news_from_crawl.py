#!/usr/bin/env python3
"""Classify news images from crawler metadata and export separate score graphics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import EndpointHandler


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _infer_gt_from_filename(file_name: str) -> tuple[int | None, str | None]:
    """Infer GT label from merged-dataset file prefixes.

    Convention used in this project:
      - hf_*    -> AI (1)
      - local_* -> Real (0)
    """
    name = file_name.lower()
    if name.startswith("hf_"):
        return 1, "ai-generated"
    if name.startswith("local_"):
        return 0, "real"
    return None, None


def _is_news_row(row: pd.Series) -> bool:
    sector = str(row.get("sector", "")).strip().lower()
    subsector = str(row.get("subsector", "")).strip().lower()
    org = str(row.get("organization_name", "")).strip().lower()
    return (
        "news" in sector
        or "news" in subsector
        or "newspaper" in subsector
        or "news" in org
    )


def _resolve_input_image(row: pd.Series) -> str | None:
    stored = str(row.get("stored_path", "")).strip()
    if stored:
        p = Path(stored)
        if p.exists():
            return str(p)

    image_url = str(row.get("image_url", "")).strip()
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url

    return None


def classify_news_images(
    metadata_csv: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    max_images: int,
) -> dict[str, Any]:
    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"Empty metadata CSV: {metadata_csv}")

    news_df = df[df.apply(_is_news_row, axis=1)].copy()
    if news_df.empty:
        raise ValueError("No rows matched news/newspaper sectors in metadata CSV")

    if max_images > 0:
        news_df = news_df.head(max_images).copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set decision threshold used by EndpointHandler.
    import os

    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))

    rows: list[dict[str, Any]] = []
    failures = 0

    for _, row in news_df.iterrows():
        image_input = _resolve_input_image(row)
        if not image_input:
            failures += 1
            continue

        try:
            pred = handler({"inputs": image_input})
            rows.append(
                {
                    "score": float(pred["score"]),
                    "predicted_label": int(pred["predicted_label"]),
                    "predicted_label_name": pred["predicted_label_name"],
                    "ground_truth_label": None,
                    "threshold": float(pred["threshold"]),
                    "image_input": image_input,
                    "image_url": row.get("image_url", ""),
                    "source": row.get("source", ""),
                    "organization_name": row.get("organization_name", ""),
                    "sector": row.get("sector", ""),
                    "subsector": row.get("subsector", ""),
                    "img_alt_text": row.get("img_alt_text", ""),
                    "img_title": row.get("img_title", ""),
                    "page_url": row.get("page_url", ""),
                    "stored_path": row.get("stored_path", ""),
                }
            )
        except Exception:
            failures += 1

    if not rows:
        raise RuntimeError("No images could be classified from selected news rows")

    pred_df = pd.DataFrame(rows)
    pred_csv = output_dir / "news_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    _plot_score_curve(pred_df, output_dir)
    _plot_score_histogram(pred_df, output_dir)
    _plot_testing_graphics(pred_df, output_dir)

    summary = {
        "metadata_csv": str(metadata_csv),
        "selected_news_rows": int(len(news_df)),
        "classified": int(len(pred_df)),
        "failed": int(failures),
        "threshold": float(threshold),
        "ai_count": int((pred_df["predicted_label"] == 1).sum()),
        "real_count": int((pred_df["predicted_label"] == 0).sum()),
        "score_mean": float(pred_df["score"].mean()),
        "score_median": float(pred_df["score"].median()),
        "score_min": float(pred_df["score"].min()),
        "score_max": float(pred_df["score"].max()),
        "artifacts": {
            "predictions_csv": str(pred_csv),
            "score_plot": str(output_dir / "score.png"),
            "score_histogram": str(output_dir / "score_histogram.png"),
            "testing_graphics": str(output_dir / "testing_graphics.png"),
        },
    }

    summary_json = output_dir / "news_analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def classify_images_dir(
    images_dir: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    max_images: int,
    assume_all_real: bool,
) -> dict[str, Any]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"Images directory does not exist or is not a directory: {images_dir}")

    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {images_dir}")

    if max_images > 0:
        image_paths = image_paths[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)

    import os

    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))

    rows: list[dict[str, Any]] = []
    failures = 0
    running_ai = 0
    running_real = 0
    running_with_gt = 0
    running_correct = 0

    per_image_csv = output_dir / "news_per_image_log.csv"
    per_image_jsonl = output_dir / "news_per_image_log.jsonl"

    csv_header = [
        "image_index",
        "total_images",
        "image_input",
        "score",
        "predicted_label",
        "predicted_label_name",
        "ground_truth_label",
        "ground_truth_label_name",
        "is_correct",
        "threshold",
        "running_ai",
        "running_real",
        "running_with_gt",
        "running_correct",
        "running_accuracy",
    ]

    import csv

    with per_image_csv.open("w", newline="", encoding="utf-8") as f_csv, per_image_jsonl.open(
        "w", encoding="utf-8"
    ) as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=csv_header)
        writer.writeheader()

        for idx, img_path in enumerate(image_paths, start=1):
            try:
                pred = handler({"inputs": str(img_path)})

                inferred_gt_label, inferred_gt_name = _infer_gt_from_filename(img_path.name)
                gt_label = 0 if assume_all_real else inferred_gt_label
                gt_name = "real" if assume_all_real else inferred_gt_name

                predicted_label = int(pred["predicted_label"])
                predicted_name = pred["predicted_label_name"]

                if predicted_label == 1:
                    running_ai += 1
                else:
                    running_real += 1

                is_correct: int | None = None
                if gt_label is not None:
                    running_with_gt += 1
                    is_correct = int(predicted_label == int(gt_label))
                    running_correct += is_correct

                running_acc = (
                    float(running_correct / running_with_gt) if running_with_gt > 0 else None
                )

                row = {
                    "score": float(pred["score"]),
                    "predicted_label": predicted_label,
                    "predicted_label_name": predicted_name,
                    "ground_truth_label": gt_label,
                    "ground_truth_label_name": gt_name,
                    "is_correct": is_correct,
                    "threshold": float(pred["threshold"]),
                    "image_input": str(img_path),
                    "image_url": "",
                    "source": "local_folder",
                    "organization_name": img_path.parent.name,
                    "sector": "news",
                    "subsector": "",
                    "img_alt_text": "",
                    "img_title": "",
                    "page_url": "",
                    "stored_path": str(img_path),
                }
                rows.append(row)

                log_row = {
                    "image_index": idx,
                    "total_images": len(image_paths),
                    "image_input": str(img_path),
                    "score": row["score"],
                    "predicted_label": predicted_label,
                    "predicted_label_name": predicted_name,
                    "ground_truth_label": gt_label,
                    "ground_truth_label_name": gt_name,
                    "is_correct": is_correct,
                    "threshold": row["threshold"],
                    "running_ai": running_ai,
                    "running_real": running_real,
                    "running_with_gt": running_with_gt,
                    "running_correct": running_correct,
                    "running_accuracy": running_acc,
                }
                writer.writerow(log_row)
                f_jsonl.write(json.dumps(log_row, ensure_ascii=False) + "\n")

                acc_txt = f"{running_acc:.4f}" if running_acc is not None else "n/a"
                gt_txt = gt_name if gt_name is not None else "unknown"
                ok_txt = str(is_correct) if is_correct is not None else "n/a"
                print(
                    " | ".join(
                        [
                            f"img={idx}/{len(image_paths)}",
                            f"score={row['score']:.6f}",
                            f"pred={predicted_name}",
                            f"gt={gt_txt}",
                            f"ok={ok_txt}",
                            f"ai={running_ai}",
                            f"real={running_real}",
                            f"acc={acc_txt}",
                        ]
                    ),
                    flush=True,
                )

            except Exception as ex:
                failures += 1
                print(
                    f"img={idx}/{len(image_paths)} | ERROR | input={img_path} | detail={ex}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No images could be classified from images_dir")

    pred_df = pd.DataFrame(rows)
    pred_csv = output_dir / "news_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    _plot_score_curve(pred_df, output_dir)
    _plot_score_histogram(pred_df, output_dir)
    _plot_testing_graphics(pred_df, output_dir)

    summary = {
        "images_dir": str(images_dir),
        "selected_images": int(len(image_paths)),
        "classified": int(len(pred_df)),
        "failed": int(failures),
        "assume_all_real": bool(assume_all_real),
        "threshold": float(threshold),
        "ai_count": int((pred_df["predicted_label"] == 1).sum()),
        "real_count": int((pred_df["predicted_label"] == 0).sum()),
        "ground_truth_real_count": int((pred_df["ground_truth_label"] == 0).sum())
        if pred_df["ground_truth_label"].notna().any()
        else None,
        "ground_truth_ai_count": int((pred_df["ground_truth_label"] == 1).sum())
        if pred_df["ground_truth_label"].notna().any()
        else None,
        "with_ground_truth": int(pred_df["ground_truth_label"].notna().sum()),
        "correct_with_ground_truth": int(pred_df["is_correct"].fillna(0).astype(int).sum())
        if "is_correct" in pred_df.columns
        else None,
        "score_mean": float(pred_df["score"].mean()),
        "score_median": float(pred_df["score"].median()),
        "score_min": float(pred_df["score"].min()),
        "score_max": float(pred_df["score"].max()),
        "artifacts": {
            "predictions_csv": str(pred_csv),
            "per_image_log_csv": str(per_image_csv),
            "per_image_log_jsonl": str(per_image_jsonl),
            "score_plot": str(output_dir / "score.png"),
            "score_histogram": str(output_dir / "score_histogram.png"),
            "testing_graphics": str(output_dir / "testing_graphics.png"),
        },
    }

    summary_json = output_dir / "news_analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _plot_score_curve(pred_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = pred_df.sort_values("score", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(11, 4.5))
    x = np.arange(1, len(ordered) + 1)
    plt.plot(x, ordered["score"].to_numpy(), color="#0b7285", linewidth=1.8)
    plt.axhline(float(ordered["threshold"].iloc[0]), color="#c92a2a", linestyle="--", linewidth=1.6)
    plt.title("Score")
    plt.xlabel("Image Rank (high to low score)")
    plt.ylabel("SPAI score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "score.png", dpi=170)
    plt.close()


def _plot_score_histogram(pred_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8.5, 5))
    plt.hist(pred_df["score"].to_numpy(), bins=30, color="#1971c2", edgecolor="white", alpha=0.9)
    plt.axvline(float(pred_df["threshold"].iloc[0]), color="#c92a2a", linestyle="--", linewidth=1.6)
    plt.title("Histograma de scores")
    plt.xlabel("SPAI score")
    plt.ylabel("Count")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "score_histogram.png", dpi=170)
    plt.close()


def _plot_testing_graphics(pred_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ai_count = int((pred_df["predicted_label"] == 1).sum())
    real_count = int((pred_df["predicted_label"] == 0).sum())
    axes[0].bar(["Real", "AI"], [real_count, ai_count], color=["#2f9e44", "#d9480f"])
    axes[0].set_title("Predicted Classes")
    axes[0].set_ylabel("Images")
    axes[0].grid(axis="y", alpha=0.2)

    cdf_scores = np.sort(pred_df["score"].to_numpy())
    cdf = np.arange(1, len(cdf_scores) + 1) / len(cdf_scores)
    axes[1].plot(cdf_scores, cdf, color="#5f3dc4", linewidth=2.0)
    axes[1].axvline(float(pred_df["threshold"].iloc[0]), color="#c92a2a", linestyle="--")
    axes[1].set_title("Testing Graphics: CDF")
    axes[1].set_xlabel("SPAI score")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].grid(alpha=0.25)

    true_label_series = pred_df.get("ground_truth_label")
    if true_label_series is not None and true_label_series.notna().any():
        from sklearn.metrics import confusion_matrix

        y_true = true_label_series.fillna(0).astype(int).to_numpy()
        y_pred = pred_df["predicted_label"].astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        im = axes[2].imshow(cm, cmap="Blues", aspect="auto")
        axes[2].set_xticks([0, 1])
        axes[2].set_yticks([0, 1])
        axes[2].set_xticklabels(["Real", "AI"])
        axes[2].set_yticklabels(["Real", "AI"])
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("True")
        axes[2].set_title("Testing Graphics: Confusion Matrix")
        for i in range(2):
            for j in range(2):
                axes[2].text(j, i, cm[i, j], ha="center", va="center", fontweight="bold")
        plt.colorbar(im, ax=axes[2])
    else:
        top_orgs = pred_df["organization_name"].fillna("unknown").value_counts().head(10).index
        subset = pred_df[pred_df["organization_name"].fillna("unknown").isin(top_orgs)]
        grouped = [subset[subset["organization_name"] == org]["score"].to_numpy() for org in top_orgs]
        axes[2].boxplot(grouped, labels=top_orgs, showfliers=False)
        axes[2].tick_params(axis="x", rotation=55)
        axes[2].set_title("Testing Graphics: Score by News Site")
        axes[2].set_ylabel("SPAI score")
        axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "testing_graphics.png", dpi=170)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify crawled news images and export separate graphics"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--metadata-csv", type=Path, help="Crawler metadata CSV")
    input_group.add_argument("--images-dir", type=Path, help="Local folder with news images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write analysis")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/fhome/aaasidar/spai-hf"),
        help="SPAI model root directory",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Decision threshold")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of news images to classify (0 = all)",
    )
    parser.add_argument(
        "--assume-all-real",
        action="store_true",
        help="Assume every input image has ground-truth label 0 (real)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.images_dir is not None:
        summary = classify_images_dir(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            threshold=args.threshold,
            max_images=args.max_images,
            assume_all_real=args.assume_all_real,
        )
    else:
        summary = classify_news_images(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            threshold=args.threshold,
            max_images=args.max_images,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
