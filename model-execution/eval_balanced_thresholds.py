from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image

from inference import EndpointHandler


def _image_to_input(image_value: Any) -> Any:
    if isinstance(image_value, Image.Image):
        return image_value

    if isinstance(image_value, dict):
        if "path" in image_value:
            return image_value["path"]
        if "bytes" in image_value:
            return image_value["bytes"]
        if "url" in image_value:
            return image_value["url"]

    return image_value


def _resolve_split_name(requested_split: str | None, available_splits: list[str]) -> tuple[str, str]:
    if not available_splits:
        raise ValueError("Dataset has no available splits")

    if requested_split is None:
        effective = "test" if "test" in available_splits else available_splits[0]
        return effective, effective

    if requested_split in available_splits:
        return requested_split, requested_split

    if requested_split == "test" and "train" in available_splits:
        print(
            "Requested split 'test' is not available. Using split 'train' as alias for test inference.",
            flush=True,
        )
        return requested_split, "train"

    raise ValueError(f"Bad split: {requested_split}. Available splits: {available_splits}")


def _detect_image_columns(item: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key, value in item.items():
        if isinstance(value, Image.Image):
            columns.append(key)
        elif isinstance(value, dict) and any(k in value for k in ("path", "bytes", "url")):
            columns.append(key)
    return columns


def _parse_thresholds(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("No thresholds provided")

    thresholds = sorted({float(x) for x in values})
    for t in thresholds:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"Threshold out of range [0,1]: {t}")
    return thresholds


def _threshold_to_col(threshold: float) -> str:
    return f"pred_t_{str(threshold).replace('.', '_')}"


def _compute_metrics(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    pred = (df["score"] >= threshold).astype(int)
    gt = pd.to_numeric(df["gt_label"], errors="coerce").astype("Int64")

    out: dict[str, Any] = {
        "threshold": threshold,
        "total": int(len(df)),
        "pred_ai": int((pred == 1).sum()),
        "pred_real": int((pred == 0).sum()),
        "with_gt": int(gt.notna().sum()),
    }

    if gt.notna().any():
        gtv = gt[gt.notna()].astype(int)
        prv = pred[gt.notna()].astype(int)

        tp = int(((gtv == 1) & (prv == 1)).sum())
        fn = int(((gtv == 1) & (prv == 0)).sum())
        tn = int(((gtv == 0) & (prv == 0)).sum())
        fp = int(((gtv == 0) & (prv == 1)).sum())

        accuracy = float((gtv == prv).mean())
        recall_ai = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        fn_rate = float(fn / (tp + fn)) if (tp + fn) > 0 else None
        specificity_real = float(tn / (tn + fp)) if (tn + fp) > 0 else None
        fpr_real = float(fp / (tn + fp)) if (tn + fp) > 0 else None

        out.update(
            {
                "accuracy": accuracy,
                "ai_tp": tp,
                "ai_fn": fn,
                "real_tn": tn,
                "real_fp": fp,
                "ai_recall": recall_ai,
                "ai_fn_rate": fn_rate,
                "real_specificity": specificity_real,
                "real_fpr": fpr_real,
            }
        )
    else:
        out.update(
            {
                "accuracy": None,
                "ai_tp": None,
                "ai_fn": None,
                "real_tn": None,
                "real_fp": None,
                "ai_recall": None,
                "ai_fn_rate": None,
                "real_specificity": None,
                "real_fpr": None,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced multi-threshold evaluation over HF datasets."
    )
    parser.add_argument("--dataset", type=str, default="BaiqiL/GenAI-Bench")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument(
        "--image-columns",
        type=str,
        default=None,
        help="Comma-separated list of image columns. If omitted, image-like columns are auto-detected.",
    )
    parser.add_argument("--total-images", type=int, default=120)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated thresholds in [0,1]",
    )
    parser.add_argument("--gt-label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--model-dir", type=str, default="/fhome/aaasidar/spai-hf")
    parser.add_argument("--output-dir", type=Path, default=Path("balanced_eval"))
    parser.add_argument("--output-prefix", type=str, default="balanced")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--log-threshold",
        type=float,
        default=0.5,
        help="Threshold used for online per-image log prediction summaries",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    if args.log_threshold < 0.0 or args.log_threshold > 1.0:
        raise ValueError("log-threshold must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handler = EndpointHandler(path=args.model_dir)

    dataset_dict = load_dataset(args.dataset, streaming=True)
    available_splits = list(dataset_dict.keys())
    requested_split, split_name = _resolve_split_name(args.split, available_splits)
    dataset = dataset_dict[split_name]

    dataset_iter = iter(dataset)
    try:
        first_item = next(dataset_iter)
    except StopIteration as exc:
        raise ValueError(
            f"Split '{split_name}' from dataset '{args.dataset}' is empty"
        ) from exc

    detected_columns = _detect_image_columns(first_item)
    if args.image_columns:
        image_columns = [c.strip() for c in args.image_columns.split(",") if c.strip()]
    else:
        image_columns = detected_columns

    if not image_columns:
        raise ValueError("No image columns found. Use --image-columns to specify them.")

    missing = [c for c in image_columns if c not in first_item]
    if missing:
        raise KeyError(
            f"Columns not found in dataset item: {missing}. Available: {list(first_item.keys())}"
        )

    n_cols = len(image_columns)
    if args.total_images < n_cols:
        raise ValueError(
            f"total-images ({args.total_images}) must be >= number of image columns ({n_cols})"
        )

    per_col = args.total_images // n_cols
    remainder = args.total_images % n_cols
    targets = {col: per_col + (1 if idx < remainder else 0) for idx, col in enumerate(image_columns)}
    counts = {col: 0 for col in image_columns}

    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Requested split: {requested_split} | Resolved split: {split_name}", flush=True)
    print(f"Image columns: {image_columns}", flush=True)
    print(f"Total target images: {args.total_images}", flush=True)
    print(f"Per-column targets: {targets}", flush=True)
    print(f"Thresholds: {thresholds}", flush=True)

    rows: list[dict[str, Any]] = []
    start_time = pd.Timestamp.utcnow().timestamp()
    ai_count = 0
    real_count = 0
    correct_count = 0

    # Build one fixed, balanced sample and reuse it for every threshold.
    for row_idx, item in enumerate(chain([first_item], dataset_iter)):
        all_done = all(counts[col] >= targets[col] for col in image_columns)
        if all_done:
            break

        for col in image_columns:
            if counts[col] >= targets[col]:
                continue
            if col not in item:
                continue

            image_value = _image_to_input(item[col])
            prediction = handler({"inputs": image_value})
            score = float(prediction["score"])
            pred_for_log = int(score >= args.log_threshold)
            pred_for_log_name = "ai-generated" if pred_for_log == 1 else "real"
            if pred_for_log == 1:
                ai_count += 1
            else:
                real_count += 1
            is_correct = int(pred_for_log == args.gt_label)
            correct_count += is_correct
            processed = len(rows) + 1

            sample = {
                "dataset": args.dataset,
                "requested_split": requested_split,
                "resolved_split": split_name,
                "row_index": row_idx,
                "image_model": col,
                "image_ref": str(item.get("Index", row_idx)),
                "score": score,
                "gt_label": args.gt_label,
                "prompt": item.get("Prompt"),
                "tags_json": json.dumps(item.get("Tags", {}), ensure_ascii=True, sort_keys=True),
                "human_ratings_json": json.dumps(item.get("HumanRatings", {}), ensure_ascii=True, sort_keys=True),
            }
            rows.append(sample)
            counts[col] += 1

            if len(rows) % max(args.log_every, 1) == 0:
                elapsed = pd.Timestamp.utcnow().timestamp() - start_time
                ips = processed / elapsed if elapsed > 0 else 0.0
                acc = correct_count / processed if processed > 0 else 0.0
                print(
                    " | ".join(
                        [
                            f"img={processed}",
                            f"ref={item.get('Index', row_idx)}",
                            f"model={col}",
                            f"score={score:.6f}",
                            f"pred={pred_for_log_name}",
                            f"gt={'ai-generated' if args.gt_label == 1 else 'real'}",
                            f"ok={is_correct}",
                            f"ai={ai_count}",
                            f"real={real_count}",
                            f"acc={acc:.4f}",
                            f"ips={ips:.2f}",
                        ]
                    ),
                    flush=True,
                )

            all_done = all(counts[c] >= targets[c] for c in image_columns)
            if all_done:
                break

    df = pd.DataFrame(rows)

    # Keep the sample ordering stable so every threshold is evaluated on the same images.
    df = df.sort_values(["image_model", "row_index", "image_ref"], kind="stable").reset_index(drop=True)

    if len(df) < args.total_images:
        print(
            f"Warning: collected {len(df)} images, below requested total {args.total_images}.",
            flush=True,
        )

    for threshold in thresholds:
        df[_threshold_to_col(threshold)] = (df["score"] >= threshold).astype(int)

    global_metrics = [_compute_metrics(df, t) for t in thresholds]
    global_df = pd.DataFrame(global_metrics)

    model_metrics: list[dict[str, Any]] = []
    for model_name, group in df.groupby("image_model"):
        for threshold in thresholds:
            row = _compute_metrics(group, threshold)
            row["image_model"] = model_name
            model_metrics.append(row)
    by_model_df = pd.DataFrame(model_metrics)

    scores_path = args.output_dir / f"{args.output_prefix}_scores.csv"
    global_path = args.output_dir / f"{args.output_prefix}_thresholds_global.csv"
    by_model_path = args.output_dir / f"{args.output_prefix}_thresholds_by_model.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.json"

    df.to_csv(scores_path, index=False)
    global_df.to_csv(global_path, index=False)
    by_model_df.to_csv(by_model_path, index=False)

    summary = {
        "dataset": args.dataset,
        "requested_split": requested_split,
        "resolved_split": split_name,
        "total_requested": args.total_images,
        "total_collected": int(len(df)),
        "image_columns": image_columns,
        "column_targets": targets,
        "column_counts": counts,
        "balanced_across_models": True,
        "shared_sample_for_all_thresholds": True,
        "thresholds": thresholds,
        "outputs": {
            "scores_csv": str(scores_path),
            "thresholds_global_csv": str(global_path),
            "thresholds_by_model_csv": str(by_model_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print("Done.", flush=True)
    print(f"Scores: {scores_path}", flush=True)
    print(f"Global thresholds: {global_path}", flush=True)
    print(f"Thresholds by model: {by_model_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
