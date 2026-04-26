from __future__ import annotations

import argparse
import json
import os
import signal
from itertools import chain
from pathlib import Path
import time
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image

from inference import EndpointHandler


DATASET_PROFILES: dict[str, dict[str, Any]] = {
    # Balanced dataset with 0 = AI-generated and 1 = Real.
    "Parveshiiii/AI-vs-Real": {
        "gt_label_column": "binary_label",
        "label_names": {
            0: "ai-generated",
            1: "real",
        },
        # SPAI model convention is 1 = AI-generated, 0 = real.
        "label_to_model": {
            0: 1,
            1: 0,
        },
    },
    # ClassLabel names: 0=AiArtData (AI), 1=RealArt (Real).
    "Hemg/AI-Generated-vs-Real-Images-Datasets": {
        "gt_label_column": "label",
        "label_names": {
            0: "ai-generated",
            1: "real",
        },
        # SPAI model convention is 1 = AI-generated, 0 = real.
        "label_to_model": {
            0: 1,
            1: 0,
        },
    },
    # ClassLabel names: 0=Artificial, 1=Deepfake, 2=Real.
    "prithivMLmods/AI-vs-Deepfake-vs-Real": {
        "gt_label_column": "label",
        "label_names": {
            0: "ai-generated",
            1: "deepfake",
            2: "real",
        },
        # Collapse artificial + deepfake to the positive AI-like class.
        "label_to_model": {
            0: 1,
            1: 1,
            2: 0,
        },
    },
}


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


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Image.Image):
        return {
            "_type": "PIL.Image",
            "mode": value.mode,
            "size": [value.width, value.height],
        }

    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]

    return str(value)


def _infer_dataset_profile(dataset_name: str) -> dict[str, Any]:
    if dataset_name in DATASET_PROFILES:
        return DATASET_PROFILES[dataset_name]
    return {}


def _resolve_gt_label(
    item: dict[str, Any],
    dataset: str,
    gt_label_column: str | None,
    default_gt_label: int | None,
    label_to_model_override: dict[int, int] | None = None,
) -> tuple[int | None, int | None, str | None, str]:
    profile = _infer_dataset_profile(dataset)
    label_to_model = label_to_model_override or profile.get("label_to_model", {})

    def _label_name_for(gt_value: int) -> str:
        label_names = profile.get("label_names", {})
        if gt_value in label_names:
            return label_names[gt_value]
        return "ai-generated" if gt_value == 0 else "real" if gt_value == 1 else str(gt_value)

    def _to_model_label(gt_value: int) -> int:
        mapped = label_to_model.get(gt_value)
        if mapped is not None:
            return int(mapped)
        return gt_value

    if gt_label_column and gt_label_column in item:
        raw = item[gt_label_column]
        try:
            gt = int(raw)
        except Exception as exc:
            raise ValueError(
                f"Could not convert gt label from column '{gt_label_column}' with value '{raw}'"
            ) from exc
        return gt, _to_model_label(gt), _label_name_for(gt), f"column:{gt_label_column}"

    profile_gt_column = profile.get("gt_label_column")
    if profile_gt_column and profile_gt_column in item:
        raw = item[profile_gt_column]
        try:
            gt = int(raw)
        except Exception as exc:
            raise ValueError(
                f"Could not convert gt label from profile column '{profile_gt_column}' with value '{raw}'"
            ) from exc

        return gt, _to_model_label(gt), _label_name_for(gt), f"dataset_profile:{profile_gt_column}"

    if default_gt_label is not None:
        return (
            default_gt_label,
            _to_model_label(default_gt_label),
            _label_name_for(default_gt_label),
            "arg:default_gt_label",
        )

    return None, None, None, "none"


def _extract_image_ref(item: dict[str, Any], selected_image_column: str, idx: int) -> str:
    for key in ("id", "image_id", "file_name", "path", "url", "Index"):
        if key in item:
            return str(item[key])

    raw_image = item.get(selected_image_column)
    if isinstance(raw_image, dict):
        for key in ("path", "url"):
            if key in raw_image:
                return str(raw_image[key])

    return f"index:{idx}"


def _append_csv_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    df.to_csv(csv_path, mode="a", header=header, index=False)


def _append_jsonl_row(jsonl_path: Path, row: dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _export_image(image_value: Any, export_dir: Path, index: int, export_size: int) -> str:
    export_dir.mkdir(parents=True, exist_ok=True)

    image = image_value if isinstance(image_value, Image.Image) else None
    if image is None:
        raise ValueError("Image export only supports PIL images in this workflow.")

    original_format = (image.format or "").lower()

    if export_size > 0:
        image = image.resize((export_size, export_size), Image.Resampling.LANCZOS)

    suffix = ".jpg"
    image_format = original_format
    if image_format in {"jpeg", "jpg"}:
        suffix = ".jpg"
    elif image_format in {"png", "webp", "gif", "bmp", "tiff"}:
        suffix = f".{image_format}"

    file_name = f"image_{index:06d}{suffix}"
    export_path = export_dir / file_name
    image.save(export_path)
    return str(export_path)


def _resolve_image_column(item: dict[str, Any], requested_column: str) -> str:
    if requested_column in item:
        return requested_column

    image_like_columns = [
        key
        for key, value in item.items()
        if isinstance(value, Image.Image)
        or (isinstance(value, dict) and any(k in value for k in ("path", "bytes", "url")))
    ]

    # Keep default behavior, but be user-friendly when datasets do not use "image".
    if requested_column == "image" and image_like_columns:
        chosen = image_like_columns[0]
        print(
            f"Column 'image' not found. Using detected image column '{chosen}'. "
            f"Detected image-like columns: {image_like_columns}"
        )
        return chosen

    available_columns = list(item.keys())
    raise KeyError(
        f"Column '{requested_column}' not found in dataset item. "
        f"Available columns: {available_columns}. "
        f"Detected image-like columns: {image_like_columns or 'none'}"
    )


def _resolve_split_name(requested_split: str | None, available_splits: list[str]) -> tuple[str, str]:
    if not available_splits:
        raise ValueError("Dataset has no available splits")

    if requested_split is None:
        effective = "test" if "test" in available_splits else available_splits[0]
        return effective, effective

    if requested_split in available_splits:
        return requested_split, requested_split

    # Common case in HF datasets: only 'train' exists.
    if requested_split == "test" and "train" in available_splits:
        print(
            "Requested split 'test' is not available. "
            "Using split 'train' as alias for test inference."
        )
        return requested_split, "train"

    raise ValueError(
        f"Bad split: {requested_split}. Available splits: {available_splits}"
    )


def _parse_excluded_labels(raw: str | None) -> set[int]:
    if raw is None or not raw.strip():
        return set()
    values: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    return values


def _parse_label_to_model_map(raw: str | None) -> dict[int, int]:
    if raw is None or not raw.strip():
        return {}

    mapping: dict[int, int] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid label mapping token '{token}'. Expected format 'src:dst'."
            )

        src_str, dst_str = token.split(":", 1)
        src = int(src_str.strip())
        dst = int(dst_str.strip())
        if dst not in {0, 1}:
            raise ValueError(f"Mapped model label must be 0 or 1. Got: {dst}")
        mapping[src] = dst

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAI inference over a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, default="prithivMLmods/AI-vs-Deepfake-vs-Real")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of images to classify from the dataset",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("results.csv"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("results.jsonl"),
        help="Image-by-image structured output (one JSON object per line)",
    )
    parser.add_argument("--model-dir", type=str, default="/fhome/aaasidar/spai-hf")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--gt-label-column",
        type=str,
        default=None,
        help="Dataset column containing ground-truth label (auto-detected from profile when omitted)",
    )
    parser.add_argument(
        "--default-gt-label",
        type=int,
        choices=[0, 1],
        default=None,
        help="Default ground-truth label for all rows when dataset has no label column",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print one inference log line every N images (1 = every image)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Flush CSV to disk every N processed images",
    )
    parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Overwrite existing output files instead of appending",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize dataset order before selecting images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when --randomize is enabled",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2000,
        help="Shuffle buffer size for streaming datasets when --randomize is enabled",
    )
    parser.add_argument(
        "--balanced-by-label",
        action="store_true",
        help="Select a balanced sample across labels (e.g., AI and Real) up to --max-images",
    )
    parser.add_argument(
        "--exclude-raw-labels",
        type=str,
        default="",
        help="Comma-separated raw dataset labels to exclude (e.g., '1' to exclude Deepfake)",
    )
    parser.add_argument(
        "--label-to-model-map",
        type=str,
        default="",
        help="Override label mapping in format 'src:dst,src:dst' (e.g., '0:1,1:0')",
    )
    parser.add_argument(
        "--export-image-dir",
        type=Path,
        default=None,
        help="Optional directory to save all sampled images in one mixed folder",
    )
    parser.add_argument(
        "--export-image-size",
        type=int,
        default=224,
        help="Size in pixels for exported preview images",
    )
    args = parser.parse_args()
    excluded_raw_labels = _parse_excluded_labels(args.exclude_raw_labels)
    label_to_model_override = _parse_label_to_model_map(args.label_to_model_map)

    if args.threshold is not None:
        os.environ["SPAI_THRESHOLD"] = str(args.threshold)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_outputs:
        if args.output_csv.exists():
            args.output_csv.unlink()
        if args.output_jsonl.exists():
            args.output_jsonl.unlink()

    handler = EndpointHandler(path=args.model_dir)
    dataset_dict = load_dataset(args.dataset, streaming=True)
    available_splits = list(dataset_dict.keys())
    requested_split, split_name = _resolve_split_name(args.split, available_splits)

    dataset = dataset_dict[split_name]
    if args.randomize:
        try:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=max(args.shuffle_buffer_size, 100))
            print(
                f"Randomized dataset order (seed={args.seed}, buffer_size={max(args.shuffle_buffer_size, 100)})",
                flush=True,
            )
        except TypeError:
            dataset = dataset.shuffle(seed=args.seed)
            print(f"Randomized dataset order (seed={args.seed})", flush=True)

    print(
        f"Using split '{split_name}' from dataset '{args.dataset}' "
        f"(requested: '{requested_split}')"
    , flush=True)

    dataset_iter = iter(dataset)
    try:
        first_item = next(dataset_iter)
    except StopIteration as exc:
        raise ValueError(
            f"Split '{split_name}' from dataset '{args.dataset}' is empty"
        ) from exc

    selected_image_column = _resolve_image_column(first_item, args.image_column)
    print(f"Using image column '{selected_image_column}'", flush=True)

    profile = _infer_dataset_profile(args.dataset)
    if profile:
        print(f"Using dataset profile for '{args.dataset}': {profile}", flush=True)
    if label_to_model_override:
        print(f"Using label_to_model override: {label_to_model_override}", flush=True)
    if excluded_raw_labels:
        print(f"Excluding raw labels: {sorted(excluded_raw_labels)}", flush=True)

    start_time = time.time()
    ai_count = 0
    real_count = 0
    correct_count = 0
    with_gt_count = 0
    total_processed = 0
    pending_rows: list[dict[str, Any]] = []
    stop_requested = False
    selected_by_model_label: dict[int, int] = {0: 0, 1: 0}
    balanced_targets: dict[int, int] | None = None
    if args.balanced_by_label:
        max_images = max(args.max_images, 1)
        balanced_targets = {
            0: max_images // 2,
            1: max_images - (max_images // 2),
        }
        print(f"Balanced sampling enabled with targets: {balanced_targets}", flush=True)

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"Received signal {signum}. Finishing current item and flushing pending results...",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    for idx, item in enumerate(chain([first_item], dataset_iter)):
        if stop_requested:
            break
        if total_processed >= max(args.max_images, 1):
            break

        gt_label_raw, gt_label, gt_label_name, gt_source = _resolve_gt_label(
            item=item,
            dataset=args.dataset,
            gt_label_column=args.gt_label_column,
            default_gt_label=args.default_gt_label,
            label_to_model_override=label_to_model_override,
        )
        if gt_label_raw is not None and gt_label_raw in excluded_raw_labels:
            continue
        if balanced_targets is not None:
            if gt_label is None or gt_label not in balanced_targets:
                continue
            if selected_by_model_label[gt_label] >= balanced_targets[gt_label]:
                continue

        raw_image_value = item[selected_image_column]
        image_value = _image_to_input(raw_image_value)
        prediction = handler({"inputs": image_value})

        predicted_label = int(prediction["predicted_label"])
        if predicted_label == 1:
            ai_count += 1
        else:
            real_count += 1

        is_correct: int | None = None
        if gt_label is not None:
            with_gt_count += 1
            is_correct = int(gt_label == predicted_label)
            if is_correct == 1:
                correct_count += 1
        if gt_label in selected_by_model_label:
            selected_by_model_label[gt_label] += 1

        image_ref = _extract_image_ref(item, selected_image_column, idx)
        metadata = {
            key: _normalize_for_json(value)
            for key, value in item.items()
            if key != selected_image_column
        }

        exported_image_path = None
        if args.export_image_dir is not None:
            exported_image_path = _export_image(
                raw_image_value,
                args.export_image_dir,
                total_processed + 1,
                max(args.export_image_size, 1),
            )

        row: dict[str, Any] = {
            "index": idx,
            "sample_index": total_processed + 1,
            "dataset": args.dataset,
            "requested_split": requested_split,
            "resolved_split": split_name,
            "image_ref": image_ref,
            "image_column": selected_image_column,
            "score": prediction["score"],
            "predicted_label": predicted_label,
            "predicted_label_name": prediction["predicted_label_name"],
            "threshold": prediction.get("threshold", args.threshold),
            "gt_label": gt_label,
            "gt_label_raw": gt_label_raw,
            "gt_label_name": gt_label_name,
            "gt_label_source": gt_source,
            "is_correct": is_correct,
            "metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
            "exported_image_path": exported_image_path,
            "export_image_size": args.export_image_size if args.export_image_dir is not None else None,
        }
        for key in ["id", "image_id", "file_name", "path", "label", "class"]:
            if key in item:
                row[key] = item[key]

        pending_rows.append(row)
        _append_jsonl_row(args.output_jsonl, row)

        save_every = max(args.save_every, 1)
        processed = total_processed + 1
        total_processed = processed
        if processed % save_every == 0:
            _append_csv_rows(args.output_csv, pending_rows)
            pending_rows = []

        log_every = max(args.log_every, 1)
        if processed % log_every == 0:
            elapsed = time.time() - start_time
            ips = processed / elapsed if elapsed > 0 else 0.0
            accuracy_text = "n/a"
            if with_gt_count > 0:
                accuracy_text = f"{(correct_count / with_gt_count):.4f}"
            print(
                " | ".join(
                    [
                        f"img={processed}",
                        f"ref={image_ref}",
                        f"score={float(prediction['score']):.6f}",
                        f"pred={prediction['predicted_label_name']}",
                        f"gt={gt_label_name if gt_label_name is not None else 'unknown'}",
                        f"ok={is_correct if is_correct is not None else 'n/a'}",
                        f"ai={ai_count}",
                        f"real={real_count}",
                        f"acc={accuracy_text}",
                        f"ips={ips:.2f}",
                    ]
                ),
                flush=True,
            )

    if pending_rows:
        _append_csv_rows(args.output_csv, pending_rows)

    total_elapsed = time.time() - start_time
    total = total_processed
    final_ips = (total / total_elapsed) if total_elapsed > 0 else 0.0
    summary = {
        "dataset": args.dataset,
        "requested_split": requested_split,
        "resolved_split": split_name,
        "image_column": selected_image_column,
        "max_images": args.max_images,
        "randomize": args.randomize,
        "seed": args.seed,
        "balanced_by_label": args.balanced_by_label,
        "excluded_raw_labels": sorted(excluded_raw_labels),
        "label_to_model_override": label_to_model_override,
        "selected_model_label_counts": selected_by_model_label,
        "total": total,
        "predicted_ai": ai_count,
        "predicted_real": real_count,
        "with_gt": with_gt_count,
        "correct": correct_count,
        "accuracy": (correct_count / with_gt_count) if with_gt_count > 0 else None,
        "elapsed_seconds": total_elapsed,
        "avg_ips": final_ips,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        f"Finished inference: total={total}, ai={ai_count}, real={real_count}, "
        f"with_gt={with_gt_count}, correct={correct_count}, elapsed={total_elapsed:.2f}s, avg_ips={final_ips:.2f}",
        flush=True,
    )
    print(f"CSV results saved to {args.output_csv}", flush=True)
    print(f"JSONL results saved to {args.output_jsonl}", flush=True)
    print(f"Run summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()