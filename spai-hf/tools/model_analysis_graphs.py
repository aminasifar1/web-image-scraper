#!/usr/bin/env python3
"""Generate per-model post-test analysis graphs using matplotlib and networkx.

Outputs (per model):
- histogram_scores.png
- score_curve.png
- confusion_network.png
- summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _pick_first_existing(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for col in columns:
        if col in frame.columns:
            return col
    return None


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _load_and_merge(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "source_file" not in df.columns:
            df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty CSV inputs found")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def _resolve_model_column(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Requested model column '{user_col}' not found")
        return user_col

    model_col = _pick_first_existing(["image_column", "model", "source", "source_file"], df)
    if model_col is None:
        df["model_name"] = "single_model"
        return "model_name"
    return model_col


def _resolve_score_column(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Requested score column '{user_col}' not found")
        return user_col

    score_col = _pick_first_existing(["score", "spai", "prediction", "confidence"], df)
    if score_col is None:
        raise ValueError("No score column found. Expected one of: score, spai, prediction, confidence")
    return score_col


def _resolve_pred_column(df: pd.DataFrame, user_col: str | None, score_col: str, threshold: float) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Requested predicted label column '{user_col}' not found")
        return user_col

    pred_col = _pick_first_existing(["predicted_label", "pred_label", "prediction_label"], df)
    if pred_col is None:
        df["predicted_label"] = (pd.to_numeric(df[score_col], errors="coerce") >= threshold).astype(int)
        return "predicted_label"
    return pred_col


def _resolve_gt_column(df: pd.DataFrame, user_col: str | None) -> str | None:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Requested ground-truth column '{user_col}' not found")
        return user_col

    return _pick_first_existing(["gt_label", "ground_truth_label", "label"], df)


def _plot_histogram(model_df: pd.DataFrame, score_col: str, threshold: float, out_file: Path) -> None:
    scores = pd.to_numeric(model_df[score_col], errors="coerce").dropna().to_numpy()

    plt.figure(figsize=(8.5, 5))
    plt.hist(scores, bins=30, color="#1971c2", edgecolor="white", alpha=0.9)
    plt.axvline(threshold, color="#c92a2a", linestyle="--", linewidth=1.5)
    plt.title("Histogram of Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def _plot_score_curve(model_df: pd.DataFrame, score_col: str, threshold: float, out_file: Path) -> None:
    ordered = model_df.copy()
    ordered[score_col] = pd.to_numeric(ordered[score_col], errors="coerce")
    ordered = ordered.dropna(subset=[score_col]).sort_values(score_col, ascending=False).reset_index(drop=True)

    x = np.arange(1, len(ordered) + 1)
    y = ordered[score_col].to_numpy()

    plt.figure(figsize=(10.5, 4.5))
    plt.plot(x, y, color="#0b7285", linewidth=1.8)
    plt.axhline(threshold, color="#c92a2a", linestyle="--", linewidth=1.5)
    plt.title("Score Curve")
    plt.xlabel("Sample rank")
    plt.ylabel("Score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def _plot_confusion_network(
    model_df: pd.DataFrame,
    pred_col: str,
    gt_col: str | None,
    out_file: Path,
) -> dict[str, int] | None:
    if gt_col is None:
        return None

    y_true = pd.to_numeric(model_df[gt_col], errors="coerce").dropna().astype(int)
    valid_idx = y_true.index
    y_pred = pd.to_numeric(model_df.loc[valid_idx, pred_col], errors="coerce").fillna(0).astype(int)

    counts = {
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
    }

    G = nx.DiGraph()
    left_nodes = ["GT:Real", "GT:AI"]
    right_nodes = ["Pred:Real", "Pred:AI"]
    for node in left_nodes + right_nodes:
        G.add_node(node)

    edge_map = {
        ("GT:Real", "Pred:Real"): counts["TN"],
        ("GT:Real", "Pred:AI"): counts["FP"],
        ("GT:AI", "Pred:Real"): counts["FN"],
        ("GT:AI", "Pred:AI"): counts["TP"],
    }

    for (u, v), w in edge_map.items():
        if w > 0:
            G.add_edge(u, v, weight=w)

    pos = {
        "GT:Real": (0.0, 0.8),
        "GT:AI": (0.0, 0.2),
        "Pred:Real": (1.0, 0.8),
        "Pred:AI": (1.0, 0.2),
    }

    plt.figure(figsize=(8, 5.5))
    nx.draw_networkx_nodes(G, pos, node_color=["#74c0fc", "#74c0fc", "#ffd8a8", "#ffd8a8"], node_size=2200)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")

    if G.number_of_edges() > 0:
        widths = [max(1.5, min(10.0, G[u][v]["weight"] / 5.0)) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle="-|>", arrowsize=18)
        labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    plt.title("Confusion Network (GT -> Pred)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()

    return counts


def _prepare_gt_pred_score(
    model_df: pd.DataFrame,
    pred_col: str,
    gt_col: str | None,
    score_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if gt_col is None:
        return None

    gt_series = pd.to_numeric(model_df[gt_col], errors="coerce")
    pred_series = pd.to_numeric(model_df[pred_col], errors="coerce")
    score_series = pd.to_numeric(model_df[score_col], errors="coerce")

    valid_mask = gt_series.notna() & pred_series.notna() & score_series.notna()
    if not bool(valid_mask.any()):
        return None

    y_true = gt_series[valid_mask].astype(int).to_numpy()
    y_pred = pred_series[valid_mask].astype(int).to_numpy()
    y_score = score_series[valid_mask].to_numpy()
    return y_true, y_pred, y_score


def _plot_accuracy_curve(y_true: np.ndarray, y_pred: np.ndarray, out_file: Path) -> float:
    correct = (y_true == y_pred).astype(float)
    cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)

    plt.figure(figsize=(10.5, 4.5))
    plt.plot(np.arange(1, len(cumulative_accuracy) + 1), cumulative_accuracy, color="#2b8a3e", linewidth=1.8)
    plt.ylim(0, 1)
    plt.title("Accuracy Curve (Cumulative)")
    plt.xlabel("Sample index")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()

    return float(cumulative_accuracy[-1])


def _plot_loss_curve(y_true: np.ndarray, y_score: np.ndarray, out_file: Path) -> float:
    eps = 1e-7
    y_score = np.clip(y_score, eps, 1.0 - eps)
    per_sample_loss = -(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))
    cumulative_loss = np.cumsum(per_sample_loss) / np.arange(1, len(per_sample_loss) + 1)

    plt.figure(figsize=(10.5, 4.5))
    plt.plot(np.arange(1, len(cumulative_loss) + 1), cumulative_loss, color="#c92a2a", linewidth=1.8)
    plt.title("Loss Curve (Cumulative BCE)")
    plt.xlabel("Sample index")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()

    return float(cumulative_loss[-1])


def _plot_predicted_vs_actual_scatter(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    out_file: Path,
) -> None:
    rng = np.random.default_rng(seed=42)
    x_jitter = y_true + rng.normal(0.0, 0.04, size=len(y_true))

    plt.figure(figsize=(8.5, 5.5))
    colors = np.where(y_true == 1, "#e8590c", "#1c7ed6")
    plt.scatter(x_jitter, y_score, alpha=0.7, s=22, c=colors, edgecolors="none")
    plt.axhline(threshold, color="#c92a2a", linestyle="--", linewidth=1.4)
    plt.xlim(-0.35, 1.35)
    plt.ylim(0, 1)
    plt.xticks([0, 1], ["Actual: Real (0)", "Actual: AI (1)"])
    plt.title("Predicted vs Actual (Score vs GT Label)")
    plt.ylabel("Predicted score")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def build_model_analysis(
    input_csvs: list[Path],
    output_dir: Path,
    model_col: str | None,
    score_col: str | None,
    pred_col: str | None,
    gt_col: str | None,
    threshold: float,
) -> dict[str, object]:
    df = _load_and_merge(input_csvs)

    resolved_model_col = _resolve_model_column(df, model_col)
    resolved_score_col = _resolve_score_column(df, score_col)
    resolved_pred_col = _resolve_pred_column(df, pred_col, resolved_score_col, threshold)
    resolved_gt_col = _resolve_gt_column(df, gt_col)

    output_dir.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, object] = {
        "rows": int(len(df)),
        "model_column": resolved_model_col,
        "score_column": resolved_score_col,
        "predicted_label_column": resolved_pred_col,
        "ground_truth_column": resolved_gt_col,
        "models": {},
    }

    grouped = df.groupby(resolved_model_col, dropna=False)
    for model_name, model_df in grouped:
        model_name_str = str(model_name)
        safe_name = _sanitize_name(model_name_str)
        model_out = output_dir / safe_name
        model_out.mkdir(parents=True, exist_ok=True)

        _plot_histogram(model_df, resolved_score_col, threshold, model_out / "histogram_scores.png")
        _plot_score_curve(model_df, resolved_score_col, threshold, model_out / "score_curve.png")
        cm_counts = _plot_confusion_network(model_df, resolved_pred_col, resolved_gt_col, model_out / "confusion_network.png")

        curves_payload = _prepare_gt_pred_score(model_df, resolved_pred_col, resolved_gt_col, resolved_score_col)
        has_gt_metrics = curves_payload is not None
        final_accuracy = None
        final_loss = None
        if curves_payload is not None:
            y_true, y_pred, y_score = curves_payload
            final_accuracy = _plot_accuracy_curve(y_true, y_pred, model_out / "accuracy_curve.png")
            final_loss = _plot_loss_curve(y_true, y_score, model_out / "loss_curve.png")
            _plot_predicted_vs_actual_scatter(
                y_true,
                y_score,
                threshold,
                model_out / "scatter_predicted_vs_actual.png",
            )

        scores = pd.to_numeric(model_df[resolved_score_col], errors="coerce").dropna()
        model_summary: dict[str, object] = {
            "rows": int(len(model_df)),
            "score_mean": float(scores.mean()) if not scores.empty else None,
            "score_median": float(scores.median()) if not scores.empty else None,
            "score_min": float(scores.min()) if not scores.empty else None,
            "score_max": float(scores.max()) if not scores.empty else None,
            "histogram_file": str(model_out / "histogram_scores.png"),
            "score_curve_file": str(model_out / "score_curve.png"),
            "confusion_network_file": str(model_out / "confusion_network.png"),
            "accuracy_curve_file": str(model_out / "accuracy_curve.png") if has_gt_metrics else None,
            "loss_curve_file": str(model_out / "loss_curve.png") if has_gt_metrics else None,
            "scatter_pred_vs_actual_file": str(model_out / "scatter_predicted_vs_actual.png") if has_gt_metrics else None,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
        }
        if cm_counts is not None:
            model_summary["confusion_counts"] = cm_counts

        (model_out / "summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
        global_summary["models"][model_name_str] = model_summary

    (output_dir / "summary_all_models.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
    return global_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build individual per-model analysis graphs (matplotlib + networkx)"
    )
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True, help="One or more test result CSV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for graphs")
    parser.add_argument("--model-column", type=str, default=None, help="Column that identifies each model")
    parser.add_argument("--score-column", type=str, default=None, help="Column with model scores")
    parser.add_argument("--pred-column", type=str, default=None, help="Column with predicted labels (0/1)")
    parser.add_argument("--gt-column", type=str, default=None, help="Column with ground-truth labels (0/1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to derive predicted label if missing")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_model_analysis(
        input_csvs=args.input_csv,
        output_dir=args.output_dir,
        model_col=args.model_column,
        score_col=args.score_column,
        pred_col=args.pred_column,
        gt_col=args.gt_column,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
