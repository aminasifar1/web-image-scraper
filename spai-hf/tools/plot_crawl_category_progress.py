#!/usr/bin/env python3
"""Create crawl progress charts by category (before/after cleaning style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_style import COLOR_AI, COLOR_REAL, STANDARD_DPI, apply_plot_style

apply_plot_style()


def load_latest_stats_json(live_dir: Path) -> Path | None:
    candidates = sorted(live_dir.glob("stats_*.json"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stacked bars with kept vs discarded after cleaning, by category."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Path to images_metadata_*.csv")
    parser.add_argument("--websites-csv", type=Path, required=True, help="Path to websites_5_categories.csv")
    parser.add_argument("--live-dir", type=Path, required=True, help="Live crawl directory (contains stats_*.json)")
    parser.add_argument("--max-images-per-site", type=int, default=50, help="Configured max images per site")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write plots and csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    websites = pd.read_csv(args.websites_csv)

    collected = (
        meta.groupby("sector", as_index=False)
        .size()
        .rename(columns={"size": "collected"})
    )

    targets = (
        websites.groupby("sector", as_index=False)
        .size()
        .rename(columns={"size": "sites"})
    )
    targets["category_total"] = targets["sites"] * int(args.max_images_per_site)

    summary = targets.merge(collected, on="sector", how="left").fillna({"collected": 0})
    summary["collected"] = summary["collected"].astype(int)
    summary["remaining_from_total"] = (summary["category_total"] - summary["collected"]).clip(lower=0)
    summary["collected_pct"] = (summary["collected"] / summary["category_total"] * 100).round(2)

    # Include global retained/discarded from stats JSON if available
    stats_path = load_latest_stats_json(args.live_dir)
    global_stats = {}
    if stats_path is not None:
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        found = int(s.get("images_found", 0))
        downloaded = int(s.get("images_downloaded", 0))
        discarded = max(found - downloaded, 0)
        global_stats = {
            "stats_file": str(stats_path),
            "images_found": found,
            "images_downloaded": downloaded,
            "images_discarded": discarded,
            "retained_pct": round((downloaded / found * 100), 2) if found else 0.0,
            "discarded_pct": round((discarded / found * 100), 2) if found else 0.0,
        }

    # Save CSV summary
    summary_csv = args.output_dir / "category_collection_progress.csv"
    summary.sort_values("sector").to_csv(summary_csv, index=False)

    # Plot 1: stacked bars kept vs discarded (from category total baseline)
    plot_df = summary.sort_values("collected", ascending=False).reset_index(drop=True)
    x = range(len(plot_df))

    plt.figure(figsize=(11, 6))
    plt.bar(x, plot_df["collected"], label="After cleaning (kept)", color=COLOR_REAL, edgecolor="black", linewidth=0.8)
    plt.bar(
        x,
        plot_df["remaining_from_total"],
        bottom=plot_df["collected"],
        label="Discarded after cleaning",
        color=COLOR_AI,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )
    plt.xticks(x, [s.replace("_", "\n") for s in plot_df["sector"]])
    plt.ylabel("Images")
    plt.title("By category (stacked): kept vs discarded after cleaning")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    for i, row in plot_df.iterrows():
        plt.text(i, row["collected"] + 4, f"{int(row['collected'])}/{int(row['category_total'])}", ha="center", fontsize=9)

    plt.tight_layout()
    stacked_png = args.output_dir / "category_collected_vs_remaining_stacked.png"
    plt.savefig(stacked_png, dpi=STANDARD_DPI)
    plt.close()

    # Plot 2: focused chart for news
    news_row = summary[summary["sector"] == "news"]
    if not news_row.empty:
        n = news_row.iloc[0]
        plt.figure(figsize=(6.2, 4.8))
        bars = plt.bar(
            ["Kept (news)", "Discarded (news)"],
            [int(n["collected"]), int(n["remaining_from_total"])],
            color=[COLOR_REAL, COLOR_AI],
            edgecolor="black",
        )
        plt.ylabel("Images")
        plt.title("News category: kept vs discarded after cleaning")
        plt.grid(axis="y", alpha=0.25)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{int(h)}", ha="center", fontsize=10)
        plt.tight_layout()
        news_png = args.output_dir / "news_collected_vs_remaining.png"
        plt.savefig(news_png, dpi=STANDARD_DPI)
        plt.close()
    else:
        news_png = None

    # Plot 3: global real cleaning bar using actual crawl stats
    global_png = None
    if global_stats:
        kept = int(global_stats["images_downloaded"])
        discarded = int(global_stats["images_discarded"])
        total_found = int(global_stats["images_found"])

        plt.figure(figsize=(7.2, 4.8))
        plt.bar(["All categories"], [kept], label="After cleaning (kept)", color=COLOR_REAL, edgecolor="black")
        plt.bar(["All categories"], [discarded], bottom=[kept], label="Discarded after cleaning", color=COLOR_AI, edgecolor="black")
        plt.ylabel("Images")
        plt.title("Global cleaning result: kept vs discarded")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.text(0, kept + 20, f"{kept}/{total_found}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        global_png = args.output_dir / "global_kept_vs_discarded_stacked.png"
        plt.savefig(global_png, dpi=STANDARD_DPI)
        plt.close()

    # Save JSON summary
    summary_json = args.output_dir / "category_collection_progress_summary.json"
    payload = {
        "metadata_csv": str(args.metadata_csv),
        "websites_csv": str(args.websites_csv),
        "max_images_per_site": int(args.max_images_per_site),
        "by_category": summary.sort_values("sector").to_dict(orient="records"),
        "global_stats": global_stats,
        "artifacts": {
            "summary_csv": str(summary_csv),
            "stacked_plot": str(stacked_png),
            "news_plot": str(news_png) if news_png else None,
            "global_plot": str(global_png) if global_png else None,
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
