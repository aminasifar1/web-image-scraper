#!/usr/bin/env python3
"""Generate a full figure pack from existing crawl results (no new crawl needed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_style import (
    CATEGORY_COLORS,
    COLOR_AI,
    COLOR_REAL,
    STANDARD_DPI,
    apply_plot_style,
)

apply_plot_style()


def _load_latest_stats_json(live_dir: Path) -> tuple[Path | None, dict]:
    stats_files = sorted(live_dir.glob("stats_*.json"))
    if not stats_files:
        return None, {}
    p = stats_files[-1]
    return p, json.loads(p.read_text(encoding="utf-8"))


def _save_fig(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all figures from existing crawl outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run root dir, e.g. crawl_runs/20260416_5x5_200117")
    parser.add_argument("--websites-csv", type=Path, required=True, help="websites_5_categories.csv path")
    args = parser.parse_args()

    run_dir = args.run_dir
    live_dir = run_dir / "live"
    out_dir = live_dir / "analysis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = max(live_dir.glob("images_metadata_*.csv"), default=None)
    if metadata_csv is None:
        raise FileNotFoundError(f"No images_metadata_*.csv found in {live_dir}")

    websites_df = pd.read_csv(args.websites_csv)
    meta = pd.read_csv(metadata_csv)
    stats_path, stats = _load_latest_stats_json(live_dir)

    artifacts: dict[str, str] = {}

    # 1) Global kept vs discarded (real stats)
    if stats:
        found = int(stats.get("images_found", 0))
        kept = int(stats.get("images_downloaded", 0))
        discarded = max(found - kept, 0)

        plt.figure(figsize=(7.2, 4.8))
        plt.bar(["All categories"], [kept], label="After cleaning (kept)", color=COLOR_REAL, edgecolor="black")
        plt.bar(["All categories"], [discarded], bottom=[kept], label="Discarded after cleaning", color=COLOR_AI, edgecolor="black")
        plt.title("Global cleaning result: kept vs discarded")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        if found:
            plt.text(0, kept + 12, f"{kept}/{found} ({kept/found*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")
        artifacts["global_kept_vs_discarded_stacked"] = _save_fig(out_dir / "global_kept_vs_discarded_stacked.png")

    # 2) Filtering breakdown
    if stats:
        keys = [
            ("images_filtered_ads", "Ads"),
            ("images_filtered_ui", "UI/Icon"),
            ("images_filtered_quality", "Quality"),
            ("images_filtered_sticker", "Sticker"),
            ("images_filtered_video", "Video thumbnail"),
            ("images_filtered_near_duplicate", "Near duplicate"),
            ("images_filtered_missing_alt", "Missing alt/caption"),
        ]
        labels = []
        vals = []
        for k, label in keys:
            v = int(stats.get(k, 0))
            if v > 0:
                labels.append(label)
                vals.append(v)

        if vals:
            plt.figure(figsize=(10.5, 5.2))
            bars = plt.bar(labels, vals, color="#f08c00", edgecolor="black")
            plt.title("Filtering breakdown (counts)")
            plt.ylabel("Count")
            plt.xticks(rotation=22, ha="right")
            plt.grid(axis="y", alpha=0.25)
            for b, v in zip(bars, vals):
                plt.text(b.get_x() + b.get_width()/2, v + 8, f"{v}", ha="center", fontsize=9)
            artifacts["filtering_breakdown"] = _save_fig(out_dir / "filtering_breakdown_counts.png")

    # 3) Retained images by category
    cat_counts = meta["sector"].value_counts().rename_axis("sector").reset_index(name="count")
    cat_counts = cat_counts.sort_values("count", ascending=False)

    plt.figure(figsize=(9.0, 5.0))
    bars = plt.bar(
        cat_counts["sector"],
        cat_counts["count"],
        color=[CATEGORY_COLORS.get(s, "#999999") for s in cat_counts["sector"]],
        edgecolor="black",
    )
    plt.title("Retained images by category")
    plt.ylabel("Images")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, cat_counts["count"]):
        plt.text(b.get_x() + b.get_width()/2, v + 2, f"{int(v)}", ha="center", fontsize=9)
    artifacts["retained_by_category"] = _save_fig(out_dir / "retained_by_category.png")

    # 4) Stacked by category: kept vs discarded from category total target
    target_by_cat = websites_df.groupby("sector").size().rename("sites").reset_index()
    target_by_cat["category_total"] = target_by_cat["sites"] * 50
    merged = target_by_cat.merge(cat_counts, on="sector", how="left").fillna({"count": 0})
    merged = merged.rename(columns={"count": "kept"})
    merged["kept"] = merged["kept"].astype(int)
    merged["discarded"] = (merged["category_total"] - merged["kept"]).clip(lower=0)
    merged = merged.sort_values("kept", ascending=False)

    plt.figure(figsize=(11, 6))
    x = range(len(merged))
    plt.bar(x, merged["kept"], label="After cleaning (kept)", color=COLOR_REAL, edgecolor="black")
    plt.bar(x, merged["discarded"], bottom=merged["kept"], label="Discarded after cleaning", color=COLOR_AI, edgecolor="black")
    plt.xticks(x, [s.replace("_", "\n") for s in merged["sector"]])
    plt.title("By category (stacked): kept vs discarded")
    plt.ylabel("Images")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for i, r in merged.reset_index(drop=True).iterrows():
        plt.text(i, r["kept"] + 3, f"{int(r['kept'])}/{int(r['category_total'])}", ha="center", fontsize=9)
    artifacts["category_stacked_kept_discarded"] = _save_fig(out_dir / "category_collected_vs_remaining_stacked.png")

    # 5) News-only stacked bar
    news = merged[merged["sector"] == "news"]
    if not news.empty:
        n = news.iloc[0]
        plt.figure(figsize=(6.2, 4.8))
        bars = plt.bar(["Kept (news)", "Discarded (news)"], [int(n["kept"]), int(n["discarded"])], color=[COLOR_REAL, COLOR_AI], edgecolor="black")
        plt.title("News category: kept vs discarded")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.25)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h + 1.5, f"{int(h)}", ha="center", fontsize=10)
        artifacts["news_kept_vs_discarded"] = _save_fig(out_dir / "news_collected_vs_remaining.png")

    # 6) Website-level retained counts (top 15)
    by_org = (
        meta.groupby(["sector", "organization_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    top = by_org.head(15).copy()
    labels = [f"{r.organization_name}\n({r.sector})" for r in top.itertuples(index=False)]
    plt.figure(figsize=(13, 6))
    bars = plt.bar(labels, top["count"], color=[CATEGORY_COLORS.get(s, "#999999") for s in top["sector"]], edgecolor="black")
    plt.title("Top retained images by website")
    plt.ylabel("Images")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, top["count"]):
        plt.text(b.get_x() + b.get_width()/2, v + 1, f"{int(v)}", ha="center", fontsize=8)
    artifacts["retained_top_websites"] = _save_fig(out_dir / "retained_top_websites.png")

    # 7) Attempted vs successful websites by category
    attempted = websites_df.groupby("sector", as_index=False).size().rename(columns={"size": "attempted_sites"})
    successful = (
        meta[["sector", "organization_name"]]
        .drop_duplicates()
        .groupby("sector", as_index=False)
        .size()
        .rename(columns={"size": "successful_sites"})
    )
    site_cmp = attempted.merge(successful, on="sector", how="left").fillna({"successful_sites": 0})
    site_cmp["successful_sites"] = site_cmp["successful_sites"].astype(int)
    site_cmp["failed_sites"] = site_cmp["attempted_sites"] - site_cmp["successful_sites"]

    plt.figure(figsize=(10, 5.5))
    x = range(len(site_cmp))
    plt.bar(x, site_cmp["successful_sites"], label="Successful sites", color=COLOR_REAL, edgecolor="black")
    plt.bar(x, site_cmp["failed_sites"], bottom=site_cmp["successful_sites"], label="No images / failed", color=COLOR_AI, edgecolor="black")
    plt.xticks(x, [s.replace("_", "\n") for s in site_cmp["sector"]])
    plt.ylabel("Websites")
    plt.title("Attempted vs successful websites by category")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for i, r in site_cmp.reset_index(drop=True).iterrows():
        plt.text(i, r["successful_sites"] + 0.05, f"{int(r['successful_sites'])}/{int(r['attempted_sites'])}", ha="center", fontsize=9)
    artifacts["attempted_vs_successful_sites"] = _save_fig(out_dir / "attempted_vs_successful_sites_by_category.png")

    # tabular outputs
    cat_table = merged[["sector", "category_total", "kept", "discarded"]].copy()
    cat_table["kept_pct"] = (cat_table["kept"] / cat_table["category_total"] * 100).round(2)
    cat_table.to_csv(out_dir / "category_kept_discarded_table.csv", index=False)
    by_org.to_csv(out_dir / "retained_by_website_table.csv", index=False)
    site_cmp.to_csv(out_dir / "attempted_vs_successful_sites_table.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "metadata_csv": str(metadata_csv),
        "stats_json": str(stats_path) if stats_path else None,
        "artifacts": artifacts,
    }
    (out_dir / "all_existing_results_figures_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
