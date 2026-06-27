from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from lunar_icenav.features.texture import robust_normalize


def save_feature_panel(base: np.ndarray, features: dict[str, np.ndarray], mask: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.8))
    panels = [
        ("SRI intensity", base, "gray"),
        ("CPR-style LH/LV ratio proxy", features["cpr_style_ratio_proxy"], "magma"),
        ("Polarization imbalance proxy", features["polarization_imbalance_proxy"], "viridis"),
        ("Texture roughness proxy", features["texture"], "inferno"),
        ("Candidate score", features["candidate_score"], "plasma"),
        ("Candidate mask", mask.astype(float), "Blues"),
    ]
    for ax, (name, arr, cmap) in zip(axes.ravel(), panels):
        display = robust_normalize(arr, np.isfinite(arr)) if name != "Candidate mask" else arr
        ax.imshow(display, cmap=cmap)
        ax.set_title(name, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.01, 0.02, "Screening outputs only; candidate regions require independent validation.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.08, wspace=0.02, hspace=0.20)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_overlay(
    base: np.ndarray,
    path: Path,
    title: str,
    masks: list[tuple[np.ndarray, str, tuple[float, float, float, float]]],
    routes: dict[str, list[tuple[int, int]]] | None = None,
    points: pd.DataFrame | None = None,
    note: str = "Preliminary screening output; validation required. Not a compositional or certified operations product.",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aspect = base.shape[1] / max(base.shape[0], 1)
    if aspect < 0.75:
        fig_w = 5.4
        fig_h = 7.2
    else:
        fig_w = 12
        fig_h = min(7.2, max(4.6, fig_w / max(aspect, 1.0) + 1.1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    legend_items: list[Any] = []
    for mask, label, color in masks:
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = np.where(mask, color[3], 0)
        ax.imshow(rgba)
        legend_items.append(Line2D([0], [0], color=color[:3], lw=6, alpha=color[3], label=label))
    if routes:
        route_colors = {"shortest": "#ffcc00", "safest": "#ff2bbd", "science_priority": "#00ffff"}
        for name, route in routes.items():
            if not route:
                continue
            rr = [p[0] for p in route]
            cc = [p[1] for p in route]
            ax.plot(cc, rr, color=route_colors.get(name, "#ffff00"), linewidth=2.8, label=name)
            legend_items.append(Line2D([0], [0], color=route_colors.get(name, "#ffff00"), lw=4, label=f"{name} route"))
    if points is not None and not points.empty:
        ax.scatter(points["col"], points["row"], s=82, c="#00ff66", edgecolor="black", linewidth=0.8, marker="o", label="candidate landing site")
        for _, row in points.head(3).iterrows():
            ax.text(row["col"] + 4, row["row"] + 4, row["site_id"], color="white", fontsize=10, weight="bold")
        legend_items.append(Line2D([0], [0], marker="o", color="black", markerfacecolor="#00ff66", lw=0, label="landing candidate"))
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", frameon=True, framealpha=0.90, fontsize=10)
    fig.text(0.01, 0.025, note, fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_slope_map(slope: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aspect = slope.shape[1] / max(slope.shape[0], 1)
    fig_w = 10
    fig_h = min(6.5, max(4.2, fig_w / max(aspect, 1.0) + 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(slope, cmap="terrain", vmin=0, vmax=np.nanpercentile(slope, 98))
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Slope / terrain layer value (deg)")
    fig.text(0.01, 0.02, "DEM-derived terrain context for planning prototype.", fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_histogram(df: pd.DataFrame, column: str, path: Path, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if not df.empty and column in df:
        ax.hist(df[column].dropna(), bins=min(20, max(5, len(df))), color="#2f80ed", alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if not df.empty and x in df and y in df:
        ax.bar(df[x].astype(str), df[y], color=["#ffcc00", "#ff2bbd", "#00bcd4"][: len(df)])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_workflow_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.set_axis_off()
    steps = [
        "Data audit",
        "Faustini AOI",
        "SAR proxies",
        "Candidate mask",
        "Patch evaluation",
        "DEM terrain",
        "Landing score",
        "Rover navigation",
        "Weak ML module",
        "Validation report",
    ]
    xs = np.linspace(0.04, 0.96, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.add_patch(plt.Rectangle((x - 0.045, 0.42), 0.09, 0.18, facecolor="#f2f6fb", edgecolor="#1f4e79", linewidth=1.5))
        ax.text(x, 0.51, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.05, 0.51), xytext=(x + 0.05, 0.51), arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.4))
    ax.text(0.5, 0.78, "LunaQuest Research and Mission Decision Pipeline", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.22, "Research outputs, GeoTIFF/PNG/CSV products, notebook outputs, and validation reports require independent validation.", ha="center", fontsize=10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_ohrc_hazard_overlay(gray: np.ndarray, hazard: np.ndarray, path: Path, source: str, note: str) -> None:
    save_overlay(
        gray,
        path,
        title="OHRC browse hazard proxy (context only)",
        masks=[(hazard, "rough/edge-rich proxy", (1.0, 0.28, 0.0, 0.45))],
        note="OHRC browse context only; no Faustini AOI co-registration demonstrated.",
    )


def save_inventory_chart(inventory: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = inventory["product_type"].fillna("unknown").value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = ["#78909c", "#00acc1", "#43a047", "#5c6bc0", "#f4511e"][: len(counts)]
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_title("Dataset Inventory by Product Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("File/product rows")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    fig.text(0.02, 0.03, "Source-data audit: SAR/DFSAR, OHRC bundles, DEM rasters, and project documents.", fontsize=8)
    fig.subplots_adjust(left=0.22, right=0.94, top=0.86, bottom=0.16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_coverage_map(sar_scores: pd.DataFrame, path: Path, selected_product_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = sar_scores.copy()
    df["label"] = df["product_id"].astype(str).str.replace("ch2_sar_", "", regex=False).str[:22]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = ["#2f80ed" if p == selected_product_id else "#90a4ae" for p in df["product_id"]]
    ax.barh(df["label"], df["coverage_fraction"].astype(float), color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AOI coverage fraction")
    ax.set_title("SAR Product Coverage of Faustini F2 Prototype AOI", fontsize=14, fontweight="bold")
    ax.axvline(1.0, color="#1b5e20", lw=1, ls="--")
    for i, row in df.iterrows():
        ax.text(float(row["coverage_fraction"]) + 0.02, i, f"{float(row['pixel_size_m']):.0f} m", va="center", fontsize=8)
    fig.text(0.01, 0.01, "Coverage is based on projected AOI intersection with calibrated SRI raster bounds.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_scalar_map(arr: np.ndarray, path: Path, title: str, cmap: str, colorbar_label: str, note: str, vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aspect = arr.shape[1] / max(arr.shape[0], 1)
    fig_w = 11
    fig_h = min(6.5, max(4.2, fig_w / max(aspect, 1.0) + 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    finite = np.isfinite(arr)
    display_vmin = np.nanpercentile(arr[finite], 2) if vmin is None and np.any(finite) else vmin
    display_vmax = np.nanpercentile(arr[finite], 98) if vmax is None and np.any(finite) else vmax
    im = ax.imshow(arr, cmap=cmap, vmin=display_vmin, vmax=display_vmax)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(colorbar_label)
    fig.text(0.01, 0.02, note, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.91, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_slope_classification(slope: np.ndarray, path: Path) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = np.full(slope.shape, -1, dtype=int)
    valid = np.isfinite(slope)
    classes[valid & (slope < 5)] = 0
    classes[valid & (slope >= 5) & (slope <= 10)] = 1
    classes[valid & (slope > 10)] = 2
    cmap = ListedColormap(["#2ecc71", "#ffcc00", "#e74c3c"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(np.ma.masked_where(classes < 0, classes), cmap=cmap, vmin=0, vmax=2)
    ax.set_title("DEM Slope Classification for Preliminary Landing Safety", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    legend = [
        Line2D([0], [0], color="#2ecc71", lw=8, label="safe < 5 deg"),
        Line2D([0], [0], color="#ffcc00", lw=8, label="moderate 5-10 deg"),
        Line2D([0], [0], color="#e74c3c", lw=8, label="unsafe > 10 deg"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9)
    total = max(int(valid.sum()), 1)
    stats = {
        "safe_slope_pct": float(((classes == 0).sum() / total) * 100),
        "moderate_slope_pct": float(((classes == 1).sum() / total) * 100),
        "unsafe_slope_pct": float(((classes == 2).sum() / total) * 100),
    }
    fig.text(0.01, 0.02, "Slope classes are DEM-derived screening classes, not final landing certification.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return stats


def save_top_candidates_chart(patches: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = patches.sort_values("mean_candidate_score", ascending=False).head(10) if not patches.empty else patches
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    if not top.empty:
        axes[0].bar(top["candidate_id"], top["mean_candidate_score"], color="#00acc1")
        axes[0].set_ylabel("Mean score")
        axes[0].set_ylim(0, 1)
        axes[1].bar(top["candidate_id"], top["area_m2"], color="#f4511e")
        axes[1].set_ylabel("Area (m2)")
    axes[0].set_title("Top 10 Radar-Based Candidate Patches", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Candidate ID")
    axes[1].tick_params(axis="x", rotation=30)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.text(0.02, 0.02, "Ranking is by screening score; patches require independent validation.", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18, hspace=0.12)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_score_distribution(features: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = features["valid"].astype(bool)
    score = features["candidate_score"][valid]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(score[np.isfinite(score)], bins=40, color="#5c6bc0", alpha=0.9)
    ax.set_title("Radar Candidate Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalized screening score")
    ax.set_ylabel("Pixels")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_centroid_map(base: np.ndarray, patches: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    if not patches.empty:
        sizes = np.clip(patches["area_pixels"].astype(float), 8, 120)
        ax.scatter(patches["centroid_col"], patches["centroid_row"], s=sizes, c=patches["mean_candidate_score"], cmap="cool", edgecolor="black", linewidth=0.3)
        for _, row in patches.head(8).iterrows():
            ax.text(row["centroid_col"] + 4, row["centroid_row"] + 4, row["candidate_id"], fontsize=7, color="white")
    ax.set_title("Candidate Patch Centroids on SAR Quicklook", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.02, "Centroids are derived from connected components in the preliminary candidate mask.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_landing_score_components(landing_sites: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    if not landing_sites.empty:
        ax.bar(landing_sites["site_id"], landing_sites["suitability_score"], color="#43a047")
    ax.set_ylim(0, 1)
    ax.set_title("Top Preliminary Landing Candidate Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("Landing candidate")
    ax.set_ylabel("Suitability score")
    fig.text(0.01, 0.01, "Scores combine low slope, terrain safety proxy, candidate proximity, and illumination-like proxy.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_route_comparison(route_summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    metrics = [("length_m", "Length (m)"), ("cost", "Cost"), ("max_slope_deg", "Max slope (deg)")]
    colors = ["#ffcc00", "#ff2bbd", "#00bcd4"]
    for ax, (col, label) in zip(axes, metrics):
        if not route_summary.empty and col in route_summary:
            ax.bar(route_summary["route_type"], route_summary[col].astype(float), color=colors[: len(route_summary)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Conceptual Rover Route Comparison", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Routes are conceptual variants on proxy cost maps; validation required.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_unet_training_curve(history: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if not history.empty:
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train loss", color="#2f80ed")
        axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="validation loss", color="#f4511e")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["val_pseudo_iou"], marker="o", label="pseudo-IoU", color="#00acc1")
        axes[1].plot(history["epoch"], history["val_pseudo_dice"], marker="s", label="pseudo-Dice", color="#43a047")
        axes[1].legend()
    axes[0].set_title("U-Net Pseudo-Label Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Pseudo-Label Agreement")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Agreement")
    fig.suptitle("Weakly Supervised U-Net Training Curves", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Metrics measure agreement with pseudo-labels, not true ice detection accuracy.", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.18, wspace=0.18)
    fig.savefig(path, dpi=300)
    plt.close(fig)
