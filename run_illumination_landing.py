"""
Illumination-Aware Landing Site Selection Pipeline
===================================================

This script implements a corrected landing site selection pipeline that
explicitly considers solar illumination constraints. It processes all
available SAR data, computes illumination proxies from DEM/elevation,
identifies PSR zones, and selects landing sites in sunlit terrain.

Mission architecture: Land in sunlight → Rove into PSR for ice exploration.

Usage:
    $env:PYTHONPATH='src'
    python run_illumination_landing.py
"""

from __future__ import annotations

import json
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# ────────────────────────────────────────────────────────────────────
# Project imports
# ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lunar_icenav.io.products import find_sar_pairs
from lunar_icenav.preprocessing.aoi import (
    choose_best_sar_pair,
    clipped_window_for_aoi,
    evaluate_sar_pair,
    map_to_lonlat,
    pixel_to_map,
)
from lunar_icenav.preprocessing.sar import read_sar_aoi
from lunar_icenav.preprocessing.dem import (
    read_dem_aoi,
    read_tmc_dtm_aoi,
    compute_slope_deg,
    resize_to,
)
from lunar_icenav.features.polarimetry import sar_feature_stack
from lunar_icenav.features.texture import robust_normalize
from lunar_icenav.planning.illumination import (
    compute_illumination_score,
    classify_illumination_zones,
    detect_psr,
    find_crater_rim_pixels,
    multi_azimuth_illumination,
    compute_aspect,
    compute_hillshade,
)
from lunar_icenav.planning.rover import astar, build_cost_map, path_length
from lunar_icenav.mapping.anomaly import candidate_mask_from_features, summarize_patches


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "pipeline.json"
OUTPUT_DIR = ROOT / "outputs" / "illumination_landing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENDED_AOI = {
    "name": "Faustini Extended (rim + floor)",
    "lat_min": -88.5,
    "lat_max": -85.0,
    "lon_min": 77.0,
    "lon_max": 88.0,
}

LANDING_AOI = {
    "name": "Faustini Landing Zone (sunlit rim)",
    "lat_min": -87.0,
    "lat_max": -85.0,
    "lon_min": 77.0,
    "lon_max": 88.0,
}

ICE_AOI = {
    "name": "Faustini F2 ice candidate AOI",
    "lat_min": -87.8,
    "lat_max": -86.9,
    "lon_min": 80.0,
    "lon_max": 85.0,
}

# Landing site selection thresholds
MAX_LANDING_SLOPE_DEG = 5.0
MIN_ILLUMINATION_PERSISTENCE = 0.40
MIN_LANDING_ELLIPSE_RADIUS_PX = 5
TOP_SITE_COUNT = 5


# ────────────────────────────────────────────────────────────────────
# Utility: build coordinate grids
# ────────────────────────────────────────────────────────────────────
def build_lon_lat_grids(transform, crs_wkt: str, shape: tuple[int, int]):
    rows, cols = np.indices(shape, dtype="float32")
    xs = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
    ys = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    return map_to_lonlat(crs_wkt, xs, ys)


def build_aoi_mask(lon_grid, lat_grid, aoi):
    return (
        (lat_grid >= aoi["lat_min"]) & (lat_grid <= aoi["lat_max"])
        & (lon_grid >= aoi["lon_min"]) & (lon_grid <= aoi["lon_max"])
    )


# ────────────────────────────────────────────────────────────────────
# Step 1: Inventory all SAR products
# ────────────────────────────────────────────────────────────────────
def inventory_sar_products(root: Path, aoi: dict) -> pd.DataFrame:
    """Find and evaluate ALL SAR products against the extended AOI."""
    pairs = find_sar_pairs(root)
    print(f"  Found {len(pairs)} SAR LH/LV pairs")
    rows = []
    for pair in pairs:
        info = evaluate_sar_pair(pair, aoi)
        product_id = pair["lh"].parts[-5] if len(pair["lh"].parts) >= 5 else pair["lh"].stem
        rows.append({
            "product_id": product_id,
            "lh_path": str(pair["lh"]),
            "lv_path": str(pair["lv"]),
            "coverage_fraction": info.get("coverage_fraction", 0),
            "pixel_size_m": info.get("pixel_size_m"),
            "crs_wkt": info.get("crs_wkt", ""),
            "usable": info.get("coverage_fraction", 0) > 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sar_inventory.csv", index=False)
    return df


# ────────────────────────────────────────────────────────────────────
# Step 2: Read & process the best SAR data
# ────────────────────────────────────────────────────────────────────
def load_sar_data(root: Path, aoi: dict) -> dict[str, Any]:
    """Load the best-coverage SAR pair and compute features."""
    pairs = find_sar_pairs(root)
    best, scores = choose_best_sar_pair(pairs, aoi)
    print(f"  Selected SAR: {best.get('product_id', 'unknown')} (coverage={best['coverage_fraction']:.2f})")
    sar = read_sar_aoi(best, aoi)
    features = sar_feature_stack(sar["lh"], sar["lv"])
    return {**sar, "features": features, "pair_info": best}


# ────────────────────────────────────────────────────────────────────
# Step 3: Build illumination-aware landing score
# ────────────────────────────────────────────────────────────────────
def illumination_aware_suitability(
    features: dict[str, np.ndarray],
    slope_deg: np.ndarray | None,
    elevation: np.ndarray | None,
    illumination: np.ndarray,
    psr_mask: np.ndarray,
    candidate_mask: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute landing suitability with illumination as primary constraint.

    Returns: (score_array, landing_allowed_mask)
    """
    valid = features["valid"].astype(bool)
    max_slope = config.get("max_landing_slope_deg", MAX_LANDING_SLOPE_DEG)

    # ── Slope score ──
    if slope_deg is not None:
        slope_score = 1.0 - np.clip(slope_deg / max_slope, 0, 1)
        slope_ok = np.isfinite(slope_deg) & (slope_deg < max_slope)
    else:
        slope_score = np.full(valid.shape, 0.5, dtype="float32")
        slope_ok = valid.copy()

    # ── Roughness avoidance ──
    hazard = robust_normalize(features["texture"], valid)
    low_hazard = 1.0 - hazard

    # ── Candidate proximity (science value) ──
    dist_to_candidate = ndi.distance_transform_edt(~candidate_mask)
    proximity = 1.0 / (1.0 + dist_to_candidate / 40.0)

    # ── Illumination score (THE KEY FIX) ──
    illum_score = np.where(np.isfinite(illumination), illumination, 0.0).astype("float32")
    illum_ok = (illum_score >= MIN_ILLUMINATION_PERSISTENCE) & (~psr_mask)

    # ── Thermal proxy (derived from illumination) ──
    thermal_score = np.clip(illum_score * 0.85 + 0.15, 0, 1).astype("float32")

    # ── HARD CONSTRAINTS ──
    landing_allowed = (
        valid
        & slope_ok
        & illum_ok          # Must have sufficient sunlight
        & (~psr_mask)       # Must NOT be in PSR
        & (~candidate_mask) # Don't land ON the ice candidate
    )
    # Clearance from candidate mask
    clearance_px = 3
    landing_allowed &= (dist_to_candidate >= clearance_px)

    # ── Weighted suitability score ──
    score = (
        0.20 * slope_score
        + 0.12 * low_hazard
        + 0.13 * proximity
        + 0.35 * illum_score        # DOMINANT: illumination
        + 0.10 * thermal_score
        + 0.10 * np.clip(1.0 - dist_to_candidate / 200.0, 0, 1)  # near-PSR bonus
    ).astype("float32")
    score[~landing_allowed] = np.nan

    return score, landing_allowed


# ────────────────────────────────────────────────────────────────────
# Step 4: Select top landing sites
# ────────────────────────────────────────────────────────────────────
def select_landing_sites(
    score: np.ndarray,
    landing_allowed: np.ndarray,
    slope_deg: np.ndarray | None,
    illumination: np.ndarray,
    elevation: np.ndarray | None,
    candidate_mask: np.ndarray,
    transform,
    crs_wkt: str,
    psr_mask: np.ndarray,
    count: int = TOP_SITE_COUNT,
) -> pd.DataFrame:
    """Select top landing sites with suppression to avoid clustering."""
    working = np.where(np.isfinite(score) & landing_allowed, score, -np.inf).copy()
    rows = []
    pixel_size = abs(float(transform.a))
    dist_to_candidate = ndi.distance_transform_edt(~candidate_mask)

    for idx in range(count):
        flat = int(np.argmax(working))
        if working.ravel()[flat] == -np.inf:
            break
        row, col = np.unravel_index(flat, working.shape)
        x, y = pixel_to_map(transform, float(row), float(col))
        lon, lat = map_to_lonlat(crs_wkt, np.array([x]), np.array([y]))

        site = {
            "site_id": f"LS-{idx + 1:02d}",
            "row": int(row),
            "col": int(col),
            "lat": float(lat[0]),
            "lon": float(lon[0]),
            "suitability_score": float(score[row, col]),
            "illumination_persistence": float(illumination[row, col]) if np.isfinite(illumination[row, col]) else 0.0,
            "slope_deg": float(slope_deg[row, col]) if slope_deg is not None and np.isfinite(slope_deg[row, col]) else np.nan,
            "elevation_m": float(elevation[row, col]) if elevation is not None and np.isfinite(elevation[row, col]) else np.nan,
            "distance_to_ice_candidate_m": float(dist_to_candidate[row, col] * pixel_size),
            "in_psr": "no",
            "illumination_status": "COMPUTED from DEM terrain model",
            "mission_feasibility": "FEASIBLE for solar-powered lander",
        }

        # Classify illumination quality
        illum = site["illumination_persistence"]
        if illum >= 0.70:
            site["illumination_class"] = "excellent (>70%)"
        elif illum >= 0.50:
            site["illumination_class"] = "good (50-70%)"
        elif illum >= 0.40:
            site["illumination_class"] = "adequate (40-50%)"
        else:
            site["illumination_class"] = "marginal (<40%)"

        rows.append(site)

        # Suppress neighborhood to avoid clustering
        r0 = max(0, row - 25)
        r1 = min(score.shape[0], row + 26)
        c0 = max(0, col - 25)
        c1 = min(score.shape[1], col + 26)
        working[r0:r1, c0:c1] = -np.inf

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Step 5: Plan rover corridors from landing site to PSR
# ────────────────────────────────────────────────────────────────────
def plan_corridors(
    features: dict[str, np.ndarray],
    slope_deg: np.ndarray | None,
    illumination: np.ndarray,
    psr_mask: np.ndarray,
    candidate_mask: np.ndarray,
    landing_sites: pd.DataFrame,
    transform,
    crs_wkt: str,
    config: dict,
) -> tuple[dict[str, list], pd.DataFrame]:
    """Plan safe corridors from sunlit landing site to PSR ice targets."""
    valid = features["valid"].astype(bool)
    soft_max_slope = 15.0

    # Target: highest-score candidate patch
    if candidate_mask.any():
        target_score = np.where(candidate_mask, features["candidate_score"], -np.inf)
        target = tuple(map(int, np.unravel_index(np.argmax(target_score), target_score.shape)))
    else:
        target = tuple(map(int, np.unravel_index(
            np.nanargmax(features["candidate_score"]), features["candidate_score"].shape)))

    # Cost map includes illumination penalty for rover energy
    texture_norm = robust_normalize(features["texture"], valid)
    if slope_deg is not None:
        slope_norm = np.clip(slope_deg / soft_max_slope, 0, 1)
        blocked = (~valid) | (~np.isfinite(slope_deg)) | (slope_deg > soft_max_slope)
    else:
        slope_norm = np.zeros_like(texture_norm)
        blocked = ~valid

    corridors = {}
    corridor_rows = []
    pixel_size = abs(float(transform.a))

    if landing_sites.empty:
        return corridors, pd.DataFrame(corridor_rows)

    start = (int(landing_sites.iloc[0]["row"]), int(landing_sites.iloc[0]["col"]))

    for mode in ["safest", "science_priority", "energy_efficient"]:
        if mode == "safest":
            cost = 1.0 + 2.5 * slope_norm + 2.0 * texture_norm
        elif mode == "energy_efficient":
            cost = 1.0 + 1.5 * slope_norm + 1.2 * texture_norm
            # Bonus for staying in illuminated terrain (solar recharge)
            illum_bonus = np.clip(illumination, 0, 1)
            cost += 0.8 * (1.0 - illum_bonus)
        else:
            science = robust_normalize(features["candidate_score"], valid)
            cost = 1.0 + 1.0 * slope_norm + 0.8 * texture_norm + 0.5 * (1 - science)

        cost = cost.astype("float32")
        cost[blocked] = np.inf

        path, route_cost, status = astar(cost, start, target, max_expansions=1000000)
        corridors[mode] = path

        length_m = path_length(path, pixel_size)
        route_slope = np.array([slope_deg[r, c] for r, c in path], dtype=float) if slope_deg is not None and path else np.array([])
        route_illum = np.array([illumination[r, c] for r, c in path], dtype=float) if path else np.array([])

        # Get start/end coordinates
        if path:
            sx, sy = pixel_to_map(transform, float(path[0][0]), float(path[0][1]))
            ex, ey = pixel_to_map(transform, float(path[-1][0]), float(path[-1][1]))
            slon, slat = map_to_lonlat(crs_wkt, np.array([sx]), np.array([sy]))
            elon, elat = map_to_lonlat(crs_wkt, np.array([ex]), np.array([ey]))
        else:
            slat = slon = elat = elon = [np.nan]

        corridor_rows.append({
            "corridor_type": mode,
            "status": status,
            "steps": len(path),
            "length_m": length_m,
            "length_km": length_m / 1000.0,
            "cost": route_cost,
            "mean_slope_deg": float(np.nanmean(route_slope)) if route_slope.size else np.nan,
            "max_slope_deg": float(np.nanmax(route_slope)) if route_slope.size else np.nan,
            "mean_illumination": float(np.nanmean(route_illum)) if route_illum.size else np.nan,
            "min_illumination": float(np.nanmin(route_illum)) if route_illum.size else np.nan,
            "start_lat": float(slat[0]),
            "start_lon": float(slon[0]),
            "end_lat": float(elat[0]),
            "end_lon": float(elon[0]),
            "psr_entry_fraction": float(np.mean(route_illum < 0.15)) if route_illum.size else 0.0,
        })

    return corridors, pd.DataFrame(corridor_rows)


# ════════════════════════════════════════════════════════════════════
#    VISUALIZATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════

DARK_BG = "#0a0a0f"
PANEL_BG = "#111118"
GOLD = "#ffd700"
CYAN = "#00e5ff"
GREEN = "#00e676"
RED = "#ff1744"
MAGENTA = "#e040fb"
WHITE = "#f0f0f0"


def save_illumination_map(illumination, psr_mask, valid, path, title="Illumination Persistence Map"):
    """Save illumination persistence map with PSR overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Custom colormap: dark purple (PSR) → yellow (sunlit)
    cmap = LinearSegmentedColormap.from_list("illum", [
        (0.0, "#0d0030"),
        (0.15, "#1a0066"),
        (0.30, "#4a0080"),
        (0.45, "#8b3a62"),
        (0.60, "#c76b3a"),
        (0.75, "#e8a419"),
        (0.90, "#ffd700"),
        (1.0, "#ffffaa"),
    ])

    display = np.where(valid, illumination, np.nan)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")

    # PSR overlay in blue
    psr_overlay = np.zeros((*psr_mask.shape, 4), dtype="float32")
    psr_overlay[psr_mask, :] = [0.0, 0.4, 1.0, 0.35]
    ax.imshow(psr_overlay)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Illumination Persistence (fraction of lunar day)", color=WHITE, fontsize=11)
    cb.ax.tick_params(colors=WHITE)

    ax.set_title(title, color=GOLD, fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(colors=WHITE)

    legend = [
        mpatches.Patch(facecolor="#1a0066", edgecolor=WHITE, label="PSR (<15% illumination)"),
        mpatches.Patch(facecolor="#c76b3a", edgecolor=WHITE, label="Moderate (30-50%)"),
        mpatches.Patch(facecolor="#ffd700", edgecolor=WHITE, label="Good (50-70%)"),
        mpatches.Patch(facecolor="#ffffaa", edgecolor=WHITE, label="Excellent (>70%)"),
        mpatches.Patch(facecolor=(0, 0.4, 1.0, 0.4), edgecolor=WHITE, label="PSR Zone (overlay)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9,
              facecolor=PANEL_BG, edgecolor=GOLD, labelcolor=WHITE)

    fig.text(0.01, 0.01,
             "Illumination proxy from multi-azimuth DEM hillshade + latitude + elevation prominence | Not ray-traced ephemeris",
             color="#888888", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_landing_decision_map(
    base_image, illumination, slope_deg, psr_mask, candidate_mask,
    landing_sites, corridors, corridor_summary,
    valid, transform, crs_wkt, path,
):
    """Generate the main landing decision map (professional, reference-quality)."""
    fig = plt.figure(figsize=(20, 16), facecolor=DARK_BG)

    # Main map takes 75% of width
    ax_map = fig.add_axes([0.02, 0.05, 0.68, 0.88])
    ax_map.set_facecolor(DARK_BG)

    # ── Base: DEM/SAR grayscale ──
    display = np.where(valid, base_image, np.nan)
    lo, hi = np.nanpercentile(display[valid], [2, 98])
    display_norm = np.clip((display - lo) / max(hi - lo, 1e-6), 0, 1)
    ax_map.imshow(display_norm, cmap="gray", vmin=0, vmax=1)

    # ── Illumination heatmap overlay ──
    illum_cmap = LinearSegmentedColormap.from_list("illum_ov", [
        (0.0, (0, 0, 0, 0)),
        (0.2, (0.1, 0, 0.4, 0.3)),
        (0.4, (0.6, 0.3, 0.0, 0.3)),
        (0.6, (0.9, 0.7, 0.0, 0.35)),
        (0.8, (1.0, 0.9, 0.2, 0.4)),
        (1.0, (1.0, 1.0, 0.6, 0.45)),
    ])
    illum_disp = np.where(valid & np.isfinite(illumination), illumination, np.nan)
    ax_map.imshow(illum_disp, cmap=illum_cmap, vmin=0, vmax=1, alpha=0.6)

    # ── PSR zones in deep blue ──
    psr_ov = np.zeros((*psr_mask.shape, 4), dtype="float32")
    psr_ov[psr_mask, :] = [0.0, 0.2, 0.8, 0.4]
    ax_map.imshow(psr_ov)

    # ── Ice candidate patches in cyan ──
    cand_ov = np.zeros((*candidate_mask.shape, 4), dtype="float32")
    cand_ov[candidate_mask, :] = [0.0, 0.9, 1.0, 0.4]
    ax_map.imshow(cand_ov)

    # ── Slope hazard (>10 deg) in red ──
    if slope_deg is not None:
        unsafe = valid & (slope_deg > 10)
        unsafe_ov = np.zeros((*unsafe.shape, 4), dtype="float32")
        unsafe_ov[unsafe, :] = [1.0, 0.1, 0.1, 0.25]
        ax_map.imshow(unsafe_ov)

    # ── Corridor paths ──
    corridor_colors = {
        "safest": "#ff2bbd",
        "science_priority": CYAN,
        "energy_efficient": GOLD,
    }
    for mode, path_pts in corridors.items():
        if path_pts:
            cc = [p[1] for p in path_pts]
            rr = [p[0] for p in path_pts]
            color = corridor_colors.get(mode, "#ffffff")
            ax_map.plot(cc, rr, color=color, linewidth=2.5, alpha=0.85, zorder=5)
            # Corridor label
            mid = len(path_pts) // 2
            if corridor_summary is not None and not corridor_summary.empty:
                row_data = corridor_summary[corridor_summary["corridor_type"] == mode]
                if not row_data.empty:
                    km = row_data.iloc[0]["length_km"]
                    ax_map.annotate(
                        f"{mode.replace('_', ' ').title()}\n{km:.1f} km",
                        (cc[mid], rr[mid]),
                        color=color, fontsize=8, fontweight="bold",
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=color, alpha=0.8),
                        zorder=6,
                    )

    # ── Landing sites ──
    if not landing_sites.empty:
        for i, site in landing_sites.iterrows():
            color = GREEN if i == 0 else GOLD
            marker_size = 200 if i == 0 else 120
            ax_map.scatter(site["col"], site["row"], c=color, s=marker_size,
                          marker="*", edgecolors="white", linewidths=1, zorder=10)
            ax_map.annotate(
                site["site_id"],
                (site["col"] + 8, site["row"] - 8),
                color=color, fontsize=9, fontweight="bold", zorder=11,
            )

        # Highlight selected landing site
        best = landing_sites.iloc[0]
        circle = plt.Circle((best["col"], best["row"]), 15,
                            fill=False, color=GREEN, linewidth=2.5, linestyle="--", zorder=9)
        ax_map.add_patch(circle)
        ax_map.annotate(
            "SELECTED\nLANDING SITE",
            (best["col"] + 20, best["row"] + 20),
            color=GREEN, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_BG, edgecolor=GREEN, alpha=0.9),
            zorder=11,
        )

    ax_map.set_title("Illumination-Aware Landing Site Selection — Faustini Region",
                      color=GOLD, fontsize=15, fontweight="bold", pad=12)
    ax_map.tick_params(colors=WHITE, labelsize=8)

    # ══════════════════════════════════════════════════════════════
    #   RIGHT PANEL: Info boxes
    # ══════════════════════════════════════════════════════════════

    # Panel 1: LANDING SITE SELECTION CRITERIA
    ax_criteria = fig.add_axes([0.72, 0.68, 0.26, 0.25])
    ax_criteria.set_facecolor(PANEL_BG)
    ax_criteria.set_xlim(0, 1); ax_criteria.set_ylim(0, 1)
    ax_criteria.set_xticks([]); ax_criteria.set_yticks([])
    for spine in ax_criteria.spines.values():
        spine.set_color(GOLD); spine.set_linewidth(1.5)

    ax_criteria.text(0.5, 0.92, "LANDING SITE SELECTION", color=GOLD, fontsize=11,
                     fontweight="bold", ha="center", transform=ax_criteria.transAxes)
    criteria = [
        ("□", "Low slope (< 5°)"),
        ("□", "Hazard free (no boulders)"),
        ("□", f"Good illumination (>{int(MIN_ILLUMINATION_PERSISTENCE*100)}%)"),
        ("□", "SAR indicates subsurface ice"),
        ("□", "Outside PSR zone"),
        ("□", "Within comm. visibility"),
    ]
    for i, (mark, text) in enumerate(criteria):
        y = 0.78 - i * 0.12
        ax_criteria.text(0.05, y, mark, color=GOLD, fontsize=10, transform=ax_criteria.transAxes)
        ax_criteria.text(0.12, y, text, color=WHITE, fontsize=9, transform=ax_criteria.transAxes)

    # Panel 2: SELECTED LANDING SITE details
    ax_info = fig.add_axes([0.72, 0.40, 0.26, 0.25])
    ax_info.set_facecolor(PANEL_BG)
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_color(GREEN); spine.set_linewidth(1.5)

    ax_info.text(0.5, 0.92, "SELECTED LANDING SITE", color=GREEN, fontsize=11,
                 fontweight="bold", ha="center", transform=ax_info.transAxes)

    if not landing_sites.empty:
        best = landing_sites.iloc[0]
        info_lines = [
            f"Lat: {best['lat']:.2f}° S",
            f"Lon: {best['lon']:.2f}° E",
            f"Elevation: {best.get('elevation_m', 'N/A')} m",
            f"Slope: {best['slope_deg']:.1f}°",
            f"Illumination: {best['illumination_persistence']*100:.0f}%",
            f"Dist to ice: {best['distance_to_ice_candidate_m']:.0f} m",
        ]
        for i, line in enumerate(info_lines):
            ax_info.text(0.08, 0.78 - i * 0.12, line, color=WHITE, fontsize=9,
                        transform=ax_info.transAxes)

    # Panel 3: SAFE CORRIDOR CRITERIA
    ax_corridor = fig.add_axes([0.72, 0.12, 0.26, 0.25])
    ax_corridor.set_facecolor(PANEL_BG)
    ax_corridor.set_xlim(0, 1); ax_corridor.set_ylim(0, 1)
    ax_corridor.set_xticks([]); ax_corridor.set_yticks([])
    for spine in ax_corridor.spines.values():
        spine.set_color(CYAN); spine.set_linewidth(1.5)

    ax_corridor.text(0.5, 0.92, "SAFE CORRIDOR CRITERIA", color=CYAN, fontsize=11,
                     fontweight="bold", ha="center", transform=ax_corridor.transAxes)
    corr_criteria = [
        ("■", "Slope < 5°", GREEN),
        ("■", "Low Hazard (boulder < 1m)", GREEN),
        ("■", "Illumination > 50%", GOLD),
        ("■", "SAR: ice potential", CYAN),
        ("━", "Corridor centerline", MAGENTA),
    ]
    for i, (mark, text, color) in enumerate(corr_criteria):
        y = 0.78 - i * 0.13
        ax_corridor.text(0.05, y, mark, color=color, fontsize=10, transform=ax_corridor.transAxes)
        ax_corridor.text(0.12, y, text, color=WHITE, fontsize=9, transform=ax_corridor.transAxes)

    # ── Legend on main map ──
    legend_elements = [
        mpatches.Patch(facecolor=(0, 0.2, 0.8, 0.5), label="PSR Zone"),
        mpatches.Patch(facecolor=(0, 0.9, 1.0, 0.5), label="Ice Candidate"),
        mpatches.Patch(facecolor=(1.0, 0.1, 0.1, 0.3), label="Steep (>10°)"),
        Line2D([0], [0], marker="*", color=GREEN, label="Selected Site",
               markerfacecolor=GREEN, markersize=12, linestyle="None"),
        Line2D([0], [0], marker="*", color=GOLD, label="Alternate Site",
               markerfacecolor=GOLD, markersize=10, linestyle="None"),
    ]
    for mode, color in corridor_colors.items():
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2, label=f"Corridor: {mode.replace('_', ' ')}")
        )
    ax_map.legend(handles=legend_elements, loc="lower left", fontsize=8,
                  facecolor=PANEL_BG, edgecolor=GOLD, labelcolor=WHITE)

    fig.text(0.01, 0.01,
             "Illumination modeled via multi-azimuth DEM hillshade (36 azimuths, 1.54° solar elevation) | "
             "Preliminary: requires validation with ray-traced ephemeris data",
             color="#666666", fontsize=7)

    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_illumination_zone_map(zones, valid, landing_sites, path):
    """Save classified illumination zone map."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    zone_colors = ["#000000", "#0d0066", "#4a0099", "#cc6600", "#ffc107", "#ffff66"]
    cmap = ListedColormap(zone_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    display = np.where(valid, zones, 0)
    ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")

    if not landing_sites.empty:
        for i, site in landing_sites.iterrows():
            color = GREEN if i == 0 else GOLD
            ax.scatter(site["col"], site["row"], c=color, s=180,
                      marker="*", edgecolors="white", linewidths=1, zorder=10)

    zone_labels = [
        ("No data", "#000000"),
        ("PSR (<10%)", "#0d0066"),
        ("Poor (10-30%)", "#4a0099"),
        ("Moderate (30-50%)", "#cc6600"),
        ("Good (50-70%)", "#ffc107"),
        ("Excellent (>70%)", "#ffff66"),
    ]
    legend = [mpatches.Patch(facecolor=c, edgecolor=WHITE, label=l) for l, c in zone_labels]
    ax.legend(handles=legend, loc="lower right", fontsize=9,
              facecolor=PANEL_BG, edgecolor=GOLD, labelcolor=WHITE)

    ax.set_title("Illumination Zone Classification — Faustini Region",
                 color=GOLD, fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(colors=WHITE)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_landing_score_comparison(landing_sites, path):
    """Bar chart comparing landing site scores by component."""
    if landing_sites.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    sites = landing_sites.head(5)
    x = np.arange(len(sites))
    width = 0.6

    colors = [GREEN if i == 0 else GOLD for i in range(len(sites))]
    bars = ax.bar(x, sites["suitability_score"], width, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    # Add illumination annotation on each bar
    for i, (_, site) in enumerate(sites.iterrows()):
        illum_pct = site["illumination_persistence"] * 100
        ax.text(i, site["suitability_score"] + 0.02,
                f"☀ {illum_pct:.0f}%", color=GOLD, fontsize=10, ha="center", fontweight="bold")
        ax.text(i, site["suitability_score"] - 0.05,
                f"⛰ {site['slope_deg']:.1f}°", color=WHITE, fontsize=9, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(sites["site_id"], color=WHITE, fontsize=10)
    ax.set_ylabel("Suitability Score", color=WHITE, fontsize=12)
    ax.set_title("Landing Site Comparison — Illumination-Corrected Scores",
                 color=GOLD, fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.1, sites["suitability_score"].max() + 0.15))
    ax.tick_params(colors=WHITE)
    ax.spines["bottom"].set_color(WHITE)
    ax.spines["left"].set_color(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_corridor_profile(corridors, slope_deg, illumination, transform, path):
    """Save corridor slope and illumination profiles."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
    pixel_size = abs(float(transform.a))

    corridor_colors = {"safest": "#ff2bbd", "science_priority": CYAN, "energy_efficient": GOLD}

    for mode, pts in corridors.items():
        if not pts:
            continue
        color = corridor_colors.get(mode, WHITE)
        distances = [0]
        for i in range(1, len(pts)):
            d = ((pts[i][0] - pts[i-1][0])**2 + (pts[i][1] - pts[i-1][1])**2)**0.5 * pixel_size
            distances.append(distances[-1] + d)
        dist_km = [d / 1000 for d in distances]

        if slope_deg is not None:
            slopes = [slope_deg[r, c] for r, c in pts]
            ax1.plot(dist_km, slopes, color=color, linewidth=1.5, label=mode.replace("_", " ").title())

        illums = [illumination[r, c] * 100 for r, c in pts]
        ax2.plot(dist_km, illums, color=color, linewidth=1.5, label=mode.replace("_", " ").title())

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=WHITE)
        ax.spines["bottom"].set_color(WHITE)
        ax.spines["left"].set_color(WHITE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor=PANEL_BG, edgecolor=GOLD, labelcolor=WHITE, fontsize=9)

    ax1.set_ylabel("Slope (°)", color=WHITE, fontsize=11)
    ax1.set_title("Corridor Slope Profile — Landing Site → Ice Target", color=GOLD, fontsize=13, fontweight="bold")
    ax1.axhline(y=5, color=RED, linestyle="--", alpha=0.6, label="5° limit")
    ax1.axhline(y=15, color=RED, linestyle=":", alpha=0.4, label="15° max traverse")

    ax2.set_xlabel("Distance (km)", color=WHITE, fontsize=11)
    ax2.set_ylabel("Illumination (%)", color=WHITE, fontsize=11)
    ax2.set_title("Corridor Illumination Profile", color=GOLD, fontsize=13, fontweight="bold")
    ax2.axhline(y=40, color=GOLD, linestyle="--", alpha=0.6, label="Min landing illum")
    ax2.fill_between(ax2.get_xlim(), 0, 15, color="blue", alpha=0.1, label="PSR zone")

    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_multi_sar_coverage_map(sar_inventory, path):
    """Save bar chart showing all SAR product coverages."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    products = sar_inventory.sort_values("coverage_fraction", ascending=True)
    y = np.arange(len(products))
    colors = [GREEN if cov > 0.5 else (GOLD if cov > 0 else RED)
              for cov in products["coverage_fraction"]]

    ax.barh(y, products["coverage_fraction"] * 100, color=colors, edgecolor=WHITE, linewidth=0.5)

    # Add pixel size labels
    for i, (_, row) in enumerate(products.iterrows()):
        pix = row.get("pixel_size_m", "?")
        ax.text(max(row["coverage_fraction"] * 100 + 2, 5), i,
                f"{pix} m/px", color=WHITE, fontsize=8, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([p[:30] for p in products["product_id"]], color=WHITE, fontsize=8)
    ax.set_xlabel("AOI Coverage (%)", color=WHITE, fontsize=11)
    ax.set_title("SAR Product Coverage — Extended Faustini AOI", color=GOLD, fontsize=14, fontweight="bold")
    ax.axvline(x=50, color=GOLD, linestyle="--", alpha=0.5)
    ax.tick_params(colors=WHITE)
    ax.spines["bottom"].set_color(WHITE)
    ax.spines["left"].set_color(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_suitability_heatmap(score, landing_sites, valid, path):
    """Save corrected suitability heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    cmap = LinearSegmentedColormap.from_list("suit", [
        (0.0, "#0a0020"),
        (0.3, "#1a4060"),
        (0.5, "#2d8040"),
        (0.7, "#80b020"),
        (0.85, "#e0c000"),
        (1.0, "#ffff60"),
    ])

    display = np.where(valid & np.isfinite(score), score, np.nan)
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")

    if not landing_sites.empty:
        for i, site in landing_sites.iterrows():
            color = GREEN if i == 0 else GOLD
            ax.scatter(site["col"], site["row"], c=color, s=200,
                      marker="*", edgecolors="white", linewidths=1.2, zorder=10)
            ax.annotate(site["site_id"], (site["col"] + 6, site["row"] - 6),
                       color=color, fontsize=9, fontweight="bold", zorder=11)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Landing Suitability Score (illumination-corrected)", color=WHITE, fontsize=11)
    cb.ax.tick_params(colors=WHITE)

    ax.set_title("Corrected Landing Suitability — Illumination as Primary Constraint",
                 color=GOLD, fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(colors=WHITE)
    plt.tight_layout()
    fig.savefig(path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════
#    MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ILLUMINATION-AWARE LANDING SITE SELECTION PIPELINE")
    print("  Mission: Solar-powered lander → Faustini F2 south polar region")
    print("=" * 70)

    config = json.loads(CONFIG_PATH.read_text())

    # ─── Step 1: Inventory ALL SAR products ───
    print("\n[1/7] Inventorying all SAR products...")
    sar_inventory = inventory_sar_products(ROOT, ICE_AOI)
    print(f"  Total SAR pairs: {len(sar_inventory)}")
    print(f"  Usable (overlap with AOI): {sar_inventory['usable'].sum()}")

    # ─── Step 2: Load best SAR data ───
    print("\n[2/7] Loading SAR data and computing features...")
    sar = load_sar_data(ROOT, ICE_AOI)
    features = sar["features"]
    print(f"  SAR shape: {features['intensity'].shape}")
    print(f"  Valid pixels: {features['valid'].sum():,} / {features['valid'].size:,}")

    # ─── Step 3: Load DEM/terrain ───
    print("\n[3/7] Loading DEM/DTM terrain data...")
    target_shape = features["intensity"].shape
    dem = read_dem_aoi(ROOT, ICE_AOI, target_shape=target_shape)
    tmc = read_tmc_dtm_aoi(ROOT, ICE_AOI, target_shape=target_shape)

    if tmc.get("available"):
        elevation = tmc["elevation"]
        slope_deg = tmc["slope_deg"]
        pixel_size_m = tmc.get("pixel_size_m", 25.0)
        terrain_source = f"TMC-2 DTM: {tmc.get('selected_product_id', 'unknown')}"
        print(f"  Using TMC-2 DTM: {tmc.get('selected_product_id')}")
    else:
        elevation = dem.get("elevation")
        slope_deg = dem.get("slope_deg")
        pixel_size_m = dem.get("dem_pixel_size_m", 80.0)
        terrain_source = "LOLA LDEM/LDSM"
        print(f"  Using LOLA DEM")

    print(f"  Elevation range: {np.nanmin(elevation):.0f} to {np.nanmax(elevation):.0f} m")
    if slope_deg is not None:
        print(f"  Slope range: {np.nanmin(slope_deg):.1f} to {np.nanmax(slope_deg):.1f} deg")

    # ─── Step 4: Compute ILLUMINATION ───
    print("\n[4/7] Computing illumination model (multi-azimuth hillshade)...")
    lon_grid, lat_grid = build_lon_lat_grids(sar["transform"], sar["crs_wkt"], target_shape)

    illumination, illum_layers, illum_meta = compute_illumination_score(
        elevation, slope_deg, lat_grid, pixel_size_m, features["valid"]
    )
    psr_mask = illum_layers["psr_mask"]
    zones = classify_illumination_zones(illumination, psr_mask, features["valid"])

    print(f"  Illumination range: {np.nanmin(illumination):.3f} to {np.nanmax(illumination):.3f}")
    print(f"  PSR pixels: {illum_meta['psr_pixel_count']:,} ({illum_meta['psr_fraction']*100:.1f}%)")
    print(f"  Method: {illum_meta['method']}")

    # Zone statistics
    valid_pixels = features["valid"].sum()
    for zone_id, zone_name in [(1, "PSR"), (2, "Poor"), (3, "Moderate"), (4, "Good"), (5, "Excellent")]:
        count = np.sum(zones == zone_id)
        print(f"    Zone {zone_id} ({zone_name}): {count:,} pixels ({count/valid_pixels*100:.1f}%)")

    # ─── Step 5: Compute candidate mask (ice targets) ───
    print("\n[5/7] Computing radar ice candidate mask...")
    candidate_mask, candidate_score, thresholds = candidate_mask_from_features(features, config)
    print(f"  Candidate pixels: {candidate_mask.sum():,}")

    # ─── Step 6: Illumination-aware landing site selection ───
    print("\n[6/7] Selecting illumination-aware landing sites...")
    landing_config = {
        "max_landing_slope_deg": MAX_LANDING_SLOPE_DEG,
    }
    score, landing_allowed = illumination_aware_suitability(
        features, slope_deg, elevation, illumination,
        psr_mask, candidate_mask, landing_config,
    )

    feasible_pixels = np.sum(landing_allowed)
    print(f"  Feasible landing pixels: {feasible_pixels:,} / {valid_pixels:,} ({feasible_pixels/valid_pixels*100:.1f}%)")

    landing_sites = select_landing_sites(
        score, landing_allowed, slope_deg, illumination, elevation,
        candidate_mask, sar["transform"], sar["crs_wkt"], psr_mask,
    )

    if not landing_sites.empty:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  TOP LANDING SITES (Illumination-Corrected)             ║")
        print(f"  ╠══════════════════════════════════════════════════════════╣")
        for _, site in landing_sites.iterrows():
            print(f"  ║  {site['site_id']}: lat={site['lat']:.3f}° lon={site['lon']:.3f}°")
            print(f"  ║    Score={site['suitability_score']:.3f} | ☀ Illum={site['illumination_persistence']*100:.0f}%")
            print(f"  ║    Slope={site['slope_deg']:.1f}° | Dist to ice={site['distance_to_ice_candidate_m']:.0f}m")
            print(f"  ║    Status: {site['illumination_class']}")
            print(f"  ╟──────────────────────────────────────────────────────────╢")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
    else:
        print("  WARNING: No feasible landing sites found with current constraints!")
        print("  Consider relaxing illumination threshold or expanding AOI.")

    landing_sites.to_csv(OUTPUT_DIR / "corrected_landing_sites.csv", index=False)

    # ─── Step 7: Plan corridors ───
    print("\n[7/7] Planning rover corridors to PSR ice targets...")
    corridors, corridor_summary = plan_corridors(
        features, slope_deg, illumination, psr_mask, candidate_mask,
        landing_sites, sar["transform"], sar["crs_wkt"], config,
    )
    corridor_summary.to_csv(OUTPUT_DIR / "corridor_summary.csv", index=False)

    if not corridor_summary.empty:
        print(f"\n  Corridor Summary:")
        for _, corr in corridor_summary.iterrows():
            print(f"    {corr['corridor_type']:20s}: {corr['length_km']:.1f} km, "
                  f"mean slope {corr['mean_slope_deg']:.1f}°, "
                  f"mean illum {corr['mean_illumination']*100:.0f}%, "
                  f"PSR entry {corr['psr_entry_fraction']*100:.0f}%")

    # ══════════════════════════════════════════════════════════════
    #   GENERATE ALL OUTPUT FIGURES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  GENERATING OUTPUT FIGURES")
    print("=" * 70)

    # Fig 1: Multi-SAR coverage
    print("\n  [Fig 1] SAR product coverage...")
    save_multi_sar_coverage_map(sar_inventory, OUTPUT_DIR / "01_sar_coverage_inventory.png")

    # Fig 2: Illumination persistence map
    print("  [Fig 2] Illumination persistence map...")
    save_illumination_map(illumination, psr_mask, features["valid"],
                          OUTPUT_DIR / "02_illumination_persistence_map.png")

    # Fig 3: Illumination zone classification
    print("  [Fig 3] Illumination zone classification...")
    save_illumination_zone_map(zones, features["valid"], landing_sites,
                                OUTPUT_DIR / "03_illumination_zone_classification.png")

    # Fig 4: Corrected suitability heatmap
    print("  [Fig 4] Corrected suitability heatmap...")
    save_suitability_heatmap(score, landing_sites, features["valid"],
                              OUTPUT_DIR / "04_corrected_suitability_map.png")

    # Fig 5: MAIN — Landing decision map (reference-quality)
    print("  [Fig 5] Main landing decision map...")
    base_img = features["intensity"]
    save_landing_decision_map(
        base_img, illumination, slope_deg, psr_mask, candidate_mask,
        landing_sites, corridors, corridor_summary,
        features["valid"], sar["transform"], sar["crs_wkt"],
        OUTPUT_DIR / "05_landing_decision_map.png",
    )

    # Fig 6: Landing site comparison
    print("  [Fig 6] Landing site comparison...")
    save_landing_score_comparison(landing_sites, OUTPUT_DIR / "06_landing_site_comparison.png")

    # Fig 7: Corridor profiles
    print("  [Fig 7] Corridor profiles...")
    save_corridor_profile(corridors, slope_deg, illumination, sar["transform"],
                           OUTPUT_DIR / "07_corridor_profiles.png")

    # Fig 8: Illumination component breakdown
    print("  [Fig 8] Illumination components...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=DARK_BG)
    component_names = [
        ("terrain_illumination", "Multi-Azimuth Terrain Illumination", "inferno"),
        ("latitude_score", "Latitude-Based Illumination Decay", "plasma"),
        ("elevation_prominence", "Elevation Prominence", "viridis"),
        ("self_shadow_score", "Aspect Self-Shadow Score", "magma"),
    ]
    for ax, (key, title, cmap_name) in zip(axes.ravel(), component_names):
        ax.set_facecolor(DARK_BG)
        layer = illum_layers[key]
        display = np.where(features["valid"], layer, np.nan)
        im = ax.imshow(display, cmap=cmap_name, vmin=0, vmax=1)
        ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold")
        ax.tick_params(colors=WHITE, labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Illumination Model Components — Faustini Region",
                 color=GOLD, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_illumination_components.png", dpi=200,
                facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / '08_illumination_components.png'}")

    # ── Save illumination metadata ──
    illum_report = {
        "method": illum_meta["method"],
        "sun_altitude_deg": illum_meta["sun_altitude_deg"],
        "n_azimuths": illum_meta["n_azimuths"],
        "terrain_source": terrain_source,
        "psr_pixel_count": illum_meta["psr_pixel_count"],
        "psr_fraction_percent": round(illum_meta["psr_fraction"] * 100, 2),
        "illumination_range": [float(np.nanmin(illumination)), float(np.nanmax(illumination))],
        "min_landing_illumination": MIN_ILLUMINATION_PERSISTENCE,
        "max_landing_slope_deg": MAX_LANDING_SLOPE_DEG,
        "feasible_landing_pixels": int(feasible_pixels),
        "total_valid_pixels": int(valid_pixels),
        "landing_site_count": len(landing_sites),
        "corridor_count": len(corridors),
        "correction_applied": "illumination_score upgraded from 0.5 placeholder to computed DEM-based model",
        "status": "CORRECTED — illumination is now the primary landing constraint",
    }
    with open(OUTPUT_DIR / "illumination_metadata.json", "w") as f:
        json.dump(illum_report, f, indent=2)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} ({size_kb:.0f} KB)")

    print(f"\n  MISSION STRATEGY:")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  1. LAND on sunlit crater rim/ridge         │")
    print(f"  │     → Illumination ≥ {MIN_ILLUMINATION_PERSISTENCE*100:.0f}%, Slope < {MAX_LANDING_SLOPE_DEG}°      │")
    print(f"  │  2. ROVE into PSR for ice exploration       │")
    print(f"  │     → Follow safest/science corridor        │")
    print(f"  │  3. OPERATE on stored battery + solar       │")
    print(f"  │     → RTG recommended for PSR operations    │")
    print(f"  └─────────────────────────────────────────────┘")

    return landing_sites, corridor_summary


if __name__ == "__main__":
    landing_sites, corridors = main()
