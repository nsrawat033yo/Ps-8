from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from lunar_icenav.preprocessing.aoi import map_to_lonlat, pixel_to_map


def candidate_mask_from_features(features: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    valid = features["valid"].astype(bool)
    ratio = features["cpr_style_ratio_proxy"]
    intensity = features["intensity"]
    texture = features["texture"]
    screen_cfg = config.get("candidate_screen", {})
    rq = float(screen_cfg.get("ratio_quantile", 0.88))
    iq = float(screen_cfg.get("intensity_quantile", 0.70))
    tq = float(screen_cfg.get("max_texture_quantile", 0.98))
    ratio_thr = float(np.nanquantile(ratio[valid], rq))
    intensity_thr = float(np.nanquantile(intensity[valid], iq))
    texture_thr = float(np.nanquantile(texture[valid], tq))
    score = features["candidate_score"]
    score_thr = float(np.nanquantile(score[valid], 0.82))

    mask = valid & (ratio >= ratio_thr) & (intensity >= intensity_thr) & (texture <= texture_thr) & (score >= score_thr)
    min_pixels = int(screen_cfg.get("min_patch_pixels", 8))
    mask = remove_small_components(mask, min_pixels)
    if mask.sum() < min_pixels:
        ratio_thr = float(np.nanquantile(ratio[valid], 0.82))
        intensity_thr = float(np.nanquantile(intensity[valid], 0.62))
        score_thr = float(np.nanquantile(score[valid], 0.76))
        mask = valid & (ratio >= ratio_thr) & (intensity >= intensity_thr) & (score >= score_thr)
        mask = remove_small_components(mask, max(3, min_pixels // 2))

    thresholds = {
        "ratio_threshold": ratio_thr,
        "intensity_threshold": intensity_thr,
        "texture_threshold": texture_thr,
        "score_threshold": score_thr,
    }
    return mask.astype(bool), score.astype("float32"), thresholds


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def summarize_patches(
    mask: np.ndarray,
    features: dict[str, np.ndarray],
    transform,
    crs_wkt: str,
    product_id: str,
) -> pd.DataFrame:
    labels, n = ndi.label(mask)
    rows: list[dict[str, Any]] = []
    pixel_area_m2 = abs(float(transform.a * transform.e))
    for label_id in range(1, n + 1):
        yy, xx = np.where(labels == label_id)
        if yy.size == 0:
            continue
        centroid_row = float(np.mean(yy))
        centroid_col = float(np.mean(xx))
        x, y = pixel_to_map(transform, centroid_row, centroid_col)
        lon, lat = map_to_lonlat(crs_wkt, np.array([x]), np.array([y]))
        vals = {
            "ratio": features["cpr_style_ratio_proxy"][yy, xx],
            "intensity": features["intensity"][yy, xx],
            "imbalance": features["polarization_imbalance_proxy"][yy, xx],
            "score": features["candidate_score"][yy, xx],
            "texture": features["texture"][yy, xx],
        }
        rows.append({
            "candidate_id": f"C-{label_id:03d}",
            "product_id": product_id,
            "area_pixels": int(yy.size),
            "area_m2": float(yy.size * pixel_area_m2),
            "centroid_row": centroid_row,
            "centroid_col": centroid_col,
            "centroid_x_m": x,
            "centroid_y_m": y,
            "centroid_lat": float(lat[0]),
            "centroid_lon": float(lon[0]),
            "mean_ratio_proxy": float(np.nanmean(vals["ratio"])),
            "mean_intensity": float(np.nanmean(vals["intensity"])),
            "mean_pol_imbalance_proxy": float(np.nanmean(vals["imbalance"])),
            "mean_texture": float(np.nanmean(vals["texture"])),
            "mean_candidate_score": float(np.nanmean(vals["score"])),
            "interpretation": "radar-based candidate subsurface ice signature; requires independent validation",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mean_candidate_score", "area_pixels"], ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df
