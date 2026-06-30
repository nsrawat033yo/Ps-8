from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from lunar_icenav.features.texture import robust_normalize
from lunar_icenav.preprocessing.aoi import map_to_lonlat, pixel_to_map


def suitability_map(
    candidate_mask: np.ndarray,
    features: dict[str, np.ndarray],
    slope_deg: np.ndarray | None,
    illumination: np.ndarray | None,
    transform,
    crs_wkt: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    valid = features["valid"].astype(bool)
    texture = features["texture"]
    intensity = features["intensity"]
    planning = config.get("planning", {})
    max_slope = float(planning.get("max_slope_deg", 8.0))
    soft_max_slope = float(planning.get("soft_max_slope_deg", 25.0))
    if slope_deg is None:
        slope = np.zeros_like(texture, dtype="float32") + np.nan
        slope_score = np.ones_like(texture, dtype="float32") * 0.5
        slope_ok = valid.copy()
    else:
        slope = slope_deg.astype("float32")
        slope_score = 1.0 - np.clip(slope / max_slope, 0, 1)
        slope_ok = np.isfinite(slope) & (slope < max_slope)

    hazard_score = robust_normalize(texture, valid)
    low_hazard_score = 1.0 - hazard_score
    distance_to_candidate_px = ndi.distance_transform_edt(~candidate_mask)
    proximity = 1.0 / (1.0 + distance_to_candidate_px / 40.0)
    
    if illumination is not None:
        illumination_score = np.clip(illumination, 0, 1).astype("float32")
        illum_ok = illumination_score >= 0.40
        illum_status = "Real proxy applied"
    else:
        illumination_score = np.where(valid, 0.5, np.nan).astype("float32")
        illum_ok = valid.copy()
        illum_status = "missing_real_layer_neutral_score"
        
    temperature_score = np.where(valid, 0.5, np.nan).astype("float32")
    clearance = int(planning.get("candidate_clearance_pixels", 3))
    
    landing_allowed = valid & slope_ok & illum_ok & (~candidate_mask)
    if clearance > 0:
        landing_allowed &= distance_to_candidate_px >= clearance

    score = (
        0.35 * illumination_score
        + 0.25 * slope_score
        + 0.25 * low_hazard_score
        + 0.15 * proximity
    ).astype("float32")
    score[~landing_allowed] = np.nan
    layers = {
        "slope_score": slope_score.astype("float32"),
        "low_hazard_score": low_hazard_score.astype("float32"),
        "candidate_proximity_score": proximity.astype("float32"),
        "illumination_score": illumination_score,
        "temperature_score": temperature_score,
        "landing_allowed": landing_allowed,
        "slope_deg": slope,
        "illumination_layer_status": illum_status,
        "temperature_layer_status": "missing_real_layer_neutral_score",
    }
    sites = top_landing_sites(score, slope, distance_to_candidate_px, transform, crs_wkt, int(planning.get("top_site_count", 5)))
    return score, sites, layers


def top_landing_sites(
    score: np.ndarray,
    slope_deg: np.ndarray,
    distance_px: np.ndarray,
    transform,
    crs_wkt: str,
    count: int,
) -> pd.DataFrame:
    working = np.where(np.isfinite(score), score, -np.inf).copy()
    rows: list[dict[str, Any]] = []
    pixel_size = abs(float(transform.a))
    for idx in range(count):
        flat = int(np.argmax(working))
        if not np.isfinite(working.ravel()[flat]) or working.ravel()[flat] == -np.inf:
            break
        row, col = np.unravel_index(flat, working.shape)
        x, y = pixel_to_map(transform, float(row), float(col))
        lon, lat = map_to_lonlat(crs_wkt, np.array([x]), np.array([y]))
        rows.append({
            "site_id": f"L-{idx + 1:02d}",
            "row": int(row),
            "col": int(col),
            "x_m": x,
            "y_m": y,
            "lat": float(lat[0]),
            "lon": float(lon[0]),
            "suitability_score": float(score[row, col]),
            "slope_deg": float(slope_deg[row, col]) if np.isfinite(slope_deg[row, col]) else np.nan,
            "distance_to_candidate_m": float(distance_px[row, col] * pixel_size),
            "interpretation": "preliminary candidate landing site for prototype screening; not mission-certified",
        })
        r0, r1 = max(0, row - 18), min(score.shape[0], row + 19)
        c0, c1 = max(0, col - 18), min(score.shape[1], col + 19)
        working[r0:r1, c0:c1] = -np.inf
    return pd.DataFrame(rows)
