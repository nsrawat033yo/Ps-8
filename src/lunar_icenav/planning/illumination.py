"""Illumination modeling for lunar south polar landing site selection.

This module computes illumination persistence proxies using DEM-derived
terrain geometry (elevation, slope, aspect, hillshade) and latitude.
It identifies Permanently Shadowed Regions (PSRs), sunlit crater rims,
and classifies zones for solar-powered mission feasibility.

Scientific basis:
- At the lunar south pole, the Sun elevation never exceeds ~1.54 degrees.
- Terrain features that protrude above the local horizon (crater rims,
  ridges, peaks) receive significantly more illumination than low-lying
  areas (crater floors, valleys).
- PSRs form in topographic lows where the Sun never rises above the
  local horizon, typically inside deep craters poleward of ~85 deg S.
- This is an *approximation*; real illumination requires ray-tracing with
  ephemeris data (Mazarico et al. 2011 methodology).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


# Maximum solar elevation at the south pole (degrees)
MAX_SOLAR_ELEVATION_DEG = 1.54


def compute_aspect(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Compute terrain aspect (direction of steepest slope) in degrees from north."""
    filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
    gy, gx = np.gradient(filled, pixel_size_m, pixel_size_m)
    aspect = np.degrees(np.arctan2(-gx, gy))  # CW from north
    aspect = np.mod(aspect, 360)
    return aspect.astype("float32")


def compute_hillshade(
    elevation: np.ndarray,
    pixel_size_m: float,
    sun_altitude_deg: float = 1.5,
    sun_azimuth_deg: float = 180.0,
) -> np.ndarray:
    """Compute hillshade given Sun position. Values 0 (shadow) to 1 (fully lit)."""
    filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
    gy, gx = np.gradient(filled, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    aspect_rad = np.arctan2(-gx, gy)
    sun_alt_rad = np.radians(sun_altitude_deg)
    sun_az_rad = np.radians(sun_azimuth_deg)
    hs = (
        np.sin(sun_alt_rad) * np.cos(slope_rad)
        + np.cos(sun_alt_rad) * np.sin(slope_rad)
        * np.cos(sun_az_rad - aspect_rad)
    )
    return np.clip(hs, 0, 1).astype("float32")


def multi_azimuth_illumination(
    elevation: np.ndarray,
    pixel_size_m: float,
    sun_altitude_deg: float = 1.5,
    n_azimuths: int = 36,
) -> np.ndarray:
    """Compute illumination persistence as fraction of azimuths where pixel is lit.

    Simulates Sun sweeping around 360 degrees at a fixed low altitude
    (appropriate for the lunar south pole where the Sun circles near
    the horizon). Returns the fraction of azimuths where each pixel
    receives direct illumination.

    This is the core illumination proxy replacing the hardcoded 0.5 placeholder.
    """
    valid = np.isfinite(elevation)
    if not np.any(valid):
        return np.zeros_like(elevation, dtype="float32")

    filled = np.where(valid, elevation, np.nanmedian(elevation))
    # Pre-compute slope and aspect
    gy, gx = np.gradient(filled, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    aspect_rad = np.arctan2(-gx, gy)
    sun_alt_rad = np.radians(sun_altitude_deg)

    lit_count = np.zeros(elevation.shape, dtype="float32")

    for i in range(n_azimuths):
        azimuth_deg = i * (360.0 / n_azimuths)
        sun_az_rad = np.radians(azimuth_deg)

        # Hillshade at this azimuth
        hs = (
            np.sin(sun_alt_rad) * np.cos(slope_rad)
            + np.cos(sun_alt_rad) * np.sin(slope_rad)
            * np.cos(sun_az_rad - aspect_rad)
        )
        # Pixel is illuminated if hillshade > 0
        lit_count += (hs > 0.01).astype("float32")

    illumination_persistence = lit_count / float(n_azimuths)
    illumination_persistence[~valid] = np.nan
    return illumination_persistence.astype("float32")


def horizon_shadow_mask(
    elevation: np.ndarray,
    pixel_size_m: float,
    sun_altitude_deg: float = 1.5,
    n_azimuths: int = 36,
    max_distance_px: int = 200,
) -> np.ndarray:
    """Ray-trace simplified horizon shadows from each direction.

    For each azimuth, cast rays and check if the terrain horizon angle
    exceeds the Sun altitude. A pixel is in shadow if any terrain between
    it and the Sun blocks the line of sight.

    Returns: fraction of azimuths where pixel is NOT shadowed (0..1).
    """
    valid = np.isfinite(elevation)
    if not np.any(valid):
        return np.zeros_like(elevation, dtype="float32")

    filled = np.where(valid, elevation, np.nanmin(elevation[valid]))
    rows, cols = filled.shape
    sun_tan = np.tan(np.radians(sun_altitude_deg))
    lit_count = np.zeros(filled.shape, dtype="float32")

    for i in range(n_azimuths):
        azimuth_deg = i * (360.0 / n_azimuths)
        az_rad = np.radians(azimuth_deg)
        # Direction TO the sun (ray direction)
        dr = -np.cos(az_rad)
        dc = np.sin(az_rad)
        # For each pixel, check maximum horizon angle in this direction
        shadow_map = np.zeros(filled.shape, dtype=bool)
        step = max(abs(dr), abs(dc))
        if step < 0.01:
            continue
        ndr = dr / step
        ndc = dc / step
        # Vectorized approach: shift the array and compare
        max_horizon = np.full(filled.shape, -np.inf, dtype="float32")
        for dist in range(1, min(max_distance_px, max(rows, cols))):
            # Shift elevation array by dist pixels in the sun direction
            shift_r = int(round(ndr * dist))
            shift_c = int(round(ndc * dist))
            if abs(shift_r) >= rows or abs(shift_c) >= cols:
                break
            shifted = np.roll(np.roll(filled, -shift_r, axis=0), -shift_c, axis=1)
            # Invalidate wrapped edges
            if shift_r > 0:
                shifted[:shift_r, :] = -np.inf
            elif shift_r < 0:
                shifted[shift_r:, :] = -np.inf
            if shift_c > 0:
                shifted[:, :shift_c] = -np.inf
            elif shift_c < 0:
                shifted[:, shift_c:] = -np.inf
            # Horizon angle from current pixel to shifted pixel
            distance_m = dist * pixel_size_m * step
            horizon_angle = (shifted - filled) / max(distance_m, 1e-6)
            max_horizon = np.maximum(max_horizon, horizon_angle)

        # Pixel is shadowed if max horizon angle > sun tangent angle
        shadow_map = max_horizon > sun_tan
        lit_count += (~shadow_map).astype("float32")

    result = lit_count / float(n_azimuths)
    result[~valid] = np.nan
    return result.astype("float32")


def compute_illumination_score(
    elevation: np.ndarray,
    slope_deg: np.ndarray | None,
    lat_grid: np.ndarray,
    pixel_size_m: float,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Compute composite illumination score for landing site selection.

    Returns:
        illumination_score: 0..1 array (higher = more sunlight)
        layers: dict of intermediate arrays
        metadata: dict of status information
    """
    valid_mask = valid.astype(bool) & np.isfinite(lat_grid)
    if elevation is not None:
        valid_mask &= np.isfinite(elevation)

    # --- Component 1: Multi-azimuth terrain illumination ---
    if elevation is not None:
        terrain_illum = multi_azimuth_illumination(
            elevation, pixel_size_m,
            sun_altitude_deg=MAX_SOLAR_ELEVATION_DEG,
            n_azimuths=36,
        )
        terrain_source = "multi_azimuth_hillshade"
    else:
        terrain_illum = np.where(valid_mask, 0.5, np.nan).astype("float32")
        terrain_source = "placeholder_no_elevation"

    # --- Component 2: Latitude-based illumination decay ---
    abs_lat = np.abs(lat_grid)
    # At the pole, illumination drops sharply
    # ~83 deg: good illumination on rims (~80%+)
    # ~85 deg: moderate (~50-60%)
    # ~87 deg: poor unless elevated (~20-30%)
    # ~89 deg: near-zero except peaks of eternal light
    lat_score = np.clip((90.0 - abs_lat - 3.0) / 4.0, 0, 1).astype("float32")
    lat_score[~valid_mask] = np.nan

    # --- Component 3: Elevation prominence ---
    if elevation is not None:
        # Local prominence: how much above local median
        local_median = ndi.median_filter(
            np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation)),
            size=51,
        )
        prominence = np.clip((elevation - local_median) / 500.0, -0.5, 0.5) + 0.5
        prominence = prominence.astype("float32")
        prominence[~valid_mask] = np.nan
        prominence_source = "elevation_prominence"
    else:
        prominence = np.where(valid_mask, 0.5, np.nan).astype("float32")
        prominence_source = "placeholder"

    # --- Component 4: Self-shadowing from slope/aspect ---
    if slope_deg is not None and elevation is not None:
        # Equator-facing slopes get more illumination
        aspect = compute_aspect(elevation, pixel_size_m)
        # At south pole, sun comes from the equator side (~north)
        # Favor north-facing slopes (aspect ~0 or ~360)
        north_facing = np.cos(np.radians(aspect))  # +1 = north, -1 = south
        # Steeper north-facing slopes catch more sun at low angles
        slope_rad = np.radians(slope_deg)
        aspect_bonus = np.clip(north_facing * np.sin(slope_rad) * 5.0, -0.3, 0.3)
        self_shadow_score = np.clip(0.5 + aspect_bonus, 0, 1).astype("float32")
        self_shadow_score[~valid_mask] = np.nan
    else:
        self_shadow_score = np.where(valid_mask, 0.5, np.nan).astype("float32")

    # --- Composite illumination score ---
    illumination = (
        0.40 * terrain_illum
        + 0.25 * lat_score
        + 0.20 * prominence
        + 0.15 * self_shadow_score
    ).astype("float32")
    illumination[~valid_mask] = np.nan

    # --- PSR detection ---
    psr_mask = detect_psr(illumination, terrain_illum, abs_lat, elevation, valid_mask)

    layers = {
        "terrain_illumination": terrain_illum,
        "latitude_score": lat_score,
        "elevation_prominence": prominence,
        "self_shadow_score": self_shadow_score,
        "psr_mask": psr_mask,
    }

    metadata = {
        "terrain_source": terrain_source,
        "prominence_source": prominence_source,
        "n_azimuths": 36,
        "sun_altitude_deg": MAX_SOLAR_ELEVATION_DEG,
        "method": "multi_azimuth_hillshade + latitude_decay + elevation_prominence + aspect_self_shadow",
        "psr_pixel_count": int(np.nansum(psr_mask)),
        "psr_fraction": float(np.nansum(psr_mask) / max(np.sum(valid_mask), 1)),
        "status": "computed",
    }

    return illumination, layers, metadata


def detect_psr(
    illumination: np.ndarray,
    terrain_illum: np.ndarray,
    abs_lat: np.ndarray,
    elevation: np.ndarray | None,
    valid: np.ndarray,
) -> np.ndarray:
    """Detect Permanently Shadowed Regions.

    A pixel is classified as PSR if:
    1. Terrain illumination fraction < 10% (almost never lit), OR
    2. Latitude > 86.5 deg AND elevation is below local 25th percentile
    """
    psr = np.zeros(illumination.shape, dtype=bool)

    # Criterion 1: Very low terrain illumination
    psr |= (valid & (terrain_illum < 0.10))

    # Criterion 2: Deep polar + low elevation
    if elevation is not None:
        elev_valid = elevation[valid & np.isfinite(elevation)]
        if elev_valid.size > 0:
            p25 = np.nanpercentile(elev_valid, 25)
            psr |= (valid & (abs_lat > 86.5) & (elevation < p25))

    # Criterion 3: Composite illumination very low
    psr |= (valid & np.isfinite(illumination) & (illumination < 0.15))

    psr &= valid
    return psr


def classify_illumination_zones(
    illumination: np.ndarray,
    psr_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Classify terrain into illumination zones.

    Returns uint8 array:
        0 = invalid/nodata
        1 = PSR (permanently shadowed, <10% illumination)
        2 = Poor illumination (10-30%)
        3 = Moderate illumination (30-50%)
        4 = Good illumination (50-70%)
        5 = Excellent illumination (>70%, crater rim / peak of near-eternal light)
    """
    zones = np.zeros(illumination.shape, dtype="uint8")
    zones[valid & psr_mask] = 1
    zones[valid & ~psr_mask & (illumination < 0.30)] = 2
    zones[valid & ~psr_mask & (illumination >= 0.30) & (illumination < 0.50)] = 3
    zones[valid & ~psr_mask & (illumination >= 0.50) & (illumination < 0.70)] = 4
    zones[valid & ~psr_mask & (illumination >= 0.70)] = 5
    return zones


def find_crater_rim_pixels(
    elevation: np.ndarray,
    slope_deg: np.ndarray | None,
    illumination: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Identify crater rim pixels that are elevated and well-illuminated.

    Crater rims are:
    - Locally elevated (above local median)
    - Moderate slope (2-15 degrees typically)
    - Higher illumination than surroundings
    """
    if elevation is None:
        return np.zeros_like(valid, dtype=bool)

    filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
    local_median = ndi.median_filter(filled, size=31)
    elevated = (elevation - local_median) > 50  # >50m above local median

    if slope_deg is not None:
        moderate_slope = (slope_deg > 1.0) & (slope_deg < 20.0)
    else:
        moderate_slope = np.ones_like(valid, dtype=bool)

    well_lit = np.isfinite(illumination) & (illumination > 0.40)

    rim = valid & elevated & moderate_slope & well_lit
    # Clean up with morphological operations
    rim = ndi.binary_opening(rim, iterations=1)
    return rim
