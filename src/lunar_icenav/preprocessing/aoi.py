from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window, from_bounds


def aoi_lon_lat_corners(aoi: dict[str, float]) -> tuple[list[float], list[float]]:
    lons = [aoi["lon_min"], aoi["lon_max"], aoi["lon_max"], aoi["lon_min"]]
    lats = [aoi["lat_min"], aoi["lat_min"], aoi["lat_max"], aoi["lat_max"]]
    return lons, lats


def transform_aoi_to_dataset(ds: rasterio.DatasetReader, aoi: dict[str, float]) -> tuple[float, float, float, float]:
    if ds.crs is None:
        raise ValueError("Dataset has no CRS; cannot transform AOI.")
    crs = CRS.from_wkt(ds.crs.to_wkt())
    transformer = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    lons, lats = aoi_lon_lat_corners(aoi)
    xs, ys = transformer.transform(lons, lats)
    return min(xs), min(ys), max(xs), max(ys)


def bounds_intersection_fraction(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    inter = (right - left) * (top - bottom)
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    return float(inter / area)


def clipped_window_for_aoi(ds: rasterio.DatasetReader, aoi: dict[str, float], pad_pixels: int = 0) -> tuple[Window, tuple[float, float, float, float]]:
    aoi_bounds = transform_aoi_to_dataset(ds, aoi)
    window = from_bounds(*aoi_bounds, transform=ds.transform)
    window = window.round_offsets().round_lengths()
    if pad_pixels:
        window = Window(window.col_off - pad_pixels, window.row_off - pad_pixels, window.width + 2 * pad_pixels, window.height + 2 * pad_pixels)
    full = Window(0, 0, ds.width, ds.height)
    window = window.intersection(full)
    return window, aoi_bounds


def dataset_bounds_tuple(ds: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    return (float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top))


def evaluate_sar_pair(pair: dict[str, Path], aoi: dict[str, float]) -> dict[str, Any]:
    with rasterio.open(pair["lh"]) as ds:
        if ds.crs is None:
            return {"coverage_fraction": 0.0, "pixel_size_m": None, "reason": "missing CRS"}
        aoi_bounds = transform_aoi_to_dataset(ds, aoi)
        coverage = bounds_intersection_fraction(aoi_bounds, dataset_bounds_tuple(ds))
        return {
            "coverage_fraction": coverage,
            "pixel_size_m": float(abs(ds.transform.a)),
            "width": ds.width,
            "height": ds.height,
            "bounds": dataset_bounds_tuple(ds),
            "crs_wkt": ds.crs.to_wkt(),
        }


def choose_best_sar_pair(pairs: list[dict[str, Path]], aoi: dict[str, float]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for pair in pairs:
        score = evaluate_sar_pair(pair, aoi)
        record = dict(pair)
        record.update(score)
        record["product_id"] = pair["lh"].parts[-5] if len(pair["lh"].parts) >= 5 else pair["lh"].stem
        scored.append(record)
    usable = [r for r in scored if r.get("coverage_fraction", 0) > 0 and r.get("pixel_size_m")]
    if not usable:
        raise RuntimeError("No SRI LH/LV SAR pair intersects the configured AOI.")
    usable.sort(key=lambda r: (-float(r["coverage_fraction"]), float(r["pixel_size_m"])))
    return usable[0], scored


def pixel_to_map(transform: rasterio.Affine, row: float, col: float) -> tuple[float, float]:
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def map_to_lonlat(crs_wkt: str, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crs = CRS.from_wkt(crs_wkt)
    transformer = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = transformer.transform(xs, ys)
    return np.asarray(lon), np.asarray(lat)
