from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage as ndi

from lunar_icenav.preprocessing.aoi import clipped_window_for_aoi


def find_dem_paths(root: Path) -> dict[str, Path | None]:
    ldem = next(iter(sorted(root.rglob("LDEM*_ADJ.tif*"))), None)
    ldsm = next(iter(sorted(root.rglob("LDSM*_ADJ.tif*"))), None)
    return {"elevation": ldem, "slope": ldsm}


def read_dem_aoi(root: Path, aoi: dict[str, float], target_shape: tuple[int, int]) -> dict[str, Any]:
    paths = find_dem_paths(root)
    if paths["elevation"] is None and paths["slope"] is None:
        raise RuntimeError("No LDEM/LDSM DEM files were found.")

    out: dict[str, Any] = {"paths": {k: str(v) if v else "" for k, v in paths.items()}}
    if paths["elevation"] is not None:
        with rasterio.open(paths["elevation"]) as ds:
            window, _ = clipped_window_for_aoi(ds, aoi, pad_pixels=1)
            elev = ds.read(1, window=window).astype("float32")
            out["elevation"] = resize_to(elev, target_shape)
            out["elevation_native"] = elev
            out["dem_transform"] = ds.window_transform(window)
            out["dem_crs_wkt"] = ds.crs.to_wkt()
            out["dem_pixel_size_m"] = float(abs(ds.transform.a))
    if paths["slope"] is not None:
        with rasterio.open(paths["slope"]) as ds:
            window, _ = clipped_window_for_aoi(ds, aoi, pad_pixels=1)
            slope = ds.read(1, window=window).astype("float32")
            out["slope_deg"] = resize_to(slope, target_shape)
            out["slope_native"] = slope
    elif "elevation_native" in out:
        out["slope_deg"] = resize_to(compute_slope_deg(out["elevation_native"], out.get("dem_pixel_size_m", 80.0)), target_shape)

    if "slope_deg" in out:
        out["slope_deg"] = np.where(np.isfinite(out["slope_deg"]), out["slope_deg"], np.nan).astype("float32")
    return out


def compute_slope_deg(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
    gy, gx = np.gradient(filled, pixel_size_m, pixel_size_m)
    return np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy))).astype("float32")


def resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype("float32")
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    filled = np.where(np.isfinite(arr), arr, np.nanmedian(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0)
    return ndi.zoom(filled, factors, order=1).astype("float32")
