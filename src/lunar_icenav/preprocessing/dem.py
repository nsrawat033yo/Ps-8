from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage as ndi

from lunar_icenav.preprocessing.aoi import bounds_intersection_fraction, clipped_window_for_aoi, dataset_bounds_tuple, transform_aoi_to_dataset


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


def find_tmc_dtm_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("ch2_tmc*_d_dtm_*.tif"))


def evaluate_tmc_dtm_coverage(root: Path, aoi: dict[str, float]) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for path in find_tmc_dtm_paths(root):
        try:
            with rasterio.open(path) as ds:
                coverage = 0.0
                if ds.crs is not None:
                    aoi_bounds = transform_aoi_to_dataset(ds, aoi)
                    coverage = bounds_intersection_fraction(aoi_bounds, dataset_bounds_tuple(ds))
                coverage_class = coverage_class_from_fraction(coverage)
                rows.append({
                    "product_id": product_id_for_tmc(path),
                    "path": str(path),
                    "coverage_fraction": float(coverage),
                    "coverage_class": coverage_class,
                    "usable_for_analysis": coverage_class in {"FULL", "PARTIAL"},
                    "selected_for_terrain": False,
                    "pixel_size_m": float(abs(ds.transform.a)),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "bounds": tuple(float(v) for v in dataset_bounds_tuple(ds)),
                    "reason": coverage_reason(coverage_class),
                })
        except Exception as exc:
            rows.append({
                "product_id": product_id_for_tmc(path),
                "path": str(path),
                "coverage_fraction": 0.0,
                "coverage_class": "NO COVERAGE",
                "usable_for_analysis": False,
                "selected_for_terrain": False,
                "pixel_size_m": np.nan,
                "width": np.nan,
                "height": np.nan,
                "bounds": "",
                "reason": f"raster open/evaluation failed: {exc}",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        idx = choose_tmc_index(df)
        if idx is not None:
            df.loc[idx, "selected_for_terrain"] = True
    return df


def read_tmc_dtm_aoi(root: Path, aoi: dict[str, float], target_shape: tuple[int, int]) -> dict[str, Any]:
    coverage = evaluate_tmc_dtm_coverage(root, aoi)
    if coverage.empty or not coverage["selected_for_terrain"].astype(bool).any():
        return {"available": False, "coverage": coverage, "reason": "No TMC-2 DTM intersects the configured AOI."}
    row = coverage[coverage["selected_for_terrain"].astype(bool)].iloc[0]
    path = Path(row["path"])
    with rasterio.open(path) as ds:
        window, _ = clipped_window_for_aoi(ds, aoi, pad_pixels=1)
        elev_native = ds.read(1, window=window).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            elev_native = np.where(np.isclose(elev_native, nodata), np.nan, elev_native).astype("float32")
        elev_native = np.where(np.isfinite(elev_native), elev_native, np.nan).astype("float32")
        pixel_size = float(abs(ds.transform.a))
        slope_native = compute_slope_deg(elev_native, pixel_size)
        elevation = resize_to(elev_native, target_shape)
        slope = resize_to(slope_native, target_shape)
    return {
        "available": True,
        "coverage": coverage,
        "selected_product_id": row["product_id"],
        "selected_path": str(path),
        "coverage_fraction": float(row["coverage_fraction"]),
        "coverage_class": row["coverage_class"],
        "pixel_size_m": float(row["pixel_size_m"]),
        "elevation": elevation,
        "slope_deg": slope,
        "elevation_native": elev_native,
        "slope_native": slope_native,
        "source": "TMC-2 DTM",
        "reason": "Selected highest-coverage TMC-2 DTM for terrain/slope analysis.",
    }


def choose_tmc_index(df):
    usable = df[df["coverage_fraction"].astype(float) > 0].copy()
    if usable.empty:
        return None
    usable["_full_rank"] = np.where(usable["coverage_class"].eq("FULL"), 0, 1)
    usable["_pixels"] = usable["width"].astype(float) * usable["height"].astype(float)
    usable = usable.sort_values(["_full_rank", "pixel_size_m", "_pixels", "product_id"], ascending=[True, True, True, True])
    return usable.index[0]


def product_id_for_tmc(path: Path) -> str:
    for part in path.parts:
        if part.startswith("ch2_tmc"):
            return part
    return path.stem


def coverage_class_from_fraction(frac: float) -> str:
    if frac >= 0.999:
        return "FULL"
    if frac > 0:
        return "PARTIAL"
    return "NO COVERAGE"


def coverage_reason(coverage_class: str) -> str:
    if coverage_class == "FULL":
        return "covers the complete configured rectangular AOI"
    if coverage_class == "PARTIAL":
        return "intersects the configured AOI but does not cover it completely"
    return "does not intersect the configured AOI"


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
