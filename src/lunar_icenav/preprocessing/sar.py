from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from lunar_icenav.io.products import find_sar_pairs
from lunar_icenav.preprocessing.aoi import choose_best_sar_pair, clipped_window_for_aoi


def select_sar_pair(root: Path, aoi: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = find_sar_pairs(root)
    if not pairs:
        raise RuntimeError("No calibrated SRI LH/LV SAR pairs were found.")
    return choose_best_sar_pair(pairs, aoi)


def read_sar_aoi(pair: dict[str, Any], aoi: dict[str, float]) -> dict[str, Any]:
    with rasterio.open(pair["lh"]) as lh_ds, rasterio.open(pair["lv"]) as lv_ds:
        window, aoi_bounds = clipped_window_for_aoi(lh_ds, aoi, pad_pixels=2)
        lh = lh_ds.read(1, window=window)
        lv = lv_ds.read(1, window=window)
        transform = lh_ds.window_transform(window)
        profile = lh_ds.profile.copy()
        profile.update({
            "height": int(window.height),
            "width": int(window.width),
            "transform": transform,
            "count": 1,
        })
        return {
            "lh": lh,
            "lv": lv,
            "window": {
                "row_off": int(window.row_off),
                "col_off": int(window.col_off),
                "height": int(window.height),
                "width": int(window.width),
            },
            "aoi_projected_bounds": aoi_bounds,
            "transform": transform,
            "profile": profile,
            "crs_wkt": lh_ds.crs.to_wkt(),
            "product_id": pair["product_id"],
            "lh_path": str(pair["lh"]),
            "lv_path": str(pair["lv"]),
        }


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, mask.astype("uint8") * 255, cmap="gray")
