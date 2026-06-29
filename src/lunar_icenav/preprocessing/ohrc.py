from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

from lunar_icenav.features.texture import gradient_magnitude, local_std, robust_normalize


def _xml_float(root: ET.Element, suffix: str) -> float | None:
    for elem in root.iter():
        if elem.tag.lower().endswith(suffix.lower()):
            try:
                return float((elem.text or "").strip())
            except Exception:
                return None
    return None


def inspect_ohrc_footprints(root: Path, aoi: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("ch2_ohr*.zip")):
        with zipfile.ZipFile(path) as zf:
            data_xml = next((n for n in zf.namelist() if n.endswith("_d_img_d18.xml")), None)
            if not data_xml:
                continue
            rt = ET.fromstring(zf.read(data_xml))
            lats = [
                _xml_float(rt, "upper_left_latitude"),
                _xml_float(rt, "upper_right_latitude"),
                _xml_float(rt, "lower_left_latitude"),
                _xml_float(rt, "lower_right_latitude"),
            ]
            lons = [
                _xml_float(rt, "upper_left_longitude"),
                _xml_float(rt, "upper_right_longitude"),
                _xml_float(rt, "lower_left_longitude"),
                _xml_float(rt, "lower_right_longitude"),
            ]
            lats = [x for x in lats if x is not None]
            lons = [x for x in lons if x is not None]
            lat_overlap = bool(lats and max(lats) >= aoi["lat_min"] and min(lats) <= aoi["lat_max"])
            lon_overlap_simple = bool(lons and max(lons) >= aoi["lon_min"] and min(lons) <= aoi["lon_max"])
            coverage_fraction = approximate_lonlat_overlap_fraction(
                min(lats) if lats else np.nan,
                max(lats) if lats else np.nan,
                min(lons) if lons else np.nan,
                max(lons) if lons else np.nan,
                aoi,
            )
            coverage_class = "FULL" if coverage_fraction >= 0.999 else "PARTIAL" if coverage_fraction > 0 else "NO COVERAGE"
            rows.append({
                "product_id": path.stem,
                "path": str(path),
                "pixel_resolution_m": _xml_float(rt, "pixel_resolution"),
                "lat_min": min(lats) if lats else np.nan,
                "lat_max": max(lats) if lats else np.nan,
                "lon_min_raw": min(lons) if lons else np.nan,
                "lon_max_raw": max(lons) if lons else np.nan,
                "coverage_fraction": coverage_fraction,
                "coverage_class": coverage_class,
                "usable_for_analysis": coverage_class in {"FULL", "PARTIAL"},
                "overlaps_faustini_lat": lat_overlap,
                "simple_lon_overlap": lon_overlap_simple,
                "coverage_note": "not co-registered to Faustini AOI" if not lat_overlap else "possible latitude overlap; requires geometry CSV check",
            })
    return pd.DataFrame(rows)


def approximate_lonlat_overlap_fraction(lat_min: float, lat_max: float, lon_min: float, lon_max: float, aoi: dict[str, float]) -> float:
    if not all(np.isfinite(v) for v in [lat_min, lat_max, lon_min, lon_max]):
        return 0.0
    lat_overlap = max(0.0, min(lat_max, aoi["lat_max"]) - max(lat_min, aoi["lat_min"]))
    lat_span = max(aoi["lat_max"] - aoi["lat_min"], 1e-9)
    lon_overlap = max(0.0, min(lon_max, aoi["lon_max"]) - max(lon_min, aoi["lon_min"]))
    lon_span = max(aoi["lon_max"] - aoi["lon_min"], 1e-9)
    return float(np.clip((lat_overlap / lat_span) * (lon_overlap / lon_span), 0.0, 1.0))


def load_first_ohrc_browse(root: Path) -> tuple[np.ndarray, str] | tuple[None, str]:
    for path in sorted(root.rglob("ch2_ohr*.zip")):
        with zipfile.ZipFile(path) as zf:
            browse = next((n for n in zf.namelist() if n.lower().endswith("_b_brw_d18.png")), None)
            if browse:
                img = Image.open(BytesIO(zf.read(browse))).convert("L")
                arr = np.asarray(img).astype("float32") / 255.0
                return arr, f"{path.stem}:{browse}"
    return None, "No OHRC browse PNG found."


def ohrc_hazard_proxy(gray: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(gray)
    tex = local_std(gray, size=11, valid=valid)
    grad = gradient_magnitude(gray, valid=valid)
    score = 0.55 * robust_normalize(tex, valid) + 0.45 * robust_normalize(grad, valid)
    hazard = score >= np.nanquantile(score[valid], 0.90)
    return {"gray": gray, "texture": tex, "gradient": grad, "hazard_score": score, "hazard_mask": hazard}
