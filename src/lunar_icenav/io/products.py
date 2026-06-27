from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import rasterio


RASTER_EXTS = {".tif", ".tiff", ".img"}
DOC_EXTS = {".xml", ".csv", ".json", ".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SarPair:
    product_id: str
    band: str
    lh_path: Path
    lv_path: Path
    xml_path: Path | None
    pixel_size_m: float
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    crs_wkt: str
    coverage_fraction: float


def product_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("ch2_"):
            return part
    return path.stem


def classify_file(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    sensor = "document"
    role = "metadata"
    if "sar" in name or any("ch2_sar" in p.lower() for p in path.parts):
        sensor = "SAR/DFSAR"
    elif "ohr" in name or any("ch2_ohr" in p.lower() for p in path.parts):
        sensor = "OHRC"
    elif "ldem" in name or "ldsm" in name or "dem" in str(path).lower():
        sensor = "DEM/topography"

    if suffix in {".tif", ".tiff"}:
        role = "science raster"
    elif suffix == ".img":
        role = "science image"
    elif suffix == ".zip":
        role = "compressed product bundle"
    elif suffix in IMAGE_EXTS:
        role = "browse/quicklook"
    elif suffix == ".csv":
        role = "geometry/table"
    elif suffix == ".xml":
        role = "PDS/XML metadata"
    elif suffix == ".md":
        role = "project document"
    return sensor, role


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.strip())
    except Exception:
        return None


def xml_value(root: ET.Element, suffix: str) -> str | None:
    for elem in root.iter():
        if elem.tag.lower().endswith(suffix.lower()):
            return (elem.text or "").strip()
    return None


def xml_float(root: ET.Element, suffix: str) -> float | None:
    return safe_float(xml_value(root, suffix))


def parse_xml_metadata(xml_path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        return {"xml_parse_error": str(exc)}

    keys = [
        "centre_latitude",
        "centre_longitude",
        "upper_left_latitude",
        "upper_left_longitude",
        "upper_right_latitude",
        "upper_right_longitude",
        "lower_left_latitude",
        "lower_left_longitude",
        "lower_right_latitude",
        "lower_right_longitude",
        "pixel_resolution",
        "output_pixel_spacing",
        "output_line_spacing",
        "frequency_band",
        "imaging_mode",
        "incidence_angle",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        val = xml_float(root, key)
        out[key] = val if val is not None else xml_value(root, key)
    return out


def sample_raster_metadata(path: Path, sample_pixels: int = 256) -> dict[str, Any]:
    try:
        with rasterio.open(path) as ds:
            stats: dict[str, Any] = {
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
                "dtype": ",".join(ds.dtypes),
                "crs": str(ds.crs) if ds.crs else "",
                "pixel_size_x": float(ds.transform.a),
                "pixel_size_y": float(abs(ds.transform.e)),
                "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                "nodata": ds.nodata,
            }
            h = min(sample_pixels, ds.height)
            w = min(sample_pixels, ds.width)
            window = rasterio.windows.Window(0, 0, w, h)
            arr = ds.read(1, window=window, masked=True).astype("float32")
            finite = np.asarray(arr.compressed())
            if finite.size:
                p = np.nanpercentile(finite, [2, 50, 98])
                stats.update({"p02": float(p[0]), "p50": float(p[1]), "p98": float(p[2])})
            return stats
    except Exception as exc:
        return {"raster_open_error": str(exc)}


def inspect_ohrc_zip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            xmls = [n for n in names if n.endswith(".xml")]
            browses = [n for n in names if n.lower().endswith(".png")]
            imgs = [n for n in names if n.lower().endswith(".img")]
            row: dict[str, Any] = {
                "path": str(path),
                "product_id": path.stem,
                "product_type": "OHRC zip bundle",
                "role": "optical / hazard interpretation",
                "resolution": "",
                "usable": "yes - browse/context; IMG requires extraction/PDS handling",
                "notes": f"{len(imgs)} IMG, {len(xmls)} XML, {len(browses)} browse PNG inside zip",
            }
            data_xml = next((n for n in xmls if "_d_img_" in n), None)
            if data_xml:
                root = ET.fromstring(zf.read(data_xml))
                meta = {k: xml_float(root, k) for k in [
                    "pixel_resolution",
                    "upper_left_latitude",
                    "upper_left_longitude",
                    "upper_right_latitude",
                    "upper_right_longitude",
                    "lower_left_latitude",
                    "lower_left_longitude",
                    "lower_right_latitude",
                    "lower_right_longitude",
                ]}
                if meta.get("pixel_resolution") is not None:
                    row["resolution"] = f'{meta["pixel_resolution"]} m/pixel'
                row.update(meta)
            rows.append(row)
    except Exception as exc:
        rows.append({
            "path": str(path),
            "product_id": path.stem,
            "product_type": "OHRC zip bundle",
            "role": "optical / hazard interpretation",
            "resolution": "",
            "usable": "no - zip read failed",
            "notes": str(exc),
        })
    return rows


def discover_products(root: Path, aoi_summary: str = "not checked") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skip_dirs = {"outputs", "reports", "notebooks", "src", "__pycache__", "tmp", "models"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        if len(rel_parts) >= 2 and rel_parts[0] == "data" and rel_parts[1] == "metadata":
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        suffix = path.suffix.lower()
        sensor, role = classify_file(path)
        if suffix == ".zip" and sensor == "OHRC":
            rows.extend(inspect_ohrc_zip(path))
            continue

        row: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "product_id": product_id_from_path(path),
            "product_type": sensor,
            "role": role,
            "resolution": "",
            "usable": "yes",
            "notes": aoi_summary,
        }
        if suffix in {".tif", ".tiff"}:
            meta = sample_raster_metadata(path)
            if "raster_open_error" in meta:
                row["usable"] = "needs review"
                row["notes"] = meta["raster_open_error"]
            else:
                row["resolution"] = f'{meta.get("pixel_size_x", "")} x {meta.get("pixel_size_y", "")} m/pixel'
                row["width"] = meta.get("width")
                row["height"] = meta.get("height")
                row["crs"] = "known" if meta.get("crs") else "missing"
                row["sample_p02"] = meta.get("p02")
                row["sample_p50"] = meta.get("p50")
                row["sample_p98"] = meta.get("p98")
                if sensor == "SAR/DFSAR" and "_d_sri_" in path.name.lower():
                    row["role"] = "radar candidate detection"
                if sensor == "DEM/topography":
                    row["role"] = "topography / slope"
        elif suffix == ".xml":
            meta = parse_xml_metadata(path)
            if meta.get("output_pixel_spacing"):
                row["resolution"] = f'{meta["output_pixel_spacing"]} m/pixel'
            if meta.get("centre_latitude") is not None:
                row["notes"] = f'center lat/lon {meta.get("centre_latitude")}, {meta.get("centre_longitude")}'
        elif suffix == ".png":
            row["usable"] = "yes - browse only"
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_sar_pairs(root: Path) -> list[dict[str, Path]]:
    lh_files = sorted(root.rglob("*_d_sri_xx_cp_lh_d18.tif"))
    pairs: list[dict[str, Path]] = []
    for lh in lh_files:
        lv = Path(str(lh).replace("_cp_lh_", "_cp_lv_"))
        xml = Path(str(lh).replace("_cp_lh_d18.tif", "_cp_xx_d18.xml"))
        ma = Path(str(lh).replace("_xx_cp_lh_", "_ma_cp_xx_"))
        inc = Path(str(lh).replace("_xx_cp_lh_", "_in_cp_xx_"))
        if lv.exists():
            pairs.append({"lh": lh, "lv": lv, "xml": xml if xml.exists() else None, "mask": ma if ma.exists() else None, "incidence": inc if inc.exists() else None})
    return pairs


def save_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)
