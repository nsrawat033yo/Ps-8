from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "configs/pipeline.json") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    paths = {
        "figures": Path(outputs.get("figures", "outputs/figures")),
        "tables": Path(outputs.get("tables", "outputs/tables")),
        "masks": Path(outputs.get("masks", "outputs/masks")),
        "rasters": Path(outputs.get("rasters", "outputs/rasters")),
        "routes": Path(outputs.get("routes", "outputs/routes")),
        "reports": Path(outputs.get("reports", "reports")),
        "models": Path(outputs.get("models", "models")),
    }
    paths["metadata"] = Path("data/metadata")
    paths["notebooks"] = Path("notebooks")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
