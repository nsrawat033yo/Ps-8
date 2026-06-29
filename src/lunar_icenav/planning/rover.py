from __future__ import annotations

import heapq
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from lunar_icenav.features.texture import robust_normalize
from lunar_icenav.preprocessing.aoi import map_to_lonlat, pixel_to_map


NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 2 ** 0.5),
    (-1, 1, 2 ** 0.5),
    (1, -1, 2 ** 0.5),
    (1, 1, 2 ** 0.5),
]


def choose_route_endpoints(sites: pd.DataFrame, candidate_mask: np.ndarray, candidate_score: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    if not sites.empty:
        start = (int(sites.iloc[0]["row"]), int(sites.iloc[0]["col"]))
    else:
        valid = np.isfinite(candidate_score)
        start = tuple(map(int, np.unravel_index(np.nanargmin(np.where(valid, candidate_score, np.inf)), candidate_score.shape)))
    if candidate_mask.any():
        labels = np.where(candidate_mask, candidate_score, -np.inf)
        target = tuple(map(int, np.unravel_index(np.argmax(labels), labels.shape)))
    else:
        target = tuple(map(int, np.unravel_index(np.nanargmax(candidate_score), candidate_score.shape)))
    return start, target


def build_cost_map(features: dict[str, np.ndarray], slope_deg: np.ndarray | None, mode: str, config: dict[str, Any]) -> np.ndarray:
    valid = features["valid"].astype(bool)
    texture = robust_normalize(features["texture"], valid)
    if slope_deg is None:
        slope_norm = np.zeros_like(texture)
        blocked = ~valid
    else:
        soft_max = float(config.get("planning", {}).get("soft_max_slope_deg", 25.0))
        slope_norm = np.clip(slope_deg / soft_max, 0, 1)
        blocked = (~valid) | (~np.isfinite(slope_deg)) | (slope_deg > soft_max)

    if mode == "shortest":
        cost = 1.0 + 0.25 * slope_norm + 0.15 * texture
    elif mode == "safest":
        cost = 1.0 + 2.7 * slope_norm + 2.2 * texture
    elif mode == "energy_efficient":
        cost = 1.0 + 1.9 * slope_norm + 1.5 * texture
    else:
        science = robust_normalize(features["candidate_score"], valid)
        cost = 1.0 + 1.3 * slope_norm + 1.0 * texture + 0.45 * (1 - science)
    cost = cost.astype("float32")
    cost[blocked] = np.inf
    return cost


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_expansions: int = 750000) -> tuple[list[tuple[int, int]], float, str]:
    rows, cols = cost.shape
    if not in_bounds(start, rows, cols) or not in_bounds(goal, rows, cols):
        return [], np.inf, "start or goal outside raster"
    if not np.isfinite(cost[start]):
        return [], np.inf, "start pixel is blocked"
    if not np.isfinite(cost[goal]):
        goal = nearest_unblocked(cost, goal)
        if goal is None:
            return [], np.inf, "goal pixel and neighbors are blocked"

    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g = {start: 0.0}
    expansions = 0
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct(came_from, current), float(g[current]), "ok"
        expansions += 1
        if expansions > max_expansions:
            return [], np.inf, "max expansions reached"
        for dr, dc, step in NEIGHBORS:
            nr, nc = current[0] + dr, current[1] + dc
            if not in_bounds((nr, nc), rows, cols) or not np.isfinite(cost[nr, nc]):
                continue
            tentative = g[current] + 0.5 * (cost[current] + cost[nr, nc]) * step
            nxt = (nr, nc)
            if tentative < g.get(nxt, np.inf):
                came_from[nxt] = current
                g[nxt] = tentative
                priority = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))
    return [], np.inf, "no path found"


def nearest_unblocked(cost: np.ndarray, goal: tuple[int, int], radius: int = 20) -> tuple[int, int] | None:
    gr, gc = goal
    candidates: list[tuple[float, tuple[int, int]]] = []
    for r in range(max(0, gr - radius), min(cost.shape[0], gr + radius + 1)):
        for c in range(max(0, gc - radius), min(cost.shape[1], gc + radius + 1)):
            if np.isfinite(cost[r, c]):
                candidates.append(((r - gr) ** 2 + (c - gc) ** 2, (r, c)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def in_bounds(p: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= p[0] < rows and 0 <= p[1] < cols


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def reconstruct(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plan_routes(
    features: dict[str, np.ndarray],
    slope_deg: np.ndarray | None,
    sites: pd.DataFrame,
    candidate_mask: np.ndarray,
    transform,
    crs_wkt: str,
    config: dict[str, Any],
    target_candidate_id: str | None = None,
) -> tuple[dict[str, list[tuple[int, int]]], pd.DataFrame]:
    start, goal = choose_route_endpoints(sites, candidate_mask, features["candidate_score"])
    candidate_labels, _ = ndi.label(candidate_mask)
    if target_candidate_id is None:
        target_candidate_id = f"C-{int(candidate_labels[goal]):03d}" if candidate_labels[goal] > 0 else "nearest_candidate_pixel"
    routes: dict[str, list[tuple[int, int]]] = {}
    rows: list[dict[str, Any]] = []
    pixel_size = abs(float(transform.a))
    start_landing_site_id = str(sites.iloc[0]["site_id"]) if not sites.empty and "site_id" in sites else "not_available"
    for mode in ["shortest", "safest", "science_priority", "energy_efficient"]:
        cost = build_cost_map(features, slope_deg, mode, config)
        path, route_cost, status = astar(cost, start, goal)
        routes[mode] = path
        length_m = path_length(path, pixel_size)
        route_slope = path_values(path, slope_deg) if slope_deg is not None else np.array([], dtype=float)
        rows.append({
            "route_type": mode,
            "status": status,
            "target_candidate_id": target_candidate_id,
            "start_landing_site_id": start_landing_site_id,
            "start_row": start[0],
            "start_col": start[1],
            "target_row": goal[0],
            "target_col": goal[1],
            "steps": len(path),
            "length_m": length_m,
            "cost": route_cost,
            "total_cost": route_cost,
            "mean_slope_deg": float(np.nanmean(route_slope)) if route_slope.size else np.nan,
            "max_slope_deg": float(np.nanmax(route_slope)) if route_slope.size else np.nan,
            "interpretation": "conceptual rover traverse on proxy traversability cost map",
        })
    route_df = pd.DataFrame(rows)
    if routes:
        for mode, path in routes.items():
            if path:
                rr = np.array([p[0] for p in path], dtype=float)
                cc = np.array([p[1] for p in path], dtype=float)
                xs, ys = [], []
                for r, c in zip(rr, cc):
                    x, y = pixel_to_map(transform, r, c)
                    xs.append(x)
                    ys.append(y)
                lon, lat = map_to_lonlat(crs_wkt, np.array(xs), np.array(ys))
                route_df.loc[route_df["route_type"] == mode, "start_lat"] = float(lat[0])
                route_df.loc[route_df["route_type"] == mode, "start_lon"] = float(lon[0])
                route_df.loc[route_df["route_type"] == mode, "target_lat"] = float(lat[-1])
                route_df.loc[route_df["route_type"] == mode, "target_lon"] = float(lon[-1])
    return routes, route_df


def path_length(path: list[tuple[int, int]], pixel_size: float) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) * pixel_size
    return float(total)


def path_values(path: list[tuple[int, int]], arr: np.ndarray) -> np.ndarray:
    if not path:
        return np.array([], dtype=float)
    return np.array([arr[r, c] for r, c in path], dtype=float)


def route_points_df(path: list[tuple[int, int]], transform, crs_wkt: str, route_type: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (r, c) in enumerate(path):
        x, y = pixel_to_map(transform, float(r), float(c))
        lon, lat = map_to_lonlat(crs_wkt, np.array([x]), np.array([y]))
        rows.append({"route_type": route_type, "seq": i, "row": r, "col": c, "x_m": x, "y_m": y, "lat": float(lat[0]), "lon": float(lon[0])})
    return pd.DataFrame(rows)
