from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import ndimage as ndi

from lunar_icenav.config import ensure_output_dirs, load_config
from lunar_icenav.features.polarimetry import sar_feature_stack
from lunar_icenav.features.texture import robust_normalize
from lunar_icenav.io.products import discover_products, save_run_manifest
from lunar_icenav.mapping.anomaly import candidate_mask_from_features, summarize_patches
from lunar_icenav.models.training import run_unet_prototype
from lunar_icenav.planning.landing import suitability_map
from lunar_icenav.planning.rover import plan_routes, route_points_df
from lunar_icenav.preprocessing.dem import read_dem_aoi
from lunar_icenav.preprocessing.ohrc import inspect_ohrc_footprints, load_first_ohrc_browse, ohrc_hazard_proxy
from lunar_icenav.preprocessing.sar import read_sar_aoi, save_mask_png, select_sar_pair
from lunar_icenav.viz.plots import (
    save_candidate_centroid_map,
    save_coverage_map,
    save_feature_panel,
    save_histogram,
    save_inventory_chart,
    save_landing_score_components,
    save_ohrc_hazard_overlay,
    save_overlay,
    save_route_comparison,
    save_scalar_map,
    save_score_distribution,
    save_slope_classification,
    save_slope_map,
    save_top_candidates_chart,
    save_unet_training_curve,
    save_workflow_diagram,
)


SAFE_NOTE = "Preliminary candidate decision-support output; validation required."


def run_pipeline(root: Path, config_path: Path = Path("configs/pipeline.json")) -> dict[str, Any]:
    root = Path(root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    feature_dir = Path("outputs/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    aoi = config["aoi"]

    inventory = discover_products(root)
    inventory.to_csv(paths["tables"] / "product_inventory.csv", index=False)
    inventory.to_csv(paths["metadata"] / "product_inventory.csv", index=False)
    save_inventory_chart(inventory, paths["figures"] / "01_dataset_inventory_chart.png")

    pair, sar_scores = select_sar_pair(root, aoi)
    sar_score_df = pd.DataFrame([{k: str(v) if isinstance(v, Path) else v for k, v in s.items()} for s in sar_scores])
    sar_score_df.to_csv(paths["tables"] / "sar_aoi_overlap.csv", index=False)
    save_coverage_map(sar_score_df, paths["figures"] / "02_selected_sar_coverage.png", pair["product_id"])

    sar = read_sar_aoi(pair, aoi)
    selected_metadata = selected_sar_metadata(pair, sar)
    pd.DataFrame([selected_metadata]).to_csv(paths["tables"] / "selected_sar_metadata.csv", index=False)

    features = sar_feature_stack(sar["lh"], sar["lv"])
    base = features["intensity"]
    np.savez_compressed(
        feature_dir / "sar_feature_stack.npz",
        lh=features["lh"],
        lv=features["lv"],
        valid=features["valid"],
        intensity=features["intensity"],
        cpr_style_ratio_proxy=features["cpr_style_ratio_proxy"],
        lv_lh_ratio_proxy=features["lv_lh_ratio_proxy"],
        polarization_imbalance_proxy=features["polarization_imbalance_proxy"],
        local_mean=features["local_mean"],
        local_std=features["local_std"],
        texture=features["texture"],
        candidate_score=features["candidate_score"],
    )

    candidate_mask, candidate_score, thresholds = candidate_mask_from_features(features, config)
    threshold_sensitivity, stability_map = build_threshold_sensitivity(features, config, sar["transform"])
    threshold_sensitivity.to_csv(paths["tables"] / "threshold_sensitivity.csv", index=False)
    patches = summarize_patches(candidate_mask, features, sar["transform"], sar["crs_wkt"], sar["product_id"])
    save_mask_png(candidate_mask, paths["masks"] / "candidate_mask.png")
    np.save(paths["masks"] / "candidate_mask.npy", candidate_mask)
    np.save(paths["masks"] / "candidate_stability.npy", stability_map)

    dem = read_dem_aoi(root, aoi, target_shape=candidate_mask.shape)
    elevation = dem.get("elevation")
    slope = dem.get("slope_deg")
    uncertainty = candidate_uncertainty(features["candidate_score"], thresholds["score_threshold"], features["valid"])
    patches = enrich_candidate_patches_with_slope(patches, candidate_mask, slope)
    patches = enrich_candidate_patches_with_scientific_metrics(patches, candidate_mask, features, stability_map, uncertainty)
    patches.to_csv(paths["tables"] / "candidate_patches.csv", index=False)
    if elevation is not None:
        np.save(feature_dir / "dem_elevation_resampled.npy", elevation)
    if slope is not None:
        np.save(feature_dir / "dem_slope_resampled.npy", slope)

    landing_score, landing_sites, landing_layers = suitability_map(candidate_mask, features, slope, sar["transform"], sar["crs_wkt"], config)
    landing_sites = add_landing_context_columns(landing_sites)
    landing_sites = attach_nearest_candidate_context(landing_sites, patches)
    landing_sites.to_csv(paths["tables"] / "landing_sites.csv", index=False)
    np.save(paths["masks"] / "landing_suitability.npy", landing_score)

    routes, route_summary = plan_routes(features, slope, landing_sites, candidate_mask, sar["transform"], sar["crs_wkt"], config)
    rover_navigation = build_rover_navigation_evaluation(routes, route_summary, features, slope, sar["transform"])
    route_summary = rover_navigation.copy()
    landing_sites = attach_landing_route_accessibility(landing_sites, route_summary)
    landing_sites.to_csv(paths["tables"] / "landing_sites.csv", index=False)
    landing_site_evaluation = build_landing_site_evaluation(landing_sites)
    landing_site_evaluation.to_csv(paths["tables"] / "landing_site_evaluation.csv", index=False)
    route_summary.to_csv(paths["routes"] / "route_summary.csv", index=False)
    rover_navigation.to_csv(paths["tables"] / "rover_navigation_evaluation.csv", index=False)
    route_point_frames = []
    for mode, route_path in routes.items():
        df = route_points_df(route_path, sar["transform"], sar["crs_wkt"], mode) if route_path else pd.DataFrame()
        if not df.empty:
            df.to_csv(paths["routes"] / f"route_{mode}.csv", index=False)
            route_point_frames.append(df)
    if route_point_frames:
        pd.concat(route_point_frames, ignore_index=True).to_csv(paths["routes"] / "rover_routes.csv", index=False)

    candidate_scientific_review = build_candidate_scientific_review(patches, landing_sites, route_summary)
    candidate_scientific_review.to_csv(paths["tables"] / "candidate_scientific_review.csv", index=False)
    ice_candidate_patch_review = build_ice_candidate_patch_review(candidate_scientific_review)
    ice_candidate_patch_review.to_csv(paths["tables"] / "ice_candidate_patch_review.csv", index=False)
    resource_scenarios = build_resource_scenario_estimates(candidate_scientific_review)
    resource_scenarios.to_csv(paths["tables"] / "resource_scenario_estimates.csv", index=False)
    sar_product_selection_reason = build_sar_product_selection_reason(sar_score_df, pair, aoi)
    sar_product_selection_reason.to_csv(paths["tables"] / "sar_product_selection_reason.csv", index=False)

    ohrc_fp = inspect_ohrc_footprints(root, aoi)
    ohrc_fp.to_csv(paths["tables"] / "ohrc_footprints.csv", index=False)
    ohrc_arr, ohrc_source = load_first_ohrc_browse(root)
    if ohrc_arr is not None:
        ohrc = ohrc_hazard_proxy(ohrc_arr)
        save_ohrc_hazard_overlay(ohrc["gray"], ohrc["hazard_mask"], paths["figures"] / "13_ohrc_context_hazard_proxy.png", ohrc_source, "")
        shutil.copyfile(paths["figures"] / "13_ohrc_context_hazard_proxy.png", paths["figures"] / "ohrc_hazard_overlay.png")

    unet = run_unet_prototype(features, candidate_mask, config)
    np.save(paths["masks"] / "unet_prediction_probability.npy", unet["prediction_probability"])
    save_mask_png(unet["prediction_mask"], paths["masks"] / "unet_prediction_mask.png")
    unet_metrics = dict(unet["metrics"])
    unet_metrics["checkpoint_path"] = unet.get("checkpoint_path", "")
    pd.DataFrame([unet_metrics]).to_csv(paths["tables"] / "unet_pseudo_label_metrics.csv", index=False)
    unet["history"].to_csv(paths["tables"] / "unet_training_history.csv", index=False)
    unet["tile_inventory"].to_csv(paths["tables"] / "unet_tile_inventory.csv", index=False)
    model_experiments = build_model_experiment_comparison(unet, features, candidate_mask)
    model_experiments.to_csv(paths["tables"] / "model_experiment_comparison.csv", index=False)

    candidate_summary = build_candidate_summary(candidate_mask, patches, features, thresholds, slope)
    candidate_summary.to_csv(paths["tables"] / "candidate_patch_summary.csv", index=False)
    slope_stats = build_slope_stats(slope, features["valid"]) if slope is not None else {}
    pd.DataFrame([slope_stats]).to_csv(paths["tables"] / "slope_safety_summary.csv", index=False)
    data_status = build_data_coverage_status(inventory, sar_score_df, ohrc_fp, pair, aoi)
    data_status.to_csv(paths["tables"] / "data_coverage_status.csv", index=False)

    hazard_mask = robust_normalize(features["texture"], features["valid"]) > np.nanquantile(robust_normalize(features["texture"], features["valid"])[features["valid"]], 0.90)
    unsafe_slope_mask = (features["valid"].astype(bool) & (slope > 10)) if slope is not None else np.zeros_like(candidate_mask, dtype=bool)
    landing_top_mask = landing_score >= np.nanquantile(landing_score[np.isfinite(landing_score)], 0.95)

    save_region_validation_map(sar_score_df, pair, aoi, paths["figures"] / "region_validation_map.png")
    save_scalar_map(base, paths["figures"] / "03_sar_quicklook.png", f"SAR SRI intensity quicklook - {sar['product_id']}", "gray", "log intensity proxy", SAFE_NOTE)
    save_feature_panel(base, features, candidate_mask, paths["figures"] / "04_sar_feature_panel.png", f"SAR feature panel - {sar['product_id']} - {aoi['name']}")
    save_ice_candidate_detection_map(base, candidate_mask, patches, paths["figures"] / "ice_candidate_detection_map.png")
    save_candidate_confidence_map(base, candidate_mask, patches, paths["figures"] / "ice_candidate_confidence_map.png")
    save_candidate_confidence_map(base, candidate_mask, patches, paths["figures"] / "top_candidate_confidence_map.png")
    save_threshold_sensitivity_curve(threshold_sensitivity, paths["figures"] / "threshold_sensitivity_curve.png")
    save_candidate_ranking_chart(candidate_scientific_review, paths["figures"] / "ice_candidate_ranking_chart.png")
    save_candidate_area_vs_score(candidate_scientific_review, paths["figures"] / "ice_candidate_area_vs_score.png")
    save_candidate_area_vs_score(candidate_scientific_review, paths["figures"] / "candidate_area_vs_score_scatter.png")
    save_candidate_diameter_distribution(candidate_scientific_review, paths["figures"] / "candidate_diameter_distribution.png")
    save_candidate_score_vs_slope(candidate_scientific_review, paths["figures"] / "candidate_score_vs_slope.png")
    save_resource_scenario_bar_chart(resource_scenarios, paths["figures"] / "resource_scenario_bar_chart.png")
    save_overlay(base, paths["figures"] / "05_radar_candidate_overlay.png", f"Radar candidate screening overlay - {sar['product_id']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.45))])
    save_histogram(patches, "area_m2", paths["figures"] / "06_candidate_patch_area_histogram.png", "Candidate Patch Area Distribution", "Area (m2)")
    save_top_candidates_chart(patches, paths["figures"] / "06b_top_candidate_patches.png")
    save_score_distribution(features, paths["figures"] / "06c_candidate_score_distribution.png")
    save_candidate_centroid_map(base, patches, paths["figures"] / "06d_candidate_centroid_map.png")
    save_scalar_map(uncertainty, paths["figures"] / "06e_candidate_uncertainty_map.png", "Candidate Screening Uncertainty Proxy", "cividis", "uncertainty proxy", "High values indicate pixels near the screening threshold; validation required.", vmin=0, vmax=1)
    if elevation is not None:
        save_scalar_map(elevation, paths["figures"] / "07_dem_elevation_map.png", f"DEM elevation context - {aoi['name']}", "terrain", "Elevation layer value", "DEM resampled to SAR AOI for context.")
    if slope is not None:
        save_slope_map(slope, paths["figures"] / "08_dem_slope_map.png", f"DEM slope map - {aoi['name']}")
        save_slope_classification(slope, paths["figures"] / "08b_slope_classification_map.png")
        pd.DataFrame([slope_stats]).to_csv(paths["tables"] / "slope_safety_summary.csv", index=False)
    save_scalar_map(landing_score, paths["figures"] / "09_landing_suitability_map.png", f"Preliminary landing suitability heatmap - {aoi['name']}", "YlGn", "Suitability score", "Preliminary landing candidates should be near candidate patches, but outside hazardous/steep terrain.", vmin=0, vmax=1)
    save_overlay(base, paths["figures"] / "10_top_landing_candidates_overlay.png", f"Top preliminary landing candidates - {aoi['name']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.32)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.28))], points=landing_sites)
    save_landing_score_components_evaluation(landing_site_evaluation, paths["figures"] / "landing_score_components.png")
    save_overlay(base, paths["figures"] / "landing_candidate_decision_map.png", f"Landing candidate decision map - {aoi['name']}", [(candidate_mask, "candidate ice region", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.26))], points=landing_sites, note="Preliminary landing candidates are selected near candidate patches but outside risky terrain; validation required.")
    save_nearest_landing_to_candidate_map(base, candidate_mask, patches, landing_sites, paths["figures"] / "nearest_landing_to_candidate_map.png")
    save_route_comparison(route_summary, paths["figures"] / "11_rover_route_comparison.png")
    save_overlay(base, paths["figures"] / "12_rover_route_overlay.png", f"Conceptual rover route variants - {aoi['name']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "blocked/risky proxy", (1.0, 0.20, 0.0, 0.26))], routes=routes, points=landing_sites.head(1))
    save_overlay(base, paths["figures"] / "rover_route_decision_map.png", f"Rover navigation decision map - {aoi['name']}", [(candidate_mask, "candidate ice region", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "blocked/risky proxy", (1.0, 0.20, 0.0, 0.26))], routes=routes, points=landing_sites.head(1), note="Conceptual rover routes use proxy traversability costs; validation required.")
    save_route_profile(routes, slope, paths["figures"] / "rover_route_slope_profile.png", "slope_deg", "Route Slope Profile", "Slope (deg)")
    save_route_risk_profile(routes, features, slope, paths["figures"] / "rover_route_risk_profile.png")
    save_unet_training_curve(unet["history"], paths["figures"] / "14_unet_training_curve.png")
    save_unet_training_curve(unet["history"], paths["figures"] / "unet_training_curve.png")
    save_model_experiment_comparison(model_experiments, paths["figures"] / "model_experiment_comparison.png")
    save_overlay(base, paths["figures"] / "15_unet_prediction_overlay.png", "Weakly supervised U-Net pseudo-label prediction overlay", [(unet["prediction_mask"], "U-Net pseudo-label prediction", (0.0, 0.45, 1.0, 0.42))], note=unet["note"])
    save_unet_error_map(candidate_mask, unet["prediction_mask"], paths["figures"] / "unet_error_map_against_pseudolabel.png")
    save_overlay(base, paths["figures"] / "16_combined_decision_map.png", f"Combined decision-support map - {aoi['name']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.40)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.28)), (landing_top_mask, "landing candidate", (0.0, 1.0, 0.28, 0.38))], routes=routes, points=landing_sites.head(3), note="Preliminary candidate decision-support map; validation required.")
    save_overlay(base, paths["figures"] / "combined_research_decision_map.png", f"Combined research decision map - {aoi['name']}", [(candidate_mask, "candidate ice region", (0.0, 0.85, 1.0, 0.40)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.28)), (landing_top_mask, "landing candidate", (0.0, 1.0, 0.28, 0.38))], routes=routes, points=landing_sites.head(3), note="Research decision-support output; validation required.")
    save_pseudo_label_distribution(candidate_mask, unet["prediction_mask"], paths["figures"] / "pseudo_label_distribution.png")
    save_landing_score_components(landing_sites, paths["figures"] / "landing_score_comparison.png")
    save_workflow_diagram(paths["figures"] / "workflow_diagram.png")
    copy_legacy_figure_names(paths["figures"])

    manifest = {
        "project": config.get("project_name"),
        "aoi": aoi,
        "selected_sar": {k: str(v) if isinstance(v, Path) else v for k, v in pair.items()},
        "sar_window": sar["window"],
        "thresholds": thresholds,
        "feature_note": features["feature_note"],
        "unet_note": unet["note"],
        "unet_checkpoint": unet.get("checkpoint_path", ""),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    save_run_manifest(paths["reports"] / "run_manifest.json", manifest)
    shutil.copyfile(config_path, paths["reports"] / "pipeline_config_copy.json")
    write_summary(paths["reports"] / "LUNAQUEST_PROTOTYPE_SUMMARY.md", manifest, inventory, sar_score_df, patches, landing_sites, route_summary, ohrc_fp, unet, candidate_summary, slope_stats)
    write_ohrc_download_note(paths["reports"] / "OHRC_DATA_DOWNLOAD_NEEDED.md", aoi)
    write_improvement_summary(paths["reports"] / "NEXT_5_DAYS_IMPROVEMENT_SUMMARY.md", patches, landing_sites, route_summary, unet, slope_stats)
    write_model_evaluation_report(
        paths["reports"] / "MODEL_EVALUATION_REPORT.md",
        manifest,
        inventory,
        sar_score_df,
        threshold_sensitivity,
        candidate_summary,
        candidate_scientific_review,
        landing_site_evaluation,
        rover_navigation,
        unet,
        model_experiments,
        slope_stats,
        data_status,
        resource_scenarios,
    )
    write_technical_limitations(paths["reports"] / "TECHNICAL_LIMITATIONS.md", aoi)
    write_next_data_to_download(paths["reports"] / "NEXT_DATA_TO_DOWNLOAD.md", aoi)
    write_scientific_limitation_checklist(paths["reports"] / "SCIENTIFIC_LIMITATION_CHECKLIST.md")
    create_notebook(paths["notebooks"] / "LunaQuest_BAH2026_Workflow.ipynb")
    return manifest


def selected_sar_metadata(pair: dict[str, Any], sar: dict[str, Any]) -> dict[str, Any]:
    profile = sar["profile"]
    return {
        "product_id": sar["product_id"],
        "lh_path": sar["lh_path"],
        "lv_path": sar["lv_path"],
        "width": profile.get("width"),
        "height": profile.get("height"),
        "dtype": str(profile.get("dtype")),
        "crs": str(profile.get("crs")),
        "pixel_size_m": abs(float(profile["transform"].a)),
        "window_row_off": sar["window"]["row_off"],
        "window_col_off": sar["window"]["col_off"],
        "window_height": sar["window"]["height"],
        "window_width": sar["window"]["width"],
        "coverage_fraction": pair.get("coverage_fraction"),
        "role": "selected calibrated SRI LH/LV intensity pair for radar candidate screening proxy workflow",
    }


def add_landing_context_columns(landing_sites: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    out["illumination_score_status"] = "placeholder proxy from SAR intensity; replace with illumination model"
    out["thermal_score_status"] = "future layer needed"
    out["communication_score_status"] = "future line-of-sight layer needed"
    out["safe_wording"] = "preliminary landing candidate; not mission-certified"
    return out


def enrich_candidate_patches_with_slope(patches: pd.DataFrame, candidate_mask: np.ndarray, slope: np.ndarray | None) -> pd.DataFrame:
    if patches.empty or slope is None:
        return patches
    labels, _ = ndi.label(candidate_mask)
    out = patches.copy()
    mean_slopes: list[float] = []
    max_slopes: list[float] = []
    slope_conditions: list[str] = []
    for candidate_id in out["candidate_id"]:
        try:
            label = int(str(candidate_id).split("-")[-1])
        except ValueError:
            label = -1
        mask = labels == label
        vals = slope[mask & np.isfinite(slope)]
        if vals.size:
            mean_slope = float(np.nanmean(vals))
            max_slope = float(np.nanmax(vals))
        else:
            mean_slope = np.nan
            max_slope = np.nan
        mean_slopes.append(mean_slope)
        max_slopes.append(max_slope)
        slope_conditions.append(describe_slope_condition(mean_slope, max_slope))
    out["mean_slope_deg"] = mean_slopes
    out["max_slope_deg"] = max_slopes
    out["slope_condition"] = slope_conditions
    return out


def describe_slope_condition(mean_slope: float, max_slope: float) -> str:
    if not np.isfinite(mean_slope):
        return "slope unavailable"
    if mean_slope < 5 and max_slope <= 10:
        return "low mean slope; favorable screening context"
    if mean_slope < 5 and max_slope > 15:
        return "low mean slope but includes steep pixels; review boundary"
    if mean_slope <= 10:
        return "moderate slope context; review needed"
    return "steep local context; lower landing/traverse priority"


def build_threshold_sensitivity(features: dict[str, np.ndarray], config: dict[str, Any], transform) -> tuple[pd.DataFrame, np.ndarray]:
    valid = features["valid"].astype(bool)
    ratio = features["cpr_style_ratio_proxy"]
    intensity = features["intensity"]
    texture = features["texture"]
    score = features["candidate_score"]
    screen_cfg = config.get("candidate_screen", {})
    ratio_thr = float(np.nanquantile(ratio[valid], float(screen_cfg.get("ratio_quantile", 0.88))))
    intensity_thr = float(np.nanquantile(intensity[valid], float(screen_cfg.get("intensity_quantile", 0.70))))
    texture_thr = float(np.nanquantile(texture[valid], float(screen_cfg.get("max_texture_quantile", 0.98))))
    min_pixels = int(screen_cfg.get("min_patch_pixels", 8))
    quantiles = np.array([0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90], dtype=float)
    pixel_area_m2 = abs(float(transform.a * transform.e))
    stability_count = np.zeros(score.shape, dtype="float32")
    rows: list[dict[str, Any]] = []
    valid_pixels = max(int(valid.sum()), 1)
    for q in quantiles:
        score_thr = float(np.nanquantile(score[valid], q))
        mask = valid & (ratio >= ratio_thr) & (intensity >= intensity_thr) & (texture <= texture_thr) & (score >= score_thr)
        mask = remove_small_candidate_components(mask, min_pixels)
        labels, n = ndi.label(mask)
        stability_count += mask.astype("float32")
        rows.append({
            "score_quantile": float(q),
            "score_threshold": score_thr,
            "candidate_patch_count": int(n),
            "candidate_pixels": int(mask.sum()),
            "candidate_area_m2": float(mask.sum() * pixel_area_m2),
            "candidate_area_pct_of_valid_aoi": float(mask.sum() / valid_pixels * 100),
        })
    return pd.DataFrame(rows), (stability_count / max(len(quantiles), 1)).astype("float32")


def remove_small_candidate_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, n = ndi.label(mask)
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def enrich_candidate_patches_with_scientific_metrics(
    patches: pd.DataFrame,
    candidate_mask: np.ndarray,
    features: dict[str, np.ndarray],
    stability_map: np.ndarray,
    uncertainty: np.ndarray,
) -> pd.DataFrame:
    if patches.empty:
        return patches
    labels, _ = ndi.label(candidate_mask)
    out = patches.copy()
    max_scores: list[float] = []
    stabilities: list[float] = []
    uncertainties: list[float] = []
    confidence_scores: list[float] = []
    confidence_levels: list[str] = []
    area = out["area_m2"].astype(float)
    area_norm = ((area - area.min()) / max(float(area.max() - area.min()), 1e-6)).fillna(0.0).to_numpy()
    for i, row in out.iterrows():
        label = candidate_label_from_id(row["candidate_id"])
        mask = labels == label
        scores = features["candidate_score"][mask & np.isfinite(features["candidate_score"])]
        max_score = float(np.nanmax(scores)) if scores.size else np.nan
        stability = float(np.nanmean(stability_map[mask])) if mask.any() else 0.0
        mean_uncertainty = float(np.nanmean(uncertainty[mask])) if mask.any() else 1.0
        mean_slope = float(row.get("mean_slope_deg", np.nan))
        max_slope = float(row.get("max_slope_deg", np.nan))
        if np.isfinite(mean_slope) and np.isfinite(max_slope):
            slope_context = 1.0 if mean_slope < 5 and max_slope <= 10 else 0.55 if mean_slope <= 10 else 0.05
        else:
            slope_context = 0.5
        confidence_score = (
            0.32 * float(row.get("mean_candidate_score", 0.0))
            + 0.20 * stability
            + 0.16 * float(area_norm[i])
            + 0.22 * slope_context
            + 0.10 * (1.0 - np.clip(mean_uncertainty, 0.0, 1.0))
        )
        confidence = "High" if confidence_score >= 0.78 else "Medium" if confidence_score >= 0.68 else "Low"
        max_scores.append(max_score)
        stabilities.append(stability)
        uncertainties.append(mean_uncertainty)
        confidence_scores.append(float(confidence_score))
        confidence_levels.append(confidence)
    out["area_km2"] = out["area_m2"].astype(float) / 1_000_000.0
    out["equivalent_candidate_patch_diameter_m"] = 2.0 * np.sqrt(out["area_m2"].astype(float) / np.pi)
    out["equivalent_diameter_m"] = out["equivalent_candidate_patch_diameter_m"]
    out["max_candidate_score"] = max_scores
    out["threshold_stability"] = stabilities
    out["mean_uncertainty_score"] = uncertainties
    out["confidence_score"] = confidence_scores
    out["confidence_level"] = confidence_levels
    out = out.sort_values(["confidence_score", "mean_candidate_score", "area_m2"], ascending=False).reset_index(drop=True)
    out["validation_priority_rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_label_from_id(candidate_id: Any) -> int:
    try:
        return int(str(candidate_id).split("-")[-1])
    except ValueError:
        return -1


def attach_nearest_candidate_context(landing_sites: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty or patches.empty:
        return landing_sites
    out = landing_sites.copy()
    nearest_ids: list[str] = []
    nearest_distances: list[float] = []
    nearest_confidence: list[str] = []
    target_distances: list[float] = []
    for _, site in out.iterrows():
        d = np.sqrt((patches["centroid_row"].astype(float) - float(site["row"])) ** 2 + (patches["centroid_col"].astype(float) - float(site["col"])) ** 2)
        idx = int(d.idxmin())
        patch = patches.loc[idx]
        pixel_size = float(site["distance_to_candidate_m"]) / max(float(site.get("distance_to_candidate_m", 0.0)) / 25.0, 1.0)
        # Use centroid distance for patch-level reporting; mask-edge distance remains in distance_to_candidate_m.
        centroid_distance_m = float(d.loc[idx] * 25.0)
        nearest_ids.append(str(patch["candidate_id"]))
        nearest_distances.append(centroid_distance_m)
        nearest_confidence.append(str(patch.get("confidence_level", "not_available")))
        target_distances.append(float(site.get("distance_to_candidate_m", np.nan)))
    out["nearest_candidate_id"] = nearest_ids
    out["nearest_candidate_confidence_level"] = nearest_confidence
    out["distance_to_nearest_candidate_patch_m"] = nearest_distances
    out["distance_to_target_candidate_m"] = target_distances
    return out


def attach_landing_route_accessibility(landing_sites: pd.DataFrame, route_summary: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    ok_routes = route_summary[route_summary["status"].astype(str).str.lower().eq("ok")] if not route_summary.empty else pd.DataFrame()
    target_ids = set(ok_routes["target_candidate_id"].astype(str)) if not ok_routes.empty and "target_candidate_id" in ok_routes else set()
    out["route_accessible"] = out["nearest_candidate_id"].astype(str).isin(target_ids).map({True: "yes", False: "needs route check"}) if "nearest_candidate_id" in out else "needs route check"
    out["route_accessibility_to_candidate_patch"] = out["route_accessible"]
    return out


def build_landing_site_evaluation(landing_sites: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    slope = out["slope_deg"].astype(float)
    dist = out["distance_to_candidate_m"].astype(float)
    out["low_slope_score"] = (1.0 - np.clip(slope / 12.0, 0, 1)).round(4)
    out["candidate_proximity_score"] = (1.0 / (1.0 + dist / 1000.0)).round(4)
    out["roughness_avoidance_status"] = "included via SAR texture proxy"
    out["candidate_mask_clearance_status"] = "outside candidate mask with configured clearance"
    out["local_terrain_smoothness_status"] = "represented by low slope and texture proxy"
    out["illumination_layer_status"] = "future data layer needed"
    out["thermal_layer_status"] = "future data layer needed"
    out["communication_layer_status"] = "future line-of-sight layer needed"
    out["reason_selected"] = "high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance"
    out["validation_needed"] = "illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation"
    columns = [
        "site_id", "lat", "lon", "slope_deg", "suitability_score", "distance_to_target_candidate_m",
        "nearest_candidate_id", "nearest_candidate_confidence_level", "route_accessible", "reason_selected",
        "validation_needed", "low_slope_score", "candidate_proximity_score", "roughness_avoidance_status",
        "candidate_mask_clearance_status", "illumination_layer_status", "thermal_layer_status", "communication_layer_status",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_candidate_scientific_review(patches: pd.DataFrame, landing_sites: pd.DataFrame, route_summary: pd.DataFrame) -> pd.DataFrame:
    if patches.empty:
        return patches
    out = patches.copy()
    nearest_sites: list[str] = []
    nearest_distances: list[float] = []
    for _, patch in out.iterrows():
        if landing_sites.empty:
            nearest_sites.append("")
            nearest_distances.append(np.nan)
            continue
        d = np.sqrt((landing_sites["row"].astype(float) - float(patch["centroid_row"])) ** 2 + (landing_sites["col"].astype(float) - float(patch["centroid_col"])) ** 2) * 25.0
        idx = int(d.idxmin())
        nearest_sites.append(str(landing_sites.loc[idx, "site_id"]))
        nearest_distances.append(float(d.loc[idx]))
    ok_target_ids = set(route_summary.loc[route_summary["status"].astype(str).str.lower().eq("ok"), "target_candidate_id"].astype(str)) if not route_summary.empty else set()
    out["distance_to_nearest_landing_candidate_m"] = nearest_distances
    out["nearest_landing_site_id"] = nearest_sites
    out["route_accessible_yes_no"] = out["candidate_id"].astype(str).isin(ok_target_ids).map({True: "yes", False: "needs route check"})
    keep = [
        "candidate_id", "area_m2", "area_km2", "equivalent_candidate_patch_diameter_m", "equivalent_diameter_m",
        "centroid_lat", "centroid_lon", "mean_candidate_score", "max_candidate_score", "mean_ratio_proxy",
        "mean_texture", "mean_slope_deg", "max_slope_deg", "distance_to_nearest_landing_candidate_m",
        "nearest_landing_site_id", "route_accessible_yes_no", "threshold_stability", "mean_uncertainty_score",
        "confidence_score", "confidence_level", "validation_priority_rank", "slope_condition",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("validation_priority_rank")


def build_ice_candidate_patch_review(candidate_review: pd.DataFrame) -> pd.DataFrame:
    if candidate_review.empty:
        return candidate_review
    out = candidate_review.copy()
    out["distance_to_nearest_landing_site_m"] = out["distance_to_nearest_landing_candidate_m"]
    columns = [
        "candidate_id", "area_m2", "equivalent_candidate_patch_diameter_m", "centroid_lat", "centroid_lon",
        "mean_candidate_score", "confidence_level", "mean_slope_deg", "max_slope_deg",
        "nearest_landing_site_id", "distance_to_nearest_landing_site_m", "validation_priority_rank",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_resource_scenario_estimates(candidate_review: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidate_review.empty:
        return pd.DataFrame(rows)
    top = candidate_review.sort_values("validation_priority_rank").head(top_n)
    for _, patch in top.iterrows():
        for depth_m in [1, 3, 5]:
            for ice_fraction in [0.05, 0.10, 0.20]:
                rows.append({
                    "candidate_id": patch["candidate_id"],
                    "confidence_level": patch.get("confidence_level", ""),
                    "candidate_area_m2": float(patch["area_m2"]),
                    "assumed_depth_m": depth_m,
                    "assumed_ice_fraction": ice_fraction,
                    "approx_candidate_volume_m3": float(patch["area_m2"]) * depth_m * ice_fraction,
                    "interpretation": "scenario-based potential resource volume if the candidate patch is later validated",
                })
    return pd.DataFrame(rows)


def build_sar_product_selection_reason(sar_scores: pd.DataFrame, pair: dict[str, Any], aoi: dict[str, Any]) -> pd.DataFrame:
    if sar_scores.empty:
        return pd.DataFrame()
    out = sar_scores.copy()
    out["selected_for_main_map"] = out["product_id"].astype(str).eq(str(pair["product_id"]))
    out["selection_reason"] = np.where(
        out["selected_for_main_map"],
        "selected because it provides full configured AOI coverage and usable SRI LH/LV intensity pair",
        np.where(out["coverage_fraction"].astype(float) > 0, "supporting/partial-overlap product", "excluded from main map because configured AOI overlap is zero"),
    )
    out["aoi_lat_min"] = aoi["lat_min"]
    out["aoi_lat_max"] = aoi["lat_max"]
    out["aoi_lon_min"] = aoi["lon_min"]
    out["aoi_lon_max"] = aoi["lon_max"]
    return out


def build_rover_navigation_evaluation(routes: dict[str, list[tuple[int, int]]], route_summary: pd.DataFrame, features: dict[str, np.ndarray], slope: np.ndarray | None, transform) -> pd.DataFrame:
    if route_summary.empty:
        return route_summary
    out = route_summary.copy()
    texture = robust_normalize(features["texture"], features["valid"].astype(bool))
    score = robust_normalize(features["candidate_score"], features["valid"].astype(bool))
    pixel_size = abs(float(transform.a))
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        route_type = str(row["route_type"])
        path = routes.get(route_type, [])
        slopes = path_values_for_profile(path, slope) if slope is not None else np.array([], dtype=float)
        tex = path_values_for_profile(path, texture)
        sci = path_values_for_profile(path, score)
        total = max(len(path), 1)
        under_5 = float(np.sum(slopes < 5) / total * 100) if slopes.size else np.nan
        five_to_10 = float(np.sum((slopes >= 5) & (slopes <= 10)) / total * 100) if slopes.size else np.nan
        above_10 = float(np.sum(slopes > 10) / total * 100) if slopes.size else np.nan
        blocked_avoided = int(np.sum(slopes > 15)) if slopes.size else 0
        science_reward = float(np.nanmean(sci)) if sci.size else np.nan
        energy_proxy = float(row.get("length_m", 0.0)) + 12.0 * float(np.nansum(np.clip(slopes, 0, None)) if slopes.size else 0.0) + 75.0 * float(np.nansum(tex) if tex.size else 0.0)
        risk = float(np.nanmean(np.clip(slopes / 15.0, 0, 1)) + np.nanmean(tex)) if slopes.size and tex.size else np.nan
        rows.append({
            **row.to_dict(),
            "length_m": float(row.get("length_m", 0.0)),
            "total_cost": float(row.get("total_cost", row.get("cost", np.nan))),
            "percent_route_under_5deg": under_5,
            "percent_route_5_to_10deg": five_to_10,
            "percent_route_above_10deg": above_10,
            "blocked_cells_avoided": blocked_avoided,
            "science_reward_score": science_reward,
            "energy_cost_proxy": energy_proxy,
            "traverse_risk_score": risk,
            "route_accessibility_note": "conceptual rover navigation planning prototype; not operational rover command generation",
        })
    return pd.DataFrame(rows)


def path_values_for_profile(path: list[tuple[int, int]], arr: np.ndarray | None) -> np.ndarray:
    if arr is None or not path:
        return np.array([], dtype=float)
    return np.array([arr[r, c] for r, c in path if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]], dtype=float)


def build_model_experiment_comparison(unet: dict[str, Any], features: dict[str, np.ndarray], pseudo_label: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = dict(unet.get("metrics", {}))
    rows.append({
        "experiment": "TinyUNet_5channel_weighted_loss",
        "input_channels": "SAR log intensity, ratio proxy, texture, polarization imbalance proxy, candidate score",
        "pseudo_iou": metrics.get("pseudo_iou", np.nan),
        "pseudo_dice": metrics.get("pseudo_dice", np.nan),
        "pseudo_precision": pseudo_precision(unet["prediction_mask"], pseudo_label),
        "pseudo_recall": pseudo_recall(unet["prediction_mask"], pseudo_label),
        "prediction_fraction": metrics.get("prediction_fraction", np.nan),
        "pseudo_label_fraction": metrics.get("pseudo_label_fraction", float(pseudo_label.mean())),
        "note": "weakly supervised pseudo-label agreement only",
    })
    score = robust_normalize(features["candidate_score"], features["valid"].astype(bool))
    baseline_pred = score >= np.nanquantile(score[np.isfinite(score)], 1.0 - float(np.clip(pseudo_label.mean(), 0.001, 0.20)))
    rows.append(metrics_row_for_prediction("candidate_score_quantile_baseline", baseline_pred, pseudo_label))
    strong_pred = score >= np.nanquantile(score[np.isfinite(score)], 0.90)
    rows.append(metrics_row_for_prediction("candidate_score_strict_threshold", strong_pred, pseudo_label))
    return pd.DataFrame(rows)


def metrics_row_for_prediction(name: str, pred: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    inter = float(np.logical_and(pred, label).sum())
    union = float(np.logical_or(pred, label).sum())
    denom = float(pred.sum() + label.sum())
    return {
        "experiment": name,
        "input_channels": "candidate score threshold baseline",
        "pseudo_iou": inter / union if union else 0.0,
        "pseudo_dice": 2 * inter / denom if denom else 0.0,
        "pseudo_precision": pseudo_precision(pred, label),
        "pseudo_recall": pseudo_recall(pred, label),
        "prediction_fraction": float(pred.mean()),
        "pseudo_label_fraction": float(label.mean()),
        "note": "baseline pseudo-label agreement only",
    }


def pseudo_precision(pred: np.ndarray, label: np.ndarray) -> float:
    pred = pred.astype(bool)
    label = label.astype(bool)
    tp = float(np.logical_and(pred, label).sum())
    fp = float(np.logical_and(pred, ~label).sum())
    return tp / max(tp + fp, 1.0)


def pseudo_recall(pred: np.ndarray, label: np.ndarray) -> float:
    pred = pred.astype(bool)
    label = label.astype(bool)
    tp = float(np.logical_and(pred, label).sum())
    fn = float(np.logical_and(~pred, label).sum())
    return tp / max(tp + fn, 1.0)


def build_candidate_summary(candidate_mask: np.ndarray, patches: pd.DataFrame, features: dict[str, np.ndarray], thresholds: dict[str, float], slope: np.ndarray | None) -> pd.DataFrame:
    valid = features["valid"].astype(bool)
    candidate_pixels = int(candidate_mask.sum())
    valid_pixels = int(valid.sum())
    row = {
        "candidate_patch_count": int(len(patches)),
        "candidate_pixels": candidate_pixels,
        "valid_aoi_pixels": valid_pixels,
        "candidate_area_pct_of_valid_aoi": float(candidate_pixels / max(valid_pixels, 1) * 100),
        "mean_candidate_score_in_mask": float(np.nanmean(features["candidate_score"][candidate_mask])) if candidate_pixels else np.nan,
        "score_threshold": thresholds.get("score_threshold"),
        "ratio_threshold": thresholds.get("ratio_threshold"),
        "intensity_threshold": thresholds.get("intensity_threshold"),
        "texture_threshold": thresholds.get("texture_threshold"),
        "top_patch_id": patches.iloc[0]["candidate_id"] if not patches.empty else "",
        "top_patch_area_m2": float(patches.iloc[0]["area_m2"]) if not patches.empty else np.nan,
        "top_patch_mean_score": float(patches.iloc[0]["mean_candidate_score"]) if not patches.empty else np.nan,
    }
    if slope is not None and candidate_pixels:
        row["candidate_mean_slope_deg"] = float(np.nanmean(slope[candidate_mask]))
        row["candidate_max_slope_deg"] = float(np.nanmax(slope[candidate_mask]))
    return pd.DataFrame([row])


def build_slope_stats(slope: np.ndarray | None, valid: np.ndarray) -> dict[str, float]:
    if slope is None:
        return {}
    mask = valid & np.isfinite(slope)
    total = max(int(mask.sum()), 1)
    return {
        "valid_slope_pixels": int(mask.sum()),
        "mean_slope_deg": float(np.nanmean(slope[mask])),
        "median_slope_deg": float(np.nanmedian(slope[mask])),
        "safe_lt_5deg_pct": float(((slope < 5) & mask).sum() / total * 100),
        "moderate_5_10deg_pct": float((((slope >= 5) & (slope <= 10)) & mask).sum() / total * 100),
        "unsafe_gt_10deg_pct": float(((slope > 10) & mask).sum() / total * 100),
        "blocked_gt_15deg_pct": float(((slope > 15) & mask).sum() / total * 100),
    }


def build_data_coverage_status(inventory: pd.DataFrame, sar_scores: pd.DataFrame, ohrc_fp: pd.DataFrame, pair: dict[str, Any], aoi: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"layer": "SAR/DFSAR", "status": "usable", "detail": f"Selected {pair['product_id']} with AOI coverage {float(pair['coverage_fraction']):.3f}"},
        {"layer": "DEM/LDEM/LDSM", "status": "usable", "detail": "DEM rasters cover configured south-pole AOI and were resampled to SAR AOI for planning context."},
        {"layer": "OHRC", "status": "context only", "detail": "Available OHRC footprints do not directly co-register to Faustini AOI; download calibrated overlapping product."},
        {"layer": "Ground truth labels", "status": "not available", "detail": "U-Net uses rule-based pseudo-labels; metrics are agreement only."},
    ])


def candidate_uncertainty(score: np.ndarray, threshold: float, valid: np.ndarray) -> np.ndarray:
    vals = score[valid & np.isfinite(score)]
    scale = max(float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5)), 1e-6) if vals.size else 1.0
    uncertainty = 1.0 - np.clip(np.abs(score - threshold) / (0.2 * scale), 0, 1)
    uncertainty[~valid | ~np.isfinite(uncertainty)] = np.nan
    return uncertainty.astype("float32")


def copy_legacy_figure_names(figures: Path) -> None:
    aliases = [
        ("04_sar_feature_panel.png", "sar_feature_panel.png"),
        ("05_radar_candidate_overlay.png", "sar_candidate_overlay.png"),
        ("05_radar_candidate_overlay.png", "radar_candidate_overlay.png"),
        ("06_candidate_patch_area_histogram.png", "candidate_patch_area_histogram.png"),
        ("08_dem_slope_map.png", "dem_slope_map.png"),
        ("08b_slope_classification_map.png", "slope_classification_map.png"),
        ("10_top_landing_candidates_overlay.png", "landing_suitability_overlay.png"),
        ("11_rover_route_comparison.png", "route_cost_comparison.png"),
        ("12_rover_route_overlay.png", "rover_route_overlay.png"),
        ("15_unet_prediction_overlay.png", "unet_prediction_overlay.png"),
        ("16_combined_decision_map.png", "combined_decision_map.png"),
    ]
    for src, dst in aliases:
        src_path = figures / src
        if src_path.exists():
            shutil.copyfile(src_path, figures / dst)


def save_pseudo_label_distribution(label: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "class": ["pseudo-label positive", "pseudo-label background", "prediction positive", "prediction background"],
        "pixels": [int(label.sum()), int((~label).sum()), int(prediction.sum()), int((~prediction).sum())],
    })
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(df["class"], df["pixels"], color=["#00bcd4", "#b0bec5", "#2f80ed", "#cfd8dc"])
    ax.set_title("Pseudo-label and Prediction Pixel Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Pixels")
    ax.tick_params(axis="x", rotation=18)
    fig.text(0.01, 0.01, "Class distribution for pseudo-label agreement evaluation only.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_region_validation_map(sar_scores: pd.DataFrame, pair: dict[str, Any], aoi: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = sar_scores.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.set_title("Configured Faustini/F2 Prototype AOI", fontsize=13, fontweight="bold")
    ax.add_patch(plt.Rectangle((aoi["lon_min"], aoi["lat_min"]), aoi["lon_max"] - aoi["lon_min"], aoi["lat_max"] - aoi["lat_min"], facecolor="#dff3ff", edgecolor="#0077b6", linewidth=2))
    ax.text((aoi["lon_min"] + aoi["lon_max"]) / 2, (aoi["lat_min"] + aoi["lat_max"]) / 2, "Faustini/F2 AOI", ha="center", va="center", fontsize=11, weight="bold")
    ax.set_xlabel("Longitude E")
    ax.set_ylabel("Latitude")
    ax.set_xlim(aoi["lon_min"] - 0.6, aoi["lon_max"] + 0.6)
    ax.set_ylim(aoi["lat_min"] - 0.15, aoi["lat_max"] + 0.15)
    ax.grid(alpha=0.25)
    ax = axes[1]
    if not df.empty:
        df["label"] = df["product_id"].astype(str).str.replace("ch2_sar_", "", regex=False).str[:24]
        colors = np.where(df["product_id"].astype(str).eq(str(pair["product_id"])), "#2f80ed", "#90a4ae")
        ax.barh(df["label"], df["coverage_fraction"].astype(float), color=colors)
        ax.axvline(1.0, color="#1b5e20", ls="--", lw=1)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AOI coverage fraction")
    ax.set_title("SAR Product Selection Reason", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "Selected SAR product provides full configured AOI coverage; partial/no-overlap products are supporting or excluded.", fontsize=9)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_ice_candidate_detection_map(base: np.ndarray, candidate_mask: np.ndarray, patches: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
    rgba[..., 0] = 0.0
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(candidate_mask, 0.48, 0)
    ax.imshow(rgba)
    top = patches.sort_values("validation_priority_rank").head(10) if "validation_priority_rank" in patches else patches.head(10)
    for _, row in top.iterrows():
        ax.text(row["centroid_col"] + 5, row["centroid_row"] + 5, row["candidate_id"], color="black", fontsize=8, weight="bold", bbox=dict(facecolor="white", edgecolor="#00acc1", alpha=0.8, pad=1.5))
    ax.set_title("Radar-Based Candidate Ice Regions - Faustini/F2 AOI", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    ax.legend(handles=[Line2D([0], [0], color="#00acc1", lw=7, alpha=0.6, label="candidate ice region")], loc="lower right", framealpha=0.9, fontsize=10)
    fig.text(0.01, 0.025, "Radar-based candidate ice regions; not confirmed ice. Screening result; validation required.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_confidence_map(base: np.ndarray, candidate_mask: np.ndarray, patches: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels, _ = ndi.label(candidate_mask)
    class_map = np.zeros(candidate_mask.shape, dtype=int)
    level_to_class = {"Low": 1, "Medium": 2, "High": 3}
    for _, row in patches.iterrows():
        label = candidate_label_from_id(row["candidate_id"])
        class_map[labels == label] = level_to_class.get(str(row.get("confidence_level", "Low")), 1)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    cmap = ListedColormap(["#00000000", "#ffcc00", "#00acc1", "#1b9e77"])
    ax.imshow(np.ma.masked_where(class_map == 0, class_map), cmap=cmap, vmin=0, vmax=3, alpha=0.72)
    top = patches.sort_values("validation_priority_rank").head(8) if "validation_priority_rank" in patches else patches.head(8)
    for _, row in top.iterrows():
        ax.text(row["centroid_col"] + 5, row["centroid_row"] + 5, row["candidate_id"], color="black", fontsize=8, weight="bold", bbox=dict(facecolor="white", edgecolor="black", alpha=0.78, pad=1.4))
    legend = [
        Line2D([0], [0], color="#1b9e77", lw=7, label="High confidence"),
        Line2D([0], [0], color="#00acc1", lw=7, label="Medium confidence"),
        Line2D([0], [0], color="#ffcc00", lw=7, label="Low confidence"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9, fontsize=10)
    ax.set_title("Candidate Confidence Map", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.025, "Confidence combines score, area, threshold stability, uncertainty, and slope context; validation required.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_threshold_sensitivity_curve(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(df["score_quantile"], df["candidate_patch_count"], marker="o", color="#2f80ed", label="candidate patches")
    ax2.plot(df["score_quantile"], df["candidate_area_m2"], marker="s", color="#f4511e", label="candidate area")
    ax1.set_xlabel("Candidate score quantile threshold")
    ax1.set_ylabel("Patch count", color="#2f80ed")
    ax2.set_ylabel("Candidate area (m2)", color="#f4511e")
    ax1.set_title("Threshold Sensitivity of Radar-Based Candidate Detection", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.02, "Stable candidates across thresholds receive higher confidence; validation required.", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_ranking_chart(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = review.sort_values("validation_priority_rank").head(10) if not review.empty else review
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    if not top.empty:
        axes[0].bar(top["candidate_id"], top["mean_candidate_score"], color="#00acc1")
        axes[1].bar(top["candidate_id"], top["area_m2"], color="#f4511e")
        slope_colors = top["confidence_level"].map({"High": "#1b9e77", "Medium": "#00acc1", "Low": "#ffcc00"}).fillna("#90a4ae")
        axes[2].bar(top["candidate_id"], top["mean_slope_deg"], color=slope_colors)
    labels = [("Mean score", "Mean candidate score"), ("Area (m2)", "Patch area"), ("Mean slope (deg)", "Slope context")]
    for ax, (ylabel, title) in zip(axes, labels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Candidate Patch Ranking for Validation Priority", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Ranking balances screening score, extent, confidence, and terrain context; validation required.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_area_vs_score(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    if not review.empty:
        colors = review["confidence_level"].map({"High": "#1b9e77", "Medium": "#00acc1", "Low": "#ffcc00"}).fillna("#90a4ae")
        sizes = np.clip(review["equivalent_candidate_patch_diameter_m"].astype(float) * 2.0, 35, 240)
        ax.scatter(review["area_m2"], review["mean_candidate_score"], s=sizes, c=colors, edgecolor="black", linewidth=0.4, alpha=0.86)
        for _, row in review.sort_values("validation_priority_rank").head(8).iterrows():
            ax.text(row["area_m2"] * 1.01, row["mean_candidate_score"] + 0.002, row["candidate_id"], fontsize=8)
    ax.set_xlabel("Candidate patch area (m2)")
    ax.set_ylabel("Mean candidate score")
    ax.set_title("Candidate Area vs Score", fontsize=14, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#1b9e77", lw=0, label="High confidence"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#00acc1", lw=0, label="Medium confidence"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#ffcc00", lw=0, label="Low confidence"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.01, "High score and large extent are balanced against slope context; validation required.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_diameter_distribution(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if not review.empty:
        ax.hist(review["equivalent_candidate_patch_diameter_m"].dropna(), bins=min(18, max(6, len(review) // 2)), color="#5c6bc0", alpha=0.88)
    ax.set_title("Equivalent Candidate Patch Diameter Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Equivalent candidate patch diameter (m)")
    ax.set_ylabel("Patch count")
    fig.text(0.02, 0.035, "Equivalent diameter is an area-derived screening extent, not a measured ice diameter.", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.18)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_candidate_score_vs_slope(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    if not review.empty:
        colors = review["confidence_level"].map({"High": "#1b9e77", "Medium": "#00acc1", "Low": "#ffcc00"}).fillna("#90a4ae")
        ax.scatter(review["mean_slope_deg"], review["mean_candidate_score"], s=np.clip(review["area_m2"].astype(float) / 60, 28, 180), c=colors, edgecolor="black", linewidth=0.4)
        ax.axvspan(0, 5, color="#1b9e77", alpha=0.12, label="low slope")
        ax.axvspan(5, 10, color="#ffcc00", alpha=0.10, label="moderate slope")
        ax.axvspan(10, max(25, float(review["mean_slope_deg"].max()) + 2), color="#e74c3c", alpha=0.09, label="steeper context")
        for _, row in review.sort_values("validation_priority_rank").head(8).iterrows():
            ax.text(row["mean_slope_deg"] + 0.2, row["mean_candidate_score"] + 0.002, row["candidate_id"], fontsize=8)
    ax.set_xlabel("Mean slope inside candidate patch (deg)")
    ax.set_ylabel("Mean candidate score")
    ax.set_title("Candidate Score vs Slope Context", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.text(0.01, 0.01, "High-score patches in steep terrain are validation targets, not automatic landing/traverse targets.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_resource_scenario_bar_chart(scenarios: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    if not scenarios.empty:
        top = scenarios[(scenarios["assumed_depth_m"] == 3) & (np.isclose(scenarios["assumed_ice_fraction"], 0.10))].head(10)
        ax.bar(top["candidate_id"], top["approx_candidate_volume_m3"], color="#2f80ed")
    ax.set_title("Scenario-Based Potential Resource Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Candidate patch")
    ax.set_ylabel("Scenario volume (m3)")
    ax.tick_params(axis="x", rotation=30)
    fig.text(0.01, 0.01, "Scenario shown: 3 m assumed depth, 10% assumed ice fraction; not measured ice volume.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_landing_score_components_evaluation(landing_eval: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    if not landing_eval.empty:
        axes[0].bar(landing_eval["site_id"], landing_eval["suitability_score"], color="#43a047")
        axes[1].bar(landing_eval["site_id"], landing_eval["low_slope_score"], color="#00acc1")
        axes[2].bar(landing_eval["site_id"], landing_eval["candidate_proximity_score"], color="#f4511e")
    for ax, title, ylabel in zip(axes, ["Total score", "Low-slope score", "Candidate proximity score"], ["score", "score", "score"]):
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Preliminary Landing Score Components", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Illumination, thermal, and communication layers are future inputs, not fabricated scores.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_nearest_landing_to_candidate_map(base: np.ndarray, candidate_mask: np.ndarray, patches: pd.DataFrame, landing_sites: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(candidate_mask, 0.36, 0)
    ax.imshow(rgba)
    if not landing_sites.empty:
        ax.scatter(landing_sites["col"], landing_sites["row"], s=85, c="#00ff66", edgecolor="black", linewidth=0.8, label="preliminary landing candidate")
        for _, site in landing_sites.iterrows():
            patch_rows = patches[patches["candidate_id"].astype(str).eq(str(site.get("nearest_candidate_id", "")))]
            if not patch_rows.empty:
                patch = patch_rows.iloc[0]
                ax.plot([site["col"], patch["centroid_col"]], [site["row"], patch["centroid_row"]], color="#ffcc00", lw=1.6)
                mid_x = (site["col"] + patch["centroid_col"]) / 2
                mid_y = (site["row"] + patch["centroid_row"]) / 2
                ax.text(mid_x, mid_y, f"{float(site.get('distance_to_candidate_m', np.nan)):.0f} m", fontsize=7, color="black", bbox=dict(facecolor="white", alpha=0.75, pad=1))
            ax.text(site["col"] + 4, site["row"] + 4, site["site_id"], color="white", fontsize=9, weight="bold")
    ax.set_title("Nearest Landing Candidate to Candidate Ice Patch", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.025, "Lines show planning relationship from preliminary landing candidates to nearest candidate patches; validation required.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_route_profile(routes: dict[str, list[tuple[int, int]]], slope: np.ndarray | None, path: Path, value_name: str, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    if slope is not None:
        colors = {"shortest": "#ffcc00", "safest": "#ff2bbd", "science_priority": "#00bcd4", "energy_efficient": "#43a047"}
        for name, route in routes.items():
            vals = path_values_for_profile(route, slope)
            if vals.size:
                ax.plot(np.arange(vals.size), vals, label=name, color=colors.get(name, None), lw=1.8)
    ax.axhline(5, color="#1b9e77", ls="--", lw=1)
    ax.axhline(10, color="#f4511e", ls="--", lw=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Route step")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9)
    fig.text(0.01, 0.01, "Profiles are sampled from DEM-derived slope along conceptual route variants.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_route_risk_profile(routes: dict[str, list[tuple[int, int]]], features: dict[str, np.ndarray], slope: np.ndarray | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    texture = robust_normalize(features["texture"], features["valid"].astype(bool))
    colors = {"shortest": "#ffcc00", "safest": "#ff2bbd", "science_priority": "#00bcd4", "energy_efficient": "#43a047"}
    for name, route in routes.items():
        tex = path_values_for_profile(route, texture)
        slp = path_values_for_profile(route, slope) if slope is not None else np.zeros_like(tex)
        if tex.size:
            risk = np.clip(slp / 15.0, 0, 1) + tex
            ax.plot(np.arange(risk.size), risk, label=name, color=colors.get(name, None), lw=1.8)
    ax.set_title("Rover Route Risk Profile", fontsize=14, fontweight="bold")
    ax.set_xlabel("Route step")
    ax.set_ylabel("Risk proxy")
    ax.legend(framealpha=0.9)
    fig.text(0.01, 0.01, "Risk proxy combines slope penalty and SAR texture roughness; validation required.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_model_experiment_comparison(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    if not df.empty:
        labels = df["experiment"].astype(str).str.replace("_", "\n")
        axes[0].bar(labels, df["pseudo_iou"].astype(float), color="#00acc1")
        axes[1].bar(labels, df["pseudo_dice"].astype(float), color="#43a047")
    axes[0].set_title("Pseudo-IoU")
    axes[1].set_title("Pseudo-Dice")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Model Experiment Comparison", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "All metrics are pseudo-label agreement, not independently validated composition accuracy.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_unet_error_map(pseudo_label: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error = np.zeros(pseudo_label.shape, dtype=int)
    error[np.logical_and(pseudo_label, prediction)] = 1
    error[np.logical_and(~pseudo_label, prediction)] = 2
    error[np.logical_and(pseudo_label, ~prediction)] = 3
    cmap = ListedColormap(["#f5f5f5", "#1b9e77", "#e74c3c", "#2f80ed"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(error, cmap=cmap, vmin=0, vmax=3)
    legend = [
        Line2D([0], [0], color="#1b9e77", lw=7, label="agreement positive"),
        Line2D([0], [0], color="#e74c3c", lw=7, label="prediction-only"),
        Line2D([0], [0], color="#2f80ed", lw=7, label="pseudo-label missed"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9)
    ax.set_title("U-Net Error Map Against Pseudo-Label", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.02, "Error classes compare against rule-based pseudo-labels only; not ground-truth ice labels.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    inventory: pd.DataFrame,
    sar_scores: pd.DataFrame,
    patches: pd.DataFrame,
    landing_sites: pd.DataFrame,
    route_summary: pd.DataFrame,
    ohrc_fp: pd.DataFrame,
    unet: dict[str, Any],
    candidate_summary: pd.DataFrame,
    slope_stats: dict[str, Any],
) -> None:
    selected = manifest["selected_sar"]
    overlap_rows = sar_scores[["product_id", "coverage_fraction", "pixel_size_m", "lh"]].copy() if not sar_scores.empty else pd.DataFrame()
    top_candidate = patches.iloc[0].to_dict() if not patches.empty else {}
    top_site = landing_sites.iloc[0].to_dict() if not landing_sites.empty else {}
    candidate_pct = candidate_summary.iloc[0].get("candidate_area_pct_of_valid_aoi", np.nan) if not candidate_summary.empty else np.nan
    lines = [
        "# LunaQuest Prototype Summary",
        "",
        "## What was built",
        "",
        "This run creates a research-notebook-ready prototype for radar-based candidate screening, DEM terrain context, preliminary landing suitability, conceptual rover routing, and weakly supervised pseudo-label segmentation.",
        "",
        "No output is presented as compositional proof. Candidate masks are screening outputs requiring independent validation.",
        "",
        "## Actual data availability",
        "",
        f"- Inventory rows: {len(inventory)}",
        f"- Selected SAR product: `{selected.get('product_id')}`",
        f"- Selected SAR AOI coverage fraction: {float(selected.get('coverage_fraction', 0)):.3f}",
        f"- Selected SAR pixel size: {selected.get('pixel_size_m')} m",
        "- DEM files found and used for terrain/slope context.",
        "- OHRC files are zip bundles with browse PNG and IMG data; current OHRC footprints are not directly co-registered to the configured Faustini AOI.",
        "",
        "## Candidate screening result",
        "",
        f"- Candidate patches found: {len(patches)}",
        f"- Candidate mask area: {candidate_pct:.3f}% of valid AOI pixels",
        f"- Top candidate: {top_candidate.get('candidate_id', 'none')} with area {top_candidate.get('area_m2', 'n/a')} m2 and mean score {top_candidate.get('mean_candidate_score', 'n/a')}",
        "",
        "The SAR features are intensity, LH/LV ratio proxy, polarization imbalance proxy, texture, and a combined candidate score. True CPR/DOP are not claimed because the available SRI rasters are real-valued intensity products rather than the complex/Stokes products needed for a defensible CPR/DOP derivation.",
        "",
        "## Landing and route prototype",
        "",
        f"- Landing candidates found: {len(landing_sites)}",
        f"- Top landing candidate: {top_site.get('site_id', 'none')} with suitability score {top_site.get('suitability_score', 'n/a')}",
        f"- Route variants: {', '.join(route_summary['route_type'].astype(str)) if not route_summary.empty else 'none'}",
        f"- Blocked slope threshold for routing: >15 deg",
        "",
        "## U-Net prototype",
        "",
        unet["note"],
        "",
        f"Pseudo-label metrics: `{json.dumps(unet['metrics'], indent=2)}`",
        "",
        "## Best current research outputs",
        "",
        "- `outputs/figures/04_sar_feature_panel.png`",
        "- `outputs/figures/05_radar_candidate_overlay.png`",
        "- `outputs/figures/08_dem_slope_map.png`",
        "- `outputs/figures/12_rover_route_overlay.png`",
        "- `outputs/figures/14_unet_training_curve.png`",
        "- `outputs/figures/16_combined_decision_map.png`",
        "",
        "## Limitations and next steps",
        "",
        "- Replace CPR-style proxy with published CPR/DOP only after confirming compact-pol channel convention and availability of complex/Stokes products.",
        "- Download a calibrated OHRC product overlapping the configured Faustini AOI before claiming pixel-level optical hazard/boulder analysis.",
        "- Add illumination, thermal, and communication layers for stronger landing-site scoring.",
        "- Treat U-Net metrics as pseudo-label agreement only, not lunar composition accuracy.",
        "",
        "## How to run",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m lunar_icenav.cli run --config configs/pipeline.json",
        "python -m lunar_icenav.cli notebook --config configs/pipeline.json",
        "```",
    ]
    if slope_stats:
        lines.extend(["", "## DEM slope safety summary", "", dataframe_to_markdown(pd.DataFrame([slope_stats]))])
    if not overlap_rows.empty:
        lines.extend(["", "## SAR AOI overlap table", "", dataframe_to_markdown(overlap_rows)])
    if not ohrc_fp.empty:
        lines.extend(["", "## OHRC footprint note", "", dataframe_to_markdown(ohrc_fp[["product_id", "lat_min", "lat_max", "coverage_note"]])])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ohrc_download_note(path: Path, aoi: dict[str, Any]) -> None:
    lines = [
        "# OHRC Data Download Needed",
        "",
        "Current OHRC products are not co-registered to the configured Faustini/F2 AOI. They are useful as context-only browse examples, but should not be used as Faustini hazard/boulder maps.",
        "",
        "Download calibrated OHRC products from PRADAN for:",
        "",
        f"- Lat Min: {aoi['lat_min']}",
        f"- Lat Max: {aoi['lat_max']}",
        f"- Lon Min: {aoi['lon_min']}",
        f"- Lon Max: {aoi['lon_max']}",
        "",
        "Selection notes:",
        "",
        "- Choose calibrated OHRC products.",
        "- Preserve the full product bundle with data, geometry, browse, and XML metadata.",
        "- After download, rerun `python -m lunar_icenav.cli run --config configs/pipeline.json`.",
        "- Only claim OHRC hazard/boulder context after footprint and grid alignment are demonstrated.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_improvement_summary(path: Path, patches: pd.DataFrame, landing_sites: pd.DataFrame, routes: pd.DataFrame, unet: dict[str, Any], slope_stats: dict[str, Any]) -> None:
    metrics = unet.get("metrics", {})
    lines = [
        "# Next 5 Days Improvement Summary",
        "",
        "## Improved Compared To First Prototype",
        "",
        "- Added numbered, 300 DPI research figures for inventory, coverage, SAR features, candidate patches, DEM terrain, landing, rover routes, OHRC context, U-Net curves, and combined decision support.",
        "- Added candidate patch summary, data coverage status, slope safety summary, U-Net training history, U-Net tile inventory, and checkpoint output.",
        "- Improved U-Net section with spatial tile split, augmentation, training/validation curves, pseudo-IoU, and pseudo-Dice.",
        "- Improved route table with route length, cost, mean slope, max slope, and target candidate ID.",
        "- Added OHRC download instruction report for the correct Faustini/F2 AOI.",
        "",
        "## Stronger Outputs Now Available",
        "",
        "- `outputs/figures/04_sar_feature_panel.png`",
        "- `outputs/figures/05_radar_candidate_overlay.png`",
        "- `outputs/figures/06b_top_candidate_patches.png`",
        "- `outputs/figures/08b_slope_classification_map.png`",
        "- `outputs/figures/12_rover_route_overlay.png`",
        "- `outputs/figures/14_unet_training_curve.png`",
        "- `outputs/figures/16_combined_decision_map.png`",
        "",
        "## Current Evaluation-Style Summary",
        "",
        f"- Radar-based candidate patches: {len(patches)}",
        f"- Preliminary landing candidates: {len(landing_sites)}",
        f"- Route variants: {len(routes)}",
        f"- U-Net pseudo-IoU: {metrics.get('pseudo_iou', 'n/a')}",
        f"- U-Net pseudo-Dice: {metrics.get('pseudo_dice', 'n/a')}",
        f"- Safe slope area <5 deg: {slope_stats.get('safe_lt_5deg_pct', 'n/a')}%",
        "",
        "## Still Needs Validation",
        "",
        "- CPR/DOP must be derived only after validating product convention or obtaining complex/Stokes layers.",
        "- OHRC Faustini-overlapping product is still needed for real optical hazard analysis.",
        "- Illumination, thermal, and communication layers are placeholders/future layers.",
        "- U-Net metrics are pseudo-label agreement only, not true ice detection accuracy.",
        "",
        "## Recommended Next Technical Work",
        "",
        "1. Download calibrated OHRC for the exact Faustini AOI and rerun coverage checks.",
        "2. Add at least one illumination or PSR/shadow layer if available.",
        "3. Manually review top candidate patches and mark 3-5 examples for field validation planning.",
        "4. Convert notebook outputs into a dashboard or communication artifact only after technical review.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scientific_limitation_checklist(path: Path) -> None:
    lines = [
        "# Scientific Limitation Checklist",
        "",
        "- [ ] State that there is no direct water-ice confirmation claim in the current outputs.",
        "- [ ] Do not present any output as direct compositional proof.",
        "- [ ] State that true CPR/DOP are not yet claimed from the selected SRI intensity rasters.",
        "- [ ] State that current SAR features are screening proxies: intensity, LH/LV ratio proxy, imbalance proxy, texture, and candidate score.",
        "- [ ] State that OHRC is context-only because current OHRC products do not co-register with Faustini/F2 AOI.",
        "- [ ] State that U-Net is weakly supervised with rule-based pseudo-labels only.",
        "- [ ] State that pseudo-IoU and pseudo-Dice measure agreement with pseudo-labels, not composition accuracy.",
        "- [ ] State that landing candidates are preliminary and not certified landing products.",
        "- [ ] State that rover route variants are conceptual and not mission-ready traverses.",
        "- [ ] State that illumination, thermal, communication, and higher-confidence hazard/boulder layers remain future work.",
        "- [ ] State that candidate ranking is for review priority, not ice volume estimation.",
        "",
        "Recommended report footer: Preliminary candidate screening result; independent validation required.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_evaluation_report(
    path: Path,
    manifest: dict[str, Any],
    inventory: pd.DataFrame,
    sar_scores: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_review: pd.DataFrame,
    landing_eval: pd.DataFrame,
    rover_eval: pd.DataFrame,
    unet: dict[str, Any],
    model_experiments: pd.DataFrame,
    slope_stats: dict[str, Any],
    data_status: pd.DataFrame,
    resource_scenarios: pd.DataFrame,
) -> None:
    selected = manifest["selected_sar"]
    top_candidates = candidate_review.sort_values("validation_priority_rank").head(8) if not candidate_review.empty else candidate_review
    steep_candidates = candidate_review[candidate_review["mean_slope_deg"].astype(float) > 10].head(8) if not candidate_review.empty and "mean_slope_deg" in candidate_review else pd.DataFrame()
    best_landing = landing_eval.iloc[0].to_dict() if not landing_eval.empty else {}
    route_ok = rover_eval[rover_eval["status"].astype(str).str.lower().eq("ok")] if not rover_eval.empty else pd.DataFrame()
    best_route = route_ok.sort_values(["traverse_risk_score", "energy_cost_proxy"]).iloc[0].to_dict() if not route_ok.empty else {}
    metrics = unet.get("metrics", {})
    lines = [
        "# Model Evaluation Report",
        "",
        "This report evaluates the current LunaQuest / Lunar IceNav research prototype. It focuses on radar-based candidate patch detection, candidate patch scientific review, landing-site dependency on the candidate map, rover navigation evaluation, and pseudo-label ML agreement.",
        "",
        "## 1. What Data Was Processed?",
        "",
        f"- Inventory rows processed: {len(inventory)}.",
        "- Product classes include SAR/DFSAR, DEM/topography, OHRC bundles, and documents.",
        "- OHRC remains context-only until a Faustini/F2-overlapping calibrated product is downloaded and co-registered.",
        "",
        "## 2. Selected SAR Product And Reason",
        "",
        f"- Selected SAR product: `{selected.get('product_id')}`.",
        f"- Configured AOI coverage fraction: {float(selected.get('coverage_fraction', 0)):.3f}.",
        f"- AOI: lat {manifest['aoi']['lat_min']} to {manifest['aoi']['lat_max']}, lon {manifest['aoi']['lon_min']} to {manifest['aoi']['lon_max']} E.",
        "- Partial-overlap products are supporting only; no-overlap products are excluded from the main candidate map.",
        "",
        "## 3. Extracted SAR Features",
        "",
        "- SAR log intensity.",
        "- LH channel and LV channel.",
        "- CPR-style LH/LV ratio proxy and LV/LH ratio proxy.",
        "- Polarization imbalance proxy.",
        "- Local texture / roughness, local mean, and local standard deviation.",
        "- Candidate score and threshold uncertainty.",
        "",
        "True CPR/DOP are not claimed from the selected SRI intensity rasters.",
        "",
        "## Ice Candidate Detection Before Landing Site Search",
        "",
        "The workflow now explicitly generates the radar-based candidate ice map before landing analysis. Candidate screening starts from SAR proxy features, applies thresholded candidate-score and channel constraints, removes small connected components, and then evaluates connected candidate patches before any landing-site search.",
        "",
        f"- Candidate patches generated: {int(candidate_summary.iloc[0]['candidate_patch_count']) if not candidate_summary.empty else 'n/a'}.",
        f"- Candidate area percentage of valid AOI pixels: {float(candidate_summary.iloc[0]['candidate_area_pct_of_valid_aoi']):.3f}%." if not candidate_summary.empty else "- Candidate area percentage unavailable.",
        "- Threshold sensitivity is saved to `outputs/tables/threshold_sensitivity.csv` and `outputs/figures/threshold_sensitivity_curve.png`.",
        "- Candidate stability across thresholds contributes to the `confidence_level` column.",
        "- Landing candidates are selected near evaluated candidate patches but outside the candidate mask and risky/steep zones.",
        "- This is a planning layer, not confirmed ice detection.",
        "",
        "## 4. How Candidate Patches Were Generated",
        "",
        "- Candidate pixels are selected using SAR proxy thresholds for ratio, intensity, texture, and candidate score.",
        "- Connected components define candidate patches.",
        "- Each patch is evaluated for area, equivalent candidate patch diameter, score, ratio proxy, texture, slope context, uncertainty, and threshold stability.",
        "- Confidence levels are High/Medium/Low based on score, extent, stability, uncertainty, and slope/traverse context.",
        "",
        "## 5. Strongest Scientific Candidate Patches",
        "",
        dataframe_to_markdown(top_candidates[[
            "candidate_id", "area_m2", "equivalent_candidate_patch_diameter_m", "mean_candidate_score",
            "confidence_level", "mean_slope_deg", "max_slope_deg", "nearest_landing_site_id",
            "distance_to_nearest_landing_candidate_m", "validation_priority_rank",
        ]] if not top_candidates.empty else top_candidates),
        "",
        "## 6. Candidates With Steep Terrain Tradeoff",
        "",
        "Some high-score patches are less attractive as direct landing/traverse targets because their local slope context is steep. These can still be useful validation targets, but they should not drive landing-site selection without terrain review.",
        "",
        dataframe_to_markdown(steep_candidates[[
            "candidate_id", "mean_candidate_score", "area_m2", "mean_slope_deg", "max_slope_deg", "confidence_level",
        ]] if not steep_candidates.empty else steep_candidates),
        "",
        "## 7. Best Landing Candidates And Why",
        "",
        dataframe_to_markdown(landing_eval.head(5) if not landing_eval.empty else landing_eval),
        "",
        f"Best current preliminary landing candidate: `{best_landing.get('site_id', 'n/a')}` with suitability score {best_landing.get('suitability_score', 'n/a')}. It is selected because it is low slope, near a candidate patch, outside the candidate mask, and within the current proxy safety constraints.",
        "",
        "## 8. Best Rover Route And Why",
        "",
        dataframe_to_markdown(rover_eval[[
            "route_type", "target_candidate_id", "start_landing_site_id", "length_m", "total_cost",
            "mean_slope_deg", "max_slope_deg", "percent_route_under_5deg", "percent_route_above_10deg",
            "science_reward_score", "energy_cost_proxy", "traverse_risk_score",
        ]] if not rover_eval.empty else rover_eval),
        "",
        f"Best current route by low risk and energy proxy: `{best_route.get('route_type', 'n/a')}`. This remains a conceptual rover route for planning, not operational rover command generation.",
        "",
        "## 9. What The U-Net Proves And Does Not Prove",
        "",
        f"- Pseudo-IoU: {metrics.get('pseudo_iou', 'n/a')}.",
        f"- Pseudo-Dice: {metrics.get('pseudo_dice', 'n/a')}.",
        "- These metrics measure agreement with rule-based pseudo-labels only.",
        "- They do not measure independently validated ice detection accuracy.",
        "",
        dataframe_to_markdown(model_experiments if not model_experiments.empty else pd.DataFrame()),
        "",
        "## 10. Validation Still Needed",
        "",
        "- Download and co-register calibrated OHRC for the configured Faustini/F2 AOI.",
        "- Add illumination/PSR, thermal, and communication layers.",
        "- Validate CPR/DOP only with correct product convention or complex/Stokes products.",
        "- Replace pseudo-labels with stronger labels or multi-pass scientific consistency checks.",
        "- Manually review top candidate patches and route corridors.",
        "",
        "## Supporting Tables",
        "",
        "### Data Coverage Status",
        "",
        dataframe_to_markdown(data_status),
        "",
        "### Slope Safety Summary",
        "",
        dataframe_to_markdown(pd.DataFrame([slope_stats]) if slope_stats else pd.DataFrame()),
        "",
        "### Resource Scenario Summary",
        "",
        "Scenario estimates are potential planning cases only if candidate patches are later validated.",
        "",
        dataframe_to_markdown(resource_scenarios.head(12) if not resource_scenarios.empty else resource_scenarios),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_technical_limitations(path: Path, aoi: dict[str, Any]) -> None:
    lines = [
        "# Technical Limitations",
        "",
        "- No direct water-ice confirmation claim is made from current outputs.",
        "- Current SAR products are real-valued SRI intensity rasters; true CPR/DOP are future validated modules unless complex/Stokes products and conventions are available.",
        "- Candidate maps are radar-based screening outputs and require independent validation.",
        "- Equivalent candidate patch diameter is an area-derived screening extent, not a measured ice diameter.",
        "- Resource scenario estimates are planning scenarios, not measured resource volumes.",
        "- OHRC is context-only because current products do not co-register with the configured Faustini/F2 AOI.",
        "- Landing candidates are preliminary and require illumination/PSR, thermal, communications, boulder, hazard, and manual terrain review.",
        "- Rover routes are conceptual route variants on proxy cost maps, not operational command products.",
        "- U-Net outputs are weakly supervised pseudo-label agreement results, not independently validated ice detection accuracy.",
        "",
        "## Configured AOI",
        "",
        f"- Lat: {aoi['lat_min']} to {aoi['lat_max']}",
        f"- Lon: {aoi['lon_min']} to {aoi['lon_max']} E",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_data_to_download(path: Path, aoi: dict[str, Any]) -> None:
    lines = [
        "# Next Data To Download",
        "",
        "Priority data needed to move from screening prototype toward stronger scientific validation:",
        "",
        "1. Calibrated OHRC product overlapping the configured Faustini/F2 AOI.",
        "2. Illumination / PSR / shadow persistence layer for the AOI.",
        "3. Thermal stability layer or Diviner-derived thermal context if available.",
        "4. Communication line-of-sight / Earth visibility layer.",
        "5. Complex or Stokes SAR products, or authoritative product convention documentation, before deriving true CPR/DOP.",
        "6. Additional multi-pass SAR coverage for candidate stability checks.",
        "7. Manual validation reference layers for boulders, crater rims, and roughness/hazard review.",
        "",
        "## AOI For Search",
        "",
        f"- Latitude min: {aoi['lat_min']}",
        f"- Latitude max: {aoi['lat_max']}",
        f"- Longitude min: {aoi['lon_min']}",
        f"- Longitude max: {aoi['lon_max']} E",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[col]).replace("|", "/") for col in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def create_notebook(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    root_cell = """from pathlib import Path
import sys

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'configs' / 'pipeline.json').exists() and (candidate / 'src').exists():
            return candidate
    raise RuntimeError('Could not find project root containing configs/pipeline.json and src/')

root = find_project_root(Path.cwd().resolve())
sys.path.insert(0, str(root / 'src'))
figdir = root / 'outputs' / 'figures'
tables = root / 'outputs' / 'tables'
routes_dir = root / 'outputs' / 'routes'
print(root)"""
    nb.cells = [
        md("# LunaQuest / Lunar IceNav Research Workflow\n\nStep-by-step research notebook for BAH 2026 Problem Statement 8. All maps and model outputs are candidate/prototype products and require independent validation."),
        md("## A. Problem Overview\n\nThis project screens Chandrayaan-2 SAR/DFSAR observations for radar-based candidate subsurface ice signatures near the lunar south pole, then connects those candidates to preliminary landing-site and conceptual rover-route planning. The configured focus region is Faustini/F2 because it is a high-priority south-polar candidate area. Outputs are screening results, not compositional proof, certified landing products, or mission-ready rover routes."),
        code(root_cell),
        md("## B. Dataset Inventory\n\nThe inventory separates SAR/DFSAR, OHRC, DEM/topography, browse/quicklook, geometry tables, and documents. This section also shows which data are usable for candidate screening, terrain context, or context-only interpretation."),
        code("import pandas as pd\nfrom IPython.display import display, Image\ninv = pd.read_csv(tables / 'product_inventory.csv')\ndisplay(inv[['product_type','role','resolution','usable']].value_counts().reset_index(name='rows').head(20))\ndisplay(inv[['path','product_type','role','resolution','usable','notes']].head(25))\ndisplay(Image(filename=str(figdir / '01_dataset_inventory_chart.png')))"),
        md("## C. AOI and Coverage Verification\n\nThe configured AOI is lat -87.8 to -86.9 deg and lon 80 to 85 deg. The selected SAR product is chosen from actual AOI overlap, not from filename guesswork."),
        code("import json\nmanifest = json.loads((root / 'reports' / 'run_manifest.json').read_text())\nprint('AOI:', manifest['aoi'])\ncoverage = pd.read_csv(tables / 'sar_aoi_overlap.csv')\ndisplay(coverage[['product_id','coverage_fraction','pixel_size_m']])\ndisplay(Image(filename=str(figdir / '02_selected_sar_coverage.png')))"),
        md("## D. SAR Product Inspection\n\nThis section records the selected calibrated SRI LH/LV pair, raster dimensions, pixel size, window, CRS status, and channel usage."),
        code("sar_meta = pd.read_csv(tables / 'selected_sar_metadata.csv')\ndisplay(sar_meta.T)\ndisplay(Image(filename=str(figdir / '03_sar_quicklook.png')))"),
        md("## E. SAR Feature Extraction\n\nThe selected SRI rasters are intensity products. The workflow therefore uses transparent screening proxies: log intensity, LH/LV ratio proxy, polarization imbalance proxy, texture roughness, and a normalized candidate score. True CPR/DOP are not claimed from these arrays."),
        code("import numpy as np\nfeatures = np.load(root / 'outputs' / 'features' / 'sar_feature_stack.npz')\nprint('Feature arrays:', list(features.keys()))\nprint('AOI feature shape:', features['intensity'].shape)\ndisplay(Image(filename=str(figdir / '04_sar_feature_panel.png')))"),
        md("## F. Radar Candidate Screening\n\nCandidate mask generation uses thresholded screening features, then connected-component patch extraction. These are radar-based candidate patches that require independent validation."),
        code("candidate_summary = pd.read_csv(tables / 'candidate_patch_summary.csv')\npatches = pd.read_csv(tables / 'candidate_patches.csv')\ndisplay(candidate_summary)\ndisplay(patches.head(10))\ndisplay(Image(filename=str(figdir / '05_radar_candidate_overlay.png')))"),
        md("## G. Candidate Patch Analysis\n\nThese charts show area distribution, top patches, score distribution, centroid locations, and a screening uncertainty proxy. Uncertainty is high near the decision threshold."),
        code("for name in ['06_candidate_patch_area_histogram.png','06b_top_candidate_patches.png','06c_candidate_score_distribution.png','06d_candidate_centroid_map.png','06e_candidate_uncertainty_map.png']:\n    display(Image(filename=str(figdir / name)))\nprint('Top 10 patches by score:')\ndisplay(patches.sort_values('mean_candidate_score', ascending=False).head(10)[['candidate_id','area_m2','centroid_lat','centroid_lon','mean_candidate_score','mean_ratio_proxy']])"),
        md("## H. DEM / Slope / Terrain Analysis\n\nLDEM/LDSM data are cropped/resampled to the SAR AOI. Slope classes are screening classes: safe <5 deg, moderate 5-10 deg, unsafe >10 deg."),
        code("slope_summary = pd.read_csv(tables / 'slope_safety_summary.csv')\ndisplay(slope_summary)\nfor name in ['07_dem_elevation_map.png','08_dem_slope_map.png','08b_slope_classification_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## I. Landing Site Suitability\n\nLanding suitability combines low slope, terrain safety proxy, proximity to radar candidates, and an illumination-like proxy placeholder. Future illumination, thermal, and communication layers should replace placeholders."),
        code("landing = pd.read_csv(tables / 'landing_sites.csv')\ndisplay(landing[['site_id','lat','lon','suitability_score','slope_deg','distance_to_candidate_m','safe_wording']])\nfor name in ['09_landing_suitability_map.png','10_top_landing_candidates_overlay.png','landing_score_comparison.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## J. Rover Traversal Planning\n\nThe cost map blocks slope >15 deg and generates shortest, safest, and science-priority conceptual route variants from the top landing candidate to a target candidate patch."),
        code("route_summary = pd.read_csv(routes_dir / 'route_summary.csv')\ndisplay(route_summary[['route_type','status','target_candidate_id','length_m','cost','mean_slope_deg','max_slope_deg']])\nfor name in ['11_rover_route_comparison.png','12_rover_route_overlay.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## K. OHRC Context and Footprint Issue\n\nThe current OHRC products are context-only. They do not directly co-register to the configured Faustini AOI, so they are not used as Faustini hazard/boulder maps."),
        code("ohrc = pd.read_csv(tables / 'ohrc_footprints.csv')\ndisplay(ohrc[['product_id','pixel_resolution_m','lat_min','lat_max','overlaps_faustini_lat','coverage_note']])\ndisplay(Image(filename=str(figdir / '13_ohrc_context_hazard_proxy.png')))\nprint((root / 'reports' / 'OHRC_DATA_DOWNLOAD_NEEDED.md').read_text())"),
        md("## L. Weakly Supervised U-Net Prototype\n\nThe U-Net uses five channels: SAR intensity, LH/LV ratio proxy, texture, polarization imbalance proxy, and rule candidate score. Labels are pseudo-labels from the rule-based mask. Spatial tile split is used, not random pixels."),
        code("metrics = pd.read_csv(tables / 'unet_pseudo_label_metrics.csv')\nhistory = pd.read_csv(tables / 'unet_training_history.csv')\ntiles_df = pd.read_csv(tables / 'unet_tile_inventory.csv')\ndisplay(metrics.T)\ndisplay(history.tail())\ndisplay(tiles_df.groupby('split')[['positive_pixels','positive_fraction']].agg(['count','sum','mean']))\nfor name in ['14_unet_training_curve.png','15_unet_prediction_overlay.png','pseudo_label_distribution.png']:\n    display(Image(filename=str(figdir / name)))\nprint('Limitation: these metrics measure agreement with pseudo-labels, not true ice detection accuracy.')"),
        md("## M. Combined Decision Map\n\nThe combined map overlays radar candidates, unsafe slope/rough terrain proxy, preliminary landing candidates, and conceptual route variants. This is a decision-support view, not a final mission product."),
        code("display(Image(filename=str(figdir / '16_combined_decision_map.png')))"),
        md("## N. Evaluation-Style Summary\n\nBecause there is no ground truth, this section reports screening and agreement statistics only: candidate area percentage, patch count, safe slope percentage, route cost comparison, and pseudo-label agreement."),
        code("coverage_status = pd.read_csv(tables / 'data_coverage_status.csv')\ndisplay(coverage_status)\ndisplay(candidate_summary)\ndisplay(slope_summary)\ndisplay(route_summary[['route_type','length_m','cost','mean_slope_deg','max_slope_deg']])\ndisplay(metrics[['pseudo_iou','pseudo_dice','prediction_fraction','pseudo_label_fraction']])"),
        md("## O. Reproducibility\n\nRun everything from the project root with:\n\n```powershell\n$env:PYTHONPATH='src'\npython -m lunar_icenav.cli run --config configs/pipeline.json\npython -m lunar_icenav.cli notebook --config configs/pipeline.json\n```\n\nUse notebook outputs as research artifacts and validation inputs until independent data layers are added."),
    ]
    nb.cells = research_notebook_cells(root_cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def research_notebook_cells(root_cell: str) -> list[Any]:
    return [
        md("# LunaQuest / Lunar IceNav Research Workflow\n\nResearch notebook for BAH 2026 Problem Statement 8. The workflow produces radar-based candidate patch screening, candidate patch scientific review, preliminary landing candidates, conceptual rover routes, weak ML pseudo-label experiments, and validation reports. Outputs require independent validation."),
        md("## A. Problem and Research Objective\n\nGoal: detect radar-based candidate subsurface ice regions, estimate candidate patch extent, identify preliminary safe landing candidates, plan conceptual rover navigation paths, and evaluate model reliability with proxy/scientific consistency metrics."),
        code(root_cell),
        md("## B. Region Validation and AOI\n\nConfigured AOI: lat -87.8 to -86.9, lon 80 to 85 E. Primary SAR selection is based on full AOI coverage, while partial/no-overlap products are supporting or excluded."),
        code("import pandas as pd\nfrom IPython.display import display, Image\nimport json\nmanifest = json.loads((root / 'reports' / 'run_manifest.json').read_text())\nprint('AOI:', manifest['aoi'])\ndisplay(pd.read_csv(tables / 'sar_product_selection_reason.csv')[['product_id','coverage_fraction','selected_for_main_map','selection_reason']])\ndisplay(Image(filename=str(figdir / 'region_validation_map.png')))"),
        md("## C. Data Inventory and Product Selection\n\nInventory and coverage tables document what was available, what was selected, and what remains context-only."),
        code("inv = pd.read_csv(tables / 'product_inventory.csv')\ndisplay(inv['product_type'].value_counts().reset_index(name='rows'))\ndisplay(pd.read_csv(tables / 'data_coverage_status.csv'))\ndisplay(Image(filename=str(figdir / '01_dataset_inventory_chart.png')))\ndisplay(Image(filename=str(figdir / '02_selected_sar_coverage.png')))"),
        md("## D. SAR Feature Extraction\n\nFeatures include SAR log intensity, LH/LV ratio proxy, LV/LH ratio proxy, polarization imbalance proxy, local mean, local standard deviation, texture roughness, candidate score, and threshold uncertainty. True CPR/DOP are not claimed."),
        code("sar_meta = pd.read_csv(tables / 'selected_sar_metadata.csv')\ndisplay(sar_meta.T)\nfeatures = __import__('numpy').load(root / 'outputs' / 'features' / 'sar_feature_stack.npz')\nprint('Feature arrays:', list(features.keys()))\ndisplay(Image(filename=str(figdir / '04_sar_feature_panel.png')))"),
        md("## E. Candidate Mask Generation\n\nThe candidate map is generated before landing analysis. It uses SAR proxy thresholds, connected components, and threshold sensitivity checks."),
        code("candidate_summary = pd.read_csv(tables / 'candidate_patch_summary.csv')\nthresh = pd.read_csv(tables / 'threshold_sensitivity.csv')\ndisplay(candidate_summary)\ndisplay(thresh)\ndisplay(Image(filename=str(figdir / 'ice_candidate_detection_map.png')))\ndisplay(Image(filename=str(figdir / 'threshold_sensitivity_curve.png')))"),
        md("## F. Candidate Patch Scientific Review\n\nEach candidate patch is evaluated for extent, equivalent candidate patch diameter, score, ratio proxy, texture, slope context, threshold stability, uncertainty, confidence, landing proximity, and route accessibility."),
        code("review = pd.read_csv(tables / 'candidate_scientific_review.csv')\ndisplay(review.head(12))\nfor name in ['ice_candidate_confidence_map.png','ice_candidate_ranking_chart.png','candidate_area_vs_score_scatter.png','candidate_diameter_distribution.png','candidate_score_vs_slope.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## G. Candidate Patch Extent and Resource Scenario Estimation\n\nEquivalent candidate patch diameter is an area-derived screening extent. Scenario-based resource volume is a planning case only if a patch is later validated."),
        code("resource = pd.read_csv(tables / 'resource_scenario_estimates.csv')\ndisplay(resource.head(18))\ndisplay(Image(filename=str(figdir / 'resource_scenario_bar_chart.png')))"),
        md("## H. DEM/Slope Terrain Safety\n\nDEM-derived slope classes support landing and traverse screening. They are not a complete hazard analysis."),
        code("slope_summary = pd.read_csv(tables / 'slope_safety_summary.csv')\ndisplay(slope_summary)\nfor name in ['07_dem_elevation_map.png','08_dem_slope_map.png','08b_slope_classification_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## I. Landing Site Model\n\nLanding scoring uses the candidate map as input, then searches for preliminary landing candidates near candidate patches while avoiding the candidate mask and risky/steep terrain."),
        code("landing = pd.read_csv(tables / 'landing_site_evaluation.csv')\ndisplay(landing)\nfor name in ['09_landing_suitability_map.png','landing_score_components.png','landing_candidate_decision_map.png','nearest_landing_to_candidate_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## J. Rover Navigation Model\n\nThe rover planner generates shortest, safest, science-priority, and energy-efficient conceptual route variants, then evaluates slope exposure, risk proxy, science reward, and energy proxy."),
        code("rover = pd.read_csv(tables / 'rover_navigation_evaluation.csv')\ndisplay(rover[['route_type','target_candidate_id','start_landing_site_id','length_m','total_cost','percent_route_under_5deg','percent_route_above_10deg','science_reward_score','energy_cost_proxy','traverse_risk_score']])\nfor name in ['rover_route_decision_map.png','rover_route_slope_profile.png','rover_route_risk_profile.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## K. U-Net Weakly Supervised Experiment\n\nThe U-Net and baselines are evaluated against rule-based pseudo-labels only. Metrics are pseudo-label agreement, not independently validated ice detection accuracy."),
        code("metrics = pd.read_csv(tables / 'unet_pseudo_label_metrics.csv')\nexperiments = pd.read_csv(tables / 'model_experiment_comparison.csv')\ndisplay(metrics.T)\ndisplay(experiments)\nfor name in ['unet_training_curve.png','model_experiment_comparison.png','unet_prediction_overlay.png','unet_error_map_against_pseudolabel.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## L. Model Evaluation Summary\n\nThis section links the candidate map, patch confidence, landing dependency, rover navigation, and weak ML experiments into one validation-oriented summary."),
        code("print((root / 'reports' / 'MODEL_EVALUATION_REPORT.md').read_text()[:5000])"),
        md("## M. Technical Limitations\n\nKey limitations include no direct water-ice confirmation claim, no true CPR/DOP claim yet, context-only OHRC, preliminary landing candidates, conceptual rover routes, and pseudo-label-only ML agreement."),
        code("print((root / 'reports' / 'TECHNICAL_LIMITATIONS.md').read_text())"),
        md("## N. Next Data to Download\n\nThe next data downloads should improve validation, hazard analysis, and physics-based interpretation."),
        code("print((root / 'reports' / 'NEXT_DATA_TO_DOWNLOAD.md').read_text())"),
    ]


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)
