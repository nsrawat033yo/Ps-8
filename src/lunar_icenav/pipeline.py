from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import rasterio
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
from lunar_icenav.preprocessing.aoi import map_to_lonlat
from lunar_icenav.preprocessing.dem import evaluate_tmc_dtm_coverage, read_dem_aoi, read_tmc_dtm_aoi
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

    reference_inventory = build_research_reference_inventory(root)
    reference_inventory.to_csv(paths["tables"] / "research_reference_inventory.csv", index=False)
    research_traceability = build_research_method_traceability(reference_inventory)
    research_traceability.to_csv(paths["tables"] / "research_method_traceability.csv", index=False)

    inventory = discover_products(root)
    inventory.to_csv(paths["tables"] / "product_inventory.csv", index=False)
    inventory.to_csv(paths["metadata"] / "product_inventory.csv", index=False)
    save_inventory_chart(inventory, paths["figures"] / "01_dataset_inventory_chart.png")

    pair, sar_scores = select_sar_pair(root, aoi)
    sar_score_df = pd.DataFrame([{k: str(v) if isinstance(v, Path) else v for k, v in s.items()} for s in sar_scores])
    sar_score_df.to_csv(paths["tables"] / "sar_aoi_overlap.csv", index=False)
    save_coverage_map(sar_score_df, paths["figures"] / "02_selected_sar_coverage.png", pair["product_id"])

    ohrc_fp = inspect_ohrc_footprints(root, aoi)
    ohrc_fp.to_csv(paths["tables"] / "ohrc_footprints.csv", index=False)
    tmc_coverage = evaluate_tmc_dtm_coverage(root, aoi)
    tmc_coverage.to_csv(paths["tables"] / "tmc_aoi_overlap.csv", index=False)
    coverage_validation = build_data_coverage_validation(sar_score_df, ohrc_fp, tmc_coverage)
    coverage_validation.to_csv(paths["tables"] / "data_coverage_validation.csv", index=False)
    usable_datasets = build_usable_dataset_selection(coverage_validation)
    usable_datasets.to_csv(paths["tables"] / "usable_dataset_selection.csv", index=False)

    sar = read_sar_aoi(pair, aoi)
    selected_metadata = selected_sar_metadata(pair, sar)
    pd.DataFrame([selected_metadata]).to_csv(paths["tables"] / "selected_sar_metadata.csv", index=False)

    features = sar_feature_stack(sar["lh"], sar["lv"])
    lon_grid, lat_grid = build_lon_lat_grids(sar["transform"], sar["crs_wkt"], features["valid"].shape)
    aoi_pixel_mask = build_pixel_aoi_mask_from_lonlat(lon_grid, lat_grid, aoi)
    np.save(paths["masks"] / "aoi_pixel_mask.npy", aoi_pixel_mask)
    features = apply_aoi_mask_to_features(features, aoi_pixel_mask)
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
    threshold_sensitivity, stability_map = build_threshold_sensitivity(features, config, sar["transform"], candidate_mask)
    threshold_sensitivity.to_csv(paths["tables"] / "threshold_sensitivity.csv", index=False)
    patches = summarize_patches(candidate_mask, features, sar["transform"], sar["crs_wkt"], sar["product_id"])
    save_mask_png(candidate_mask, paths["masks"] / "candidate_mask.png")
    np.save(paths["masks"] / "candidate_mask.npy", candidate_mask)
    np.save(paths["masks"] / "candidate_stability.npy", stability_map)

    dem = read_dem_aoi(root, aoi, target_shape=candidate_mask.shape)
    tmc_dem = read_tmc_dtm_aoi(root, aoi, target_shape=candidate_mask.shape)
    if tmc_dem.get("available"):
        tmc_coverage = tmc_dem["coverage"]
        tmc_coverage.to_csv(paths["tables"] / "tmc_aoi_overlap.csv", index=False)
        coverage_validation = build_data_coverage_validation(sar_score_df, ohrc_fp, tmc_coverage)
        coverage_validation.to_csv(paths["tables"] / "data_coverage_validation.csv", index=False)
        usable_datasets = build_usable_dataset_selection(coverage_validation)
        usable_datasets.to_csv(paths["tables"] / "usable_dataset_selection.csv", index=False)
        elevation = tmc_dem.get("elevation")
        slope = tmc_dem.get("slope_deg")
        terrain_source = f"TMC-2 DTM: {tmc_dem.get('selected_product_id')}"
    else:
        elevation = dem.get("elevation")
        slope = dem.get("slope_deg")
        terrain_source = "LOLA/LDEM/LDSM fallback"
    tmc_lola_slope_comparison = build_tmc2_vs_lola_slope_comparison(tmc_dem, dem)
    tmc_lola_slope_comparison.to_csv(paths["tables"] / "tmc2_vs_lola_slope_comparison.csv", index=False)
    terrain_roughness = compute_terrain_roughness(elevation, slope)
    hillshade = compute_hillshade(elevation)
    uncertainty = candidate_uncertainty(features["candidate_score"], thresholds["score_threshold"], features["valid"])
    psr_proxy = build_psr_stability_proxy(lat_grid, slope, hillshade, features["valid"])
    np.save(paths["masks"] / "psr_stability_proxy.npy", psr_proxy)
    save_probability_geotiff(psr_proxy, sar["profile"], paths["rasters"] / "psr_stability_proxy.tif", "Latitude + terrain-shadow PSR stability proxy; approximate validation layer")
    ice_probability, ice_probability_layers = build_ice_probability_map(features, terrain_roughness, psr_proxy)
    ice_confidence, shallow_likelihood, deep_likelihood, depth_class = build_ice_confidence_depth_layers(features, ice_probability, terrain_roughness, slope)
    np.save(paths["masks"] / "ice_probability_map.npy", ice_probability)
    np.save(paths["masks"] / "ice_confidence_score.npy", ice_confidence)
    np.save(paths["masks"] / "shallow_ice_likelihood.npy", shallow_likelihood)
    np.save(paths["masks"] / "deep_ice_likelihood.npy", deep_likelihood)
    np.save(paths["masks"] / "depth_likelihood_class.npy", depth_class)
    save_probability_geotiff(ice_probability, sar["profile"], paths["rasters"] / "ice_probability_map.tif")
    save_probability_geotiff(ice_confidence, sar["profile"], paths["rasters"] / "ice_confidence_score.tif", "Radar/terrain confidence score; screening only")
    patches = enrich_candidate_patches_with_slope(patches, candidate_mask, slope)
    patches = enrich_candidate_patches_with_probability(
        patches,
        candidate_mask,
        ice_probability,
        terrain_roughness,
        ice_confidence,
        shallow_likelihood,
        deep_likelihood,
        psr_proxy,
        ice_probability_layers.get("components", {}),
    )
    patches = enrich_candidate_patches_with_scientific_metrics(patches, candidate_mask, features, stability_map, uncertainty)
    patches, refined_candidate_mask = refine_candidate_patches(patches, candidate_mask)
    patches.to_csv(paths["tables"] / "candidate_patches.csv", index=False)
    build_candidate_patch_table(patches).to_csv(paths["tables"] / "candidate_patch_table.csv", index=False)
    np.save(paths["masks"] / "refined_candidate_mask.npy", refined_candidate_mask)
    save_mask_png(refined_candidate_mask, paths["masks"] / "refined_candidate_mask.png")
    if elevation is not None:
        np.save(feature_dir / "dem_elevation_resampled.npy", elevation)
    if slope is not None:
        np.save(feature_dir / "dem_slope_resampled.npy", slope)
    if terrain_roughness is not None:
        np.save(feature_dir / "terrain_roughness_resampled.npy", terrain_roughness)

    planning_features = dict(features)
    planning_features["candidate_score"] = ice_probability.astype("float32")
    candidate_mask_for_planning = refined_candidate_mask if refined_candidate_mask.any() else candidate_mask

    landing_score, landing_sites, landing_layers = suitability_map(candidate_mask_for_planning, planning_features, slope, sar["transform"], sar["crs_wkt"], config)
    landing_sites = enrich_landing_sites_with_fuzzy_components(landing_sites, landing_layers, candidate_mask, slope, aoi)
    landing_sites = add_landing_context_columns(landing_sites)
    landing_sites = attach_nearest_candidate_context(landing_sites, patches)
    landing_sites.to_csv(paths["tables"] / "landing_sites.csv", index=False)
    np.save(paths["masks"] / "landing_suitability.npy", landing_score)

    route_target_mask, route_target_candidate_id = build_route_target_mask(patches, candidate_mask, candidate_mask_for_planning)
    routes, route_summary = plan_routes(
        planning_features,
        slope,
        landing_sites,
        route_target_mask,
        sar["transform"],
        sar["crs_wkt"],
        config,
        target_candidate_id=route_target_candidate_id,
    )
    rover_navigation = build_rover_navigation_evaluation(routes, route_summary, planning_features, slope, sar["transform"])
    route_summary = rover_navigation.copy()
    landing_sites = attach_landing_route_accessibility(landing_sites, route_summary)
    landing_sites.to_csv(paths["tables"] / "landing_sites.csv", index=False)
    landing_site_evaluation = build_landing_site_evaluation(landing_sites)
    landing_site_evaluation.to_csv(paths["tables"] / "landing_site_evaluation.csv", index=False)
    fuzzy_landing_scores = build_fuzzy_landing_site_scores(landing_sites)
    fuzzy_landing_scores.to_csv(paths["tables"] / "fuzzy_landing_site_scores.csv", index=False)
    landing_crater_boundary = build_landing_crater_boundary_check(landing_sites, candidate_mask, slope, aoi)
    landing_crater_boundary.to_csv(paths["tables"] / "landing_crater_boundary_check.csv", index=False)
    route_summary.to_csv(paths["routes"] / "route_summary.csv", index=False)
    rover_navigation.to_csv(paths["tables"] / "rover_navigation_evaluation.csv", index=False)
    rover_traversal = build_rover_traversal_simulation(routes, route_summary, planning_features, slope, sar["transform"])
    rover_traversal.to_csv(paths["tables"] / "rover_traversal_simulation.csv", index=False)
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
    if slope_stats:
        slope_stats["terrain_source"] = terrain_source
        slope_stats["tmc_selected_product_id"] = tmc_dem.get("selected_product_id", "") if isinstance(tmc_dem, dict) else ""
    pd.DataFrame([slope_stats]).to_csv(paths["tables"] / "slope_safety_summary.csv", index=False)
    data_status = build_data_coverage_status(inventory, sar_score_df, ohrc_fp, pair, aoi, tmc_coverage)
    data_status.to_csv(paths["tables"] / "data_coverage_status.csv", index=False)
    validation_layer_status = build_validation_layer_status(inventory, ohrc_fp, dem, pair, tmc_dem)
    validation_layer_status.to_csv(paths["tables"] / "validation_layer_status.csv", index=False)
    candidate_external_validation = build_candidate_validation_against_external_layers(candidate_scientific_review, validation_layer_status)
    candidate_external_validation.to_csv(paths["tables"] / "candidate_validation_against_external_layers.csv", index=False)
    hazard_summary = build_hazard_proxy_summary(features, terrain_roughness, ohrc_fp)
    hazard_summary.to_csv(paths["tables"] / "hazard_proxy_summary.csv", index=False)

    hazard_mask = robust_normalize(features["texture"], features["valid"]) > np.nanquantile(robust_normalize(features["texture"], features["valid"])[features["valid"]], 0.90)
    unsafe_slope_mask = (features["valid"].astype(bool) & (slope > 10)) if slope is not None else np.zeros_like(candidate_mask, dtype=bool)
    landing_top_mask = landing_score >= np.nanquantile(landing_score[np.isfinite(landing_score)], 0.95)

    save_research_backed_pipeline_diagram(paths["figures"] / "research_backed_pipeline_diagram.png")
    save_region_validation_map(sar_score_df, pair, aoi, paths["figures"] / "region_validation_map.png")
    save_scalar_map(base, paths["figures"] / "03_sar_quicklook.png", f"SAR SRI intensity quicklook - {sar['product_id']}", "gray", "log intensity proxy", SAFE_NOTE)
    save_feature_panel(base, features, candidate_mask, paths["figures"] / "04_sar_feature_panel.png", f"SAR feature panel - {sar['product_id']} - {aoi['name']}")
    save_ice_probability_map(base, ice_probability, candidate_mask_for_planning, patches, paths["figures"] / "ice_probability_map.png")
    save_scalar_map(psr_proxy, paths["figures"] / "psr_stability_proxy_map.png", "Approximate PSR Stability Proxy", "cividis", "PSR proxy score", "Approximate latitude + terrain-shadow proxy; replace with real illumination/PSR data when available.", vmin=0, vmax=1)
    save_scalar_map(ice_confidence, paths["figures"] / "ice_confidence_score_map.png", "Ice Candidate Confidence Score", "viridis", "confidence score", "Confidence combines high radar score, low roughness, and low slope; screening only.", vmin=0, vmax=1)
    save_depth_likelihood_map(shallow_likelihood, deep_likelihood, depth_class, paths["figures"] / "depth_likelihood_map.png")
    save_ice_candidate_detection_map(base, candidate_mask, patches, paths["figures"] / "ice_candidate_detection_map.png")
    save_candidate_confidence_map(base, candidate_mask, patches, paths["figures"] / "ice_candidate_confidence_map.png")
    save_candidate_confidence_map(base, candidate_mask, patches, paths["figures"] / "top_candidate_confidence_map.png")
    save_threshold_sensitivity_curve(threshold_sensitivity, paths["figures"] / "threshold_sensitivity_curve.png")
    save_scalar_map(stability_map, paths["figures"] / "candidate_stability_map.png", "Candidate Stability Across Score Thresholds", "viridis", "Stability score", "Higher values persist across more candidate-score thresholds; validation required.", vmin=0, vmax=1)
    save_radar_roughness_ambiguity_map(base, candidate_mask, patches, paths["figures"] / "radar_roughness_ambiguity_map.png")
    save_candidate_ranking_chart(candidate_scientific_review, paths["figures"] / "ice_candidate_ranking_chart.png")
    save_candidate_area_vs_score(candidate_scientific_review, paths["figures"] / "ice_candidate_area_vs_score.png")
    save_candidate_area_vs_score(candidate_scientific_review, paths["figures"] / "candidate_area_vs_score_scatter.png")
    save_candidate_diameter_distribution(candidate_scientific_review, paths["figures"] / "candidate_diameter_distribution.png")
    save_candidate_score_vs_slope(candidate_scientific_review, paths["figures"] / "candidate_score_vs_slope.png")
    save_candidate_score_vs_roughness(candidate_scientific_review, paths["figures"] / "candidate_score_vs_roughness.png")
    save_top_candidate_review_panel(candidate_scientific_review, paths["figures"] / "top_candidate_review_panel.png")
    save_resource_scenario_bar_chart(resource_scenarios, paths["figures"] / "resource_scenario_bar_chart.png")
    save_overlay(base, paths["figures"] / "05_radar_candidate_overlay.png", f"Radar candidate screening overlay - {sar['product_id']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.45))])
    save_histogram(patches, "area_m2", paths["figures"] / "06_candidate_patch_area_histogram.png", "Candidate Patch Area Distribution", "Area (m2)")
    save_top_candidates_chart(patches, paths["figures"] / "06b_top_candidate_patches.png")
    save_score_distribution(features, paths["figures"] / "06c_candidate_score_distribution.png")
    save_candidate_centroid_map(base, patches, paths["figures"] / "06d_candidate_centroid_map.png")
    save_scalar_map(uncertainty, paths["figures"] / "06e_candidate_uncertainty_map.png", "Candidate Screening Uncertainty Proxy", "cividis", "uncertainty proxy", "High values indicate pixels near the screening threshold; validation required.", vmin=0, vmax=1)
    if elevation is not None:
        save_scalar_map(elevation, paths["figures"] / "07_dem_elevation_map.png", f"DEM elevation context - {aoi['name']}", "terrain", "Elevation layer value", "DEM resampled to SAR AOI for context.")
        save_scalar_map(elevation, paths["figures"] / "dem_elevation_map.png", f"DEM elevation context - {aoi['name']}", "terrain", "Elevation layer value", "DEM resampled to SAR AOI for context.")
    if terrain_roughness is not None:
        save_scalar_map(terrain_roughness, paths["figures"] / "terrain_roughness_map.png", f"Terrain roughness proxy - {aoi['name']}", "magma", "roughness proxy", "Local DEM/slope roughness proxy for landing and traverse screening.", vmin=0, vmax=1)
    if hillshade is not None:
        save_scalar_map(hillshade, paths["figures"] / "hillshade_map.png", f"DEM hillshade context - {aoi['name']}", "gray", "hillshade", "Hillshade derived from DEM elevation for terrain interpretation.", vmin=0, vmax=1)
    if slope is not None:
        save_slope_map(slope, paths["figures"] / "08_dem_slope_map.png", f"DEM slope map - {aoi['name']}")
        save_slope_map(slope, paths["figures"] / "dtm_slope_map.png", f"DTM/DEM slope map - {aoi['name']}")
        save_slope_classification(slope, paths["figures"] / "08b_slope_classification_map.png")
        pd.DataFrame([slope_stats]).to_csv(paths["tables"] / "slope_safety_summary.csv", index=False)
    save_tmc2_vs_lola_slope_difference(tmc_dem, dem, paths["figures"] / "tmc2_vs_lola_slope_difference.png")
    save_scalar_map(landing_score, paths["figures"] / "09_landing_suitability_map.png", f"Preliminary landing suitability heatmap - {aoi['name']}", "YlGn", "Suitability score", "Preliminary landing candidates should be near candidate patches, but outside hazardous/steep terrain.", vmin=0, vmax=1)
    save_scalar_map(landing_score, paths["figures"] / "fuzzy_landing_score_map.png", f"Fuzzy landing score map - {aoi['name']}", "YlGn", "Fuzzy score", "Research-backed preliminary landing candidates avoid candidate-mask interiors and unsafe terrain.", vmin=0, vmax=1)
    save_overlay(base, paths["figures"] / "10_top_landing_candidates_overlay.png", f"Top preliminary landing candidates - {aoi['name']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.32)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.28))], points=landing_sites)
    save_landing_score_components_evaluation(landing_site_evaluation, paths["figures"] / "landing_score_components.png")
    save_overlay(base, paths["figures"] / "landing_candidate_decision_map.png", f"Landing candidate decision map - {aoi['name']}", [(candidate_mask, "candidate ice region", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.26))], points=landing_sites, note="Preliminary landing candidates are selected near candidate patches but outside risky terrain; validation required.")
    save_overlay(base, paths["figures"] / "landing_site_map.png", f"Top preliminary landing zones - {aoi['name']}", [(candidate_mask_for_planning, "refined candidate patch", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "unsafe slope / rough proxy", (1.0, 0.20, 0.0, 0.26))], points=landing_sites.head(3), note="Landing zones are preliminary: slope <5 deg preferred, hazard proxy low, near refined radar-based candidate patches; validation required.")
    save_science_justification_overlay(base, ice_probability, candidate_mask_for_planning, patches, landing_sites, routes, route_summary, paths["figures"] / "science_justification_overlay.png")
    save_annotated_probability_map(base, ice_probability, candidate_mask_for_planning, patches, paths["figures"] / "ice_probability_map_annotated.png")
    save_annotated_landing_map(base, candidate_mask_for_planning, unsafe_slope_mask | hazard_mask, patches, landing_sites, paths["figures"] / "landing_site_map_annotated.png")
    save_nearest_landing_to_candidate_map(base, candidate_mask, patches, landing_sites, paths["figures"] / "nearest_landing_to_candidate_map.png")
    save_nearest_landing_to_candidate_map(base, candidate_mask, patches, landing_sites, paths["figures"] / "landing_to_candidate_distance_map.png")
    save_landing_vs_f2_boundary_map(base, candidate_mask, landing_sites, aoi, paths["figures"] / "landing_vs_f2_crater_boundary_map.png")
    save_validation_layer_availability_matrix(validation_layer_status, paths["figures"] / "validation_layer_availability_matrix.png")
    save_hazard_overlay_on_terrain(base, hazard_mask, terrain_roughness, paths["figures"] / "optical_hazard_proxy_map.png", "Hazard proxy from SAR/DEM texture; OHRC is context-only unless co-registered.")
    save_hazard_overlay_on_terrain(base, hazard_mask, terrain_roughness, paths["figures"] / "hazard_overlay_on_terrain.png", "Hazard overlay uses proxy roughness layers available for the configured AOI.")
    save_route_comparison(route_summary, paths["figures"] / "11_rover_route_comparison.png")
    save_route_comparison(route_summary, paths["figures"] / "rover_route_comparison_chart.png")
    save_overlay(base, paths["figures"] / "12_rover_route_overlay.png", f"Conceptual rover route variants - {aoi['name']}", [(candidate_mask, "radar candidate", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "blocked/risky proxy", (1.0, 0.20, 0.0, 0.26))], routes=routes, points=landing_sites.head(1))
    save_overlay(base, paths["figures"] / "rover_route_decision_map.png", f"Rover navigation decision map - {aoi['name']}", [(candidate_mask, "candidate ice region", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "blocked/risky proxy", (1.0, 0.20, 0.0, 0.26))], routes=routes, points=landing_sites.head(1), note="Conceptual rover routes use proxy traversability costs; validation required.")
    save_overlay(base, paths["figures"] / "rover_routes.png", f"Conceptual rover routes to top refined patch - {aoi['name']}", [(candidate_mask_for_planning, "refined candidate patch", (0.0, 0.85, 1.0, 0.36)), (unsafe_slope_mask | hazard_mask, "blocked/risky proxy", (1.0, 0.20, 0.0, 0.26))], routes=routes, points=landing_sites.head(1), note="Route variants compare shortest, safest, science-optimal, and energy-aware proxy costs; operational validation required.")
    save_annotated_route_map(base, candidate_mask_for_planning, unsafe_slope_mask | hazard_mask, landing_sites, routes, route_summary, paths["figures"] / "rover_routes_annotated.png")
    save_route_profile(routes, slope, paths["figures"] / "rover_route_slope_profile.png", "slope_deg", "Route Slope Profile", "Slope (deg)")
    save_route_risk_profile(routes, features, slope, paths["figures"] / "rover_route_risk_profile.png")
    save_rover_energy_profile(rover_traversal, paths["figures"] / "rover_energy_profile.png")
    save_rover_slope_distance_profile(rover_traversal, paths["figures"] / "rover_slope_vs_distance.png")
    save_rover_traversal_steps(base, candidate_mask_for_planning, landing_sites, routes, route_summary, paths["figures"] / "rover_traversal_steps.png")
    save_rover_traversal_animation(base, candidate_mask_for_planning, landing_sites, routes, route_summary, paths["figures"] / "rover_traversal_animation.gif")
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
        "terrain_source": terrain_source,
        "selected_tmc": {
            "product_id": tmc_dem.get("selected_product_id", "") if isinstance(tmc_dem, dict) else "",
            "path": tmc_dem.get("selected_path", "") if isinstance(tmc_dem, dict) else "",
            "coverage_fraction": tmc_dem.get("coverage_fraction", "") if isinstance(tmc_dem, dict) else "",
        },
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
        research_traceability,
        validation_layer_status,
    )
    write_technical_limitations(paths["reports"] / "TECHNICAL_LIMITATIONS.md", aoi)
    write_next_data_to_download(paths["reports"] / "NEXT_DATA_TO_DOWNLOAD.md", aoi)
    write_decision_report(paths["reports"] / "decision_report.txt", aoi, coverage_validation, usable_datasets, tmc_dem, ohrc_fp)
    write_justification_report(paths["reports"] / "justification_report.txt", aoi, patches, landing_sites, route_summary, rover_traversal, ice_probability_layers)
    write_research_paper_method_map(paths["reports"] / "RESEARCH_PAPER_METHOD_MAP.md", research_traceability, reference_inventory)
    write_research_references_used(paths["reports"] / "RESEARCH_REFERENCES_USED.md", reference_inventory)
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


def build_lon_lat_grids(transform, crs_wkt: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(shape, dtype="float32")
    pixel_cols = cols + 0.5
    pixel_rows = rows + 0.5
    xs = transform.c + transform.a * pixel_cols + transform.b * pixel_rows
    ys = transform.f + transform.d * pixel_cols + transform.e * pixel_rows
    return map_to_lonlat(crs_wkt, xs, ys)


def build_pixel_aoi_mask(transform, crs_wkt: str, shape: tuple[int, int], aoi: dict[str, Any]) -> np.ndarray:
    lon, lat = build_lon_lat_grids(transform, crs_wkt, shape)
    return build_pixel_aoi_mask_from_lonlat(lon, lat, aoi)


def build_pixel_aoi_mask_from_lonlat(lon: np.ndarray, lat: np.ndarray, aoi: dict[str, Any]) -> np.ndarray:
    return (
        (lat >= float(aoi["lat_min"]))
        & (lat <= float(aoi["lat_max"]))
        & (lon >= float(aoi["lon_min"]))
        & (lon <= float(aoi["lon_max"]))
    )


def apply_aoi_mask_to_features(features: dict[str, Any], aoi_mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray) and value.shape == aoi_mask.shape:
            if value.dtype == bool:
                out[key] = value & aoi_mask
            else:
                out[key] = np.where(aoi_mask, value, np.nan).astype(value.dtype, copy=False)
        else:
            out[key] = value
    out["aoi_mask"] = aoi_mask.astype(bool)
    out["valid"] = features["valid"].astype(bool) & aoi_mask
    return out


def coverage_class_from_fraction(frac: float) -> str:
    if frac >= 0.999:
        return "FULL"
    if frac > 0:
        return "PARTIAL"
    return "NO COVERAGE"


def build_data_coverage_validation(sar_scores: pd.DataFrame, ohrc_fp: pd.DataFrame, tmc_coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not sar_scores.empty:
        for _, row in sar_scores.iterrows():
            frac = float(row.get("coverage_fraction", 0.0))
            cls = coverage_class_from_fraction(frac)
            rows.append({
                "dataset_type": "SAR/DFSAR SRI",
                "product_id": row.get("product_id", ""),
                "coverage_fraction": frac,
                "coverage_class": cls,
                "usable_for_analysis": cls in {"FULL", "PARTIAL"},
                "selected_for_analysis": bool(str(row.get("coverage_fraction", "")) and cls == "FULL"),
                "path": row.get("lh", ""),
                "reason": "LH/LV SRI pair usable for radar screening" if cls != "NO COVERAGE" else "excluded from Faustini analysis due to no AOI overlap",
            })
    if not ohrc_fp.empty:
        for _, row in ohrc_fp.iterrows():
            frac = float(row.get("coverage_fraction", 0.0))
            cls = str(row.get("coverage_class", coverage_class_from_fraction(frac)))
            rows.append({
                "dataset_type": "OHRC",
                "product_id": row.get("product_id", ""),
                "coverage_fraction": frac,
                "coverage_class": cls,
                "usable_for_analysis": False,
                "selected_for_analysis": False,
                "path": row.get("path", ""),
                "reason": "OHRC footprint does not overlap the configured Faustini AOI; context-only, not used for hazard scoring",
            })
    if not tmc_coverage.empty:
        for _, row in tmc_coverage.iterrows():
            frac = float(row.get("coverage_fraction", 0.0))
            cls = str(row.get("coverage_class", coverage_class_from_fraction(frac)))
            rows.append({
                "dataset_type": "TMC-2 DTM",
                "product_id": row.get("product_id", ""),
                "coverage_fraction": frac,
                "coverage_class": cls,
                "usable_for_analysis": bool(row.get("usable_for_analysis", cls in {"FULL", "PARTIAL"})),
                "selected_for_analysis": bool(row.get("selected_for_terrain", False)),
                "path": row.get("path", ""),
                "reason": row.get("reason", "TMC-2 DTM terrain coverage check"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["coverage_priority"] = out["coverage_class"].map({"FULL": 0, "PARTIAL": 1, "NO COVERAGE": 2}).fillna(3)
        out = out.sort_values(["dataset_type", "coverage_priority", "product_id"]).drop(columns=["coverage_priority"])
    return out


def build_usable_dataset_selection(coverage_validation: pd.DataFrame) -> pd.DataFrame:
    if coverage_validation.empty:
        return pd.DataFrame()
    out = coverage_validation[coverage_validation["usable_for_analysis"].astype(bool)].copy()
    out["analysis_role"] = np.where(
        out["dataset_type"].eq("SAR/DFSAR SRI"),
        "radar candidate screening",
        np.where(out["dataset_type"].eq("TMC-2 DTM"), "terrain slope/roughness", "context only"),
    )
    out["filter_rule"] = "FULL/PARTIAL AOI coverage accepted; OHRC excluded unless AOI overlap exists"
    return out


def build_psr_stability_proxy(
    lat_grid: np.ndarray,
    slope: np.ndarray | None,
    hillshade: np.ndarray | None,
    valid: np.ndarray,
) -> np.ndarray:
    valid_mask = valid.astype(bool) & np.isfinite(lat_grid)
    poleward_score = np.clip((np.abs(lat_grid) - 87.0) / 1.0, 0, 1)
    if slope is not None:
        slope_shadow = np.clip(slope / 15.0, 0, 1)
    else:
        slope_shadow = np.zeros_like(poleward_score, dtype="float32") + 0.35
    if hillshade is not None:
        terrain_shadow = 1.0 - np.clip(hillshade, 0, 1)
    else:
        terrain_shadow = np.zeros_like(poleward_score, dtype="float32") + 0.5
    psr_proxy = (0.58 * poleward_score + 0.22 * terrain_shadow + 0.20 * slope_shadow).astype("float32")
    psr_proxy[~valid_mask] = np.nan
    return np.clip(psr_proxy, 0, 1).astype("float32")


def build_ice_probability_map(
    features: dict[str, np.ndarray],
    terrain_roughness: np.ndarray | None,
    psr_proxy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = features["valid"].astype(bool)
    intensity = robust_normalize(features["intensity"], valid)
    ratio = robust_normalize(features["cpr_style_ratio_proxy"], valid)
    sar_texture = robust_normalize(features["texture"], valid)
    if terrain_roughness is not None:
        terrain = robust_normalize(terrain_roughness, np.isfinite(terrain_roughness))
        roughness = np.clip(0.55 * sar_texture + 0.45 * terrain, 0, 1)
        roughness_source = "SAR texture + TMC/terrain roughness"
    else:
        roughness = sar_texture
        roughness_source = "SAR texture only"
    roughness_suitability = 1.0 - roughness
    if psr_proxy is None:
        psr_score = np.where(valid, 0.5, 0.0).astype("float32")
        psr_status = "PSR layer unavailable; neutral 0.5 placeholder used and reported as assumption"
        psr_weight_name = "psr_proximity_or_neutral_placeholder"
    else:
        psr_score = robust_normalize(psr_proxy, valid & np.isfinite(psr_proxy))
        psr_status = "Approximate PSR stability proxy used: poleward latitude + terrain shadow/slope context; not an illumination model"
        psr_weight_name = "psr_stability_proxy"
    probability = (
        0.30 * ratio
        + 0.27 * intensity
        + 0.23 * roughness_suitability
        + 0.20 * psr_score
    ).astype("float32")
    probability[~valid] = np.nan
    components = {
        "ratio_contribution": (0.30 * ratio).astype("float32"),
        "intensity_contribution": (0.27 * intensity).astype("float32"),
        "roughness_suitability_contribution": (0.23 * roughness_suitability).astype("float32"),
        "psr_proxy_contribution": (0.20 * psr_score).astype("float32"),
    }
    for arr in components.values():
        arr[~valid] = np.nan
    return probability, {
        "weights": {
            "cpr_style_ratio_proxy": 0.30,
            "sar_backscatter_intensity": 0.27,
            "roughness_penalty": 0.23,
            psr_weight_name: 0.20,
        },
        "components": components,
        "roughness_source": roughness_source,
        "psr_status": psr_status,
        "interpretation": "candidate screening probability; compositional validation required before any ice interpretation",
    }


def build_ice_confidence_depth_layers(
    features: dict[str, np.ndarray],
    ice_probability: np.ndarray,
    terrain_roughness: np.ndarray | None,
    slope: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = features["valid"].astype(bool)
    radar = robust_normalize(features["candidate_score"], valid)
    if terrain_roughness is not None:
        roughness = robust_normalize(terrain_roughness, valid & np.isfinite(terrain_roughness))
    else:
        roughness = robust_normalize(features["texture"], valid)
    if slope is not None:
        slope_safety = 1.0 - np.clip(slope / 8.0, 0, 1)
    else:
        slope_safety = np.where(valid, 0.5, np.nan)
    low_roughness = 1.0 - np.clip(roughness, 0, 1)
    confidence = (0.50 * radar + 0.25 * low_roughness + 0.25 * slope_safety).astype("float32")
    moderate_roughness = 1.0 - np.clip(np.abs(roughness - 0.45) / 0.45, 0, 1)
    shallow = (0.65 * radar + 0.35 * low_roughness).astype("float32")
    deep = (0.65 * radar + 0.35 * moderate_roughness).astype("float32")
    depth_class = np.zeros(radar.shape, dtype="uint8")
    depth_class[(valid & np.isfinite(shallow) & (shallow >= deep) & (shallow >= 0.55))] = 1
    depth_class[(valid & np.isfinite(deep) & (deep > shallow) & (deep >= 0.55))] = 2
    depth_class[(valid & (depth_class == 0))] = 3
    for arr in [confidence, shallow, deep]:
        arr[~valid] = np.nan
    return confidence, shallow, deep, depth_class


def save_probability_geotiff(arr: np.ndarray, profile: dict[str, Any], path: Path, description: str = "Radar-based candidate screening probability; validation required") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "nodata": -9999.0,
        "compress": "deflate",
    })
    data = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)
        dst.update_tags(1, description=description)


def enrich_candidate_patches_with_probability(
    patches: pd.DataFrame,
    candidate_mask: np.ndarray,
    ice_probability: np.ndarray,
    terrain_roughness: np.ndarray | None,
    ice_confidence: np.ndarray | None = None,
    shallow_likelihood: np.ndarray | None = None,
    deep_likelihood: np.ndarray | None = None,
    psr_proxy: np.ndarray | None = None,
    probability_components: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    if patches.empty:
        return patches
    labels, _ = ndi.label(candidate_mask)
    out = patches.copy()
    mean_prob: list[float] = []
    max_prob: list[float] = []
    mean_rough: list[float] = []
    mean_confidence: list[float] = []
    mean_shallow: list[float] = []
    mean_deep: list[float] = []
    mean_psr: list[float] = []
    depth_classes: list[str] = []
    component_values: dict[str, list[float]] = {k: [] for k in (probability_components or {})}
    for candidate_id in out["candidate_id"]:
        label = candidate_label_from_id(candidate_id)
        mask = labels == label
        vals = ice_probability[mask & np.isfinite(ice_probability)]
        mean_prob.append(float(np.nanmean(vals)) if vals.size else np.nan)
        max_prob.append(float(np.nanmax(vals)) if vals.size else np.nan)
        if terrain_roughness is not None:
            rough_vals = terrain_roughness[mask & np.isfinite(terrain_roughness)]
            mean_rough.append(float(np.nanmean(rough_vals)) if rough_vals.size else np.nan)
        else:
            mean_rough.append(np.nan)
        conf_vals = values_for_patch(ice_confidence, mask)
        shallow_vals = values_for_patch(shallow_likelihood, mask)
        deep_vals = values_for_patch(deep_likelihood, mask)
        psr_vals = values_for_patch(psr_proxy, mask)
        mean_confidence.append(float(np.nanmean(conf_vals)) if conf_vals.size else np.nan)
        mean_shallow.append(float(np.nanmean(shallow_vals)) if shallow_vals.size else np.nan)
        mean_deep.append(float(np.nanmean(deep_vals)) if deep_vals.size else np.nan)
        mean_psr.append(float(np.nanmean(psr_vals)) if psr_vals.size else np.nan)
        for key, arr in (probability_components or {}).items():
            comp_vals = values_for_patch(arr, mask)
            component_values[key].append(float(np.nanmean(comp_vals)) if comp_vals.size else np.nan)
        if shallow_vals.size and deep_vals.size:
            shallow_mean = float(np.nanmean(shallow_vals))
            deep_mean = float(np.nanmean(deep_vals))
            if max(shallow_mean, deep_mean) < 0.55:
                depth_classes.append("uncertain")
            elif shallow_mean >= deep_mean:
                depth_classes.append("shallow-likelihood")
            else:
                depth_classes.append("deep-likelihood")
        else:
            depth_classes.append("not_evaluated")
    out["mean_ice_probability"] = mean_prob
    out["max_ice_probability"] = max_prob
    out["mean_terrain_roughness"] = mean_rough
    out["ice_confidence_score"] = mean_confidence
    out["shallow_ice_likelihood"] = mean_shallow
    out["deep_ice_likelihood"] = mean_deep
    out["psr_stability_proxy"] = mean_psr
    out["depth_likelihood_class"] = depth_classes
    for key, values in component_values.items():
        out[f"mean_{key}"] = values
    return out


def values_for_patch(arr: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    vals = arr[mask & np.isfinite(arr)]
    return vals.astype(float) if vals.size else np.array([], dtype=float)


def refine_candidate_patches(patches: pd.DataFrame, candidate_mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if patches.empty:
        return patches, candidate_mask
    out = patches.copy()
    slope_safety = 1.0 - np.clip(out.get("mean_slope_deg", pd.Series(8, index=out.index)).astype(float) / 8.0, 0, 1)
    roughness = out.get("mean_terrain_roughness", out.get("mean_dem_roughness_proxy", pd.Series(np.nan, index=out.index))).astype(float)
    roughness = roughness.fillna(roughness.median() if np.isfinite(roughness).any() else 0.5)
    roughness_hazard = np.clip(roughness, 0, 1)
    ice_prob = out.get("mean_ice_probability", out.get("mean_candidate_score", pd.Series(0, index=out.index))).astype(float)
    refined_score = 0.50 * ice_prob + 0.30 * slope_safety + 0.20 * (1.0 - roughness_hazard)
    high_ambiguity = (
        (out["mean_candidate_score"].astype(float) >= 0.75)
        & (
            out["roughness_ambiguity_risk"].astype(str).isin(["Medium", "High"])
            | (roughness_hazard >= 0.35)
            | (out.get("mean_slope_deg", pd.Series(0, index=out.index)).astype(float) >= 10)
        )
    )
    out["slope_safety_score"] = slope_safety
    out["roughness_hazard_score"] = roughness_hazard
    out["refined_candidate_score"] = refined_score
    out["refinement_status"] = np.where(high_ambiguity, "excluded_high_ambiguity", "kept_for_planning")
    kept = out["refinement_status"].eq("kept_for_planning")
    kept_order = out.loc[kept].sort_values("refined_candidate_score", ascending=False).index
    out["refined_rank"] = np.nan
    out.loc[kept_order, "refined_rank"] = np.arange(1, len(kept_order) + 1)
    labels, _ = ndi.label(candidate_mask)
    refined_mask = np.zeros_like(candidate_mask, dtype=bool)
    for candidate_id in out.loc[kept, "candidate_id"]:
        refined_mask |= labels == candidate_label_from_id(candidate_id)
    out["_refinement_sort"] = np.where(kept, 0, 1)
    out = (
        out.sort_values(["_refinement_sort", "refined_rank", "validation_priority_rank"], na_position="last")
        .drop(columns=["_refinement_sort"])
        .reset_index(drop=True)
    )
    return out, refined_mask


def build_candidate_patch_table(patches: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate_id", "refinement_status", "refined_rank", "validation_priority_rank",
        "mean_ice_probability", "max_ice_probability", "ice_confidence_score",
        "depth_likelihood_class", "shallow_ice_likelihood", "deep_ice_likelihood",
        "psr_stability_proxy", "mean_ratio_contribution", "mean_intensity_contribution",
        "mean_roughness_suitability_contribution", "mean_psr_proxy_contribution",
        "mean_candidate_score",
        "area_m2", "equivalent_candidate_patch_diameter_m", "centroid_lat", "centroid_lon",
        "mean_slope_deg", "max_slope_deg", "slope_safety_score",
        "mean_texture", "mean_terrain_roughness", "roughness_hazard_score",
        "roughness_ambiguity_risk", "threshold_stability_class", "confidence_level",
        "confidence_score_after_roughness_penalty", "nearest_landing_site_id",
        "distance_to_nearest_landing_candidate_m", "route_accessible_yes_no",
        "reason_for_confidence",
    ]
    return patches[[c for c in cols if c in patches.columns]].copy()


def build_route_target_mask(
    patches: pd.DataFrame,
    original_candidate_mask: np.ndarray,
    fallback_candidate_mask: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    if patches.empty:
        return fallback_candidate_mask, None
    candidates = patches.copy()
    if "refinement_status" in candidates:
        kept = candidates[candidates["refinement_status"].astype(str).eq("kept_for_planning")]
        if not kept.empty:
            candidates = kept
    sort_cols = [c for c in ["refined_rank", "validation_priority_rank"] if c in candidates.columns]
    top = candidates.sort_values(sort_cols, na_position="last").iloc[0] if sort_cols else candidates.iloc[0]
    target_id = str(top["candidate_id"])
    labels, _ = ndi.label(original_candidate_mask)
    mask = labels == candidate_label_from_id(target_id)
    if not mask.any():
        return fallback_candidate_mask, target_id
    return mask, target_id


def build_tmc2_vs_lola_slope_comparison(tmc_dem: dict[str, Any], dem: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(tmc_dem, dict) or not tmc_dem.get("available"):
        return pd.DataFrame([{
            "comparison": "TMC-2 DTM vs LOLA/LDEM/LDSM slope",
            "status": "not_available",
            "reason": "No AOI-intersecting TMC-2 DTM was selected for terrain analysis.",
        }])
    tmc_slope = tmc_dem.get("slope_deg")
    lola_slope = dem.get("slope_deg") if isinstance(dem, dict) else None
    if tmc_slope is None or lola_slope is None:
        return pd.DataFrame([{
            "comparison": "TMC-2 DTM vs LOLA/LDEM/LDSM slope",
            "status": "not_available",
            "reason": "Both TMC-2 and LOLA/LDEM/LDSM slope layers are required for comparison.",
        }])
    mask = np.isfinite(tmc_slope) & np.isfinite(lola_slope)
    if not mask.any():
        return pd.DataFrame([{
            "comparison": "TMC-2 DTM vs LOLA/LDEM/LDSM slope",
            "status": "not_available",
            "reason": "No overlapping finite slope pixels after AOI clipping/resampling.",
        }])
    diff = tmc_slope[mask].astype(float) - lola_slope[mask].astype(float)
    return pd.DataFrame([{
        "comparison": "TMC-2 DTM vs LOLA/LDEM/LDSM slope",
        "status": "available",
        "tmc2_product_id": tmc_dem.get("selected_product_id", ""),
        "tmc2_pixel_size_m": tmc_dem.get("pixel_size_m", np.nan),
        "overlap_pixels": int(mask.sum()),
        "tmc2_mean_slope_deg": float(np.nanmean(tmc_slope[mask])),
        "lola_ldem_ldsm_mean_slope_deg": float(np.nanmean(lola_slope[mask])),
        "mean_difference_deg_tmc2_minus_lola": float(np.nanmean(diff)),
        "median_abs_difference_deg": float(np.nanmedian(np.abs(diff))),
        "p95_abs_difference_deg": float(np.nanpercentile(np.abs(diff), 95)),
        "interpretation": "terrain-model sensitivity check for planning; not an accuracy claim without co-registration validation",
    }])


def build_research_reference_inventory(root: Path) -> pd.DataFrame:
    downloads = root.parent
    expected = [
        {
            "filename": "remotesensing-14-04863.pdf",
            "paper_title": "Selection of Lunar South Pole Landing Site Based on Constructing and Analyzing Fuzzy Cognitive Maps",
            "source": "Remote Sensing 2022",
            "duplicate_group": "remote_sensing_fcm_2022",
            "duplicate_status": "unique_used",
        },
        {
            "filename": "1-s2.0-S2095927325001999-main.pdf",
            "paper_title": "Upper limit of ice content at the lunar south pole as revealed by the Earth-based SYISR-FAST bistatic radar system",
            "source": "Science Bulletin 2025",
            "duplicate_group": "science_bulletin_sysisr_fast_2025",
            "duplicate_status": "unique_used",
        },
        {
            "filename": "1-s2.0-S2095927325001999-main (1).pdf",
            "paper_title": "Upper limit of ice content at the lunar south pole as revealed by the Earth-based SYISR-FAST bistatic radar system",
            "source": "Science Bulletin 2025",
            "duplicate_group": "science_bulletin_sysisr_fast_2025",
            "duplicate_status": "duplicate_ignored",
        },
        {
            "filename": "pnas.1802345115.sapp.pdf",
            "paper_title": "Supporting information for direct evidence of surface-exposed water ice in the lunar polar regions",
            "source": "PNAS supporting information",
            "duplicate_group": "pnas_surface_exposed_ice_si",
            "duplicate_status": "unique_used",
        },
        {
            "filename": "1-s2.0-S0094576525004898-main.pdf",
            "paper_title": "Path planning algorithm for a South Pole lunar rover mission",
            "source": "Acta Astronautica 2025",
            "duplicate_group": "acta_rover_path_planning_2025",
            "duplicate_status": "unique_used",
        },
    ]
    rows = []
    canonical_by_group: dict[str, str] = {}
    for row in expected:
        path = downloads / row["filename"]
        exists = bool(path.exists())
        digest = file_sha256_12(path) if exists else ""
        canonical = canonical_by_group.setdefault(row["duplicate_group"], row["filename"])
        duplicate_status = row["duplicate_status"]
        used = duplicate_status == "unique_used" and exists and canonical == row["filename"]
        if canonical != row["filename"]:
            duplicate_status = "duplicate_ignored"
            used = False
        rows.append({
            **row,
            "duplicate_status": duplicate_status,
            "canonical_reference_filename": canonical,
            "duplicate_detection_basis": "duplicate group/title match; hash recorded for audit",
            "sha256_12": digest,
            "path": str(path),
            "exists": exists,
            "size_bytes": int(path.stat().st_size) if exists else 0,
            "used_in_pipeline": used,
        })
    return pd.DataFrame(rows)


def minmax_score(values: pd.Series) -> pd.Series:
    vals = values.astype(float)
    finite = np.isfinite(vals)
    if not finite.any():
        return pd.Series(0.5, index=values.index)
    lo = float(vals[finite].min())
    hi = float(vals[finite].max())
    if hi - lo < 1e-9:
        return pd.Series(1.0, index=values.index)
    return ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0.5)


def inverse_minmax_score(values: pd.Series) -> pd.Series:
    return 1.0 - minmax_score(values)


def file_sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:12]


def build_research_method_traceability(reference_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Paper / filename": "remotesensing-14-04863.pdf",
            "Main scientific idea": "Fuzzy multi-factor landing site selection near PSRs using slope, rocks/roughness, illumination, maximum temperature, and PSR proximity.",
            "What factor/module it supports": "Fuzzy landing candidate scoring and landing safety constraints.",
            "How it will be used in LunaQuest": "Implements fuzzy membership scores for candidate proximity, slope safety, roughness hazard, and neutral placeholders for missing illumination/temperature layers.",
            "What not to claim": "Do not present preliminary landing candidates as certified landing products or use missing illumination/temperature layers as measured values.",
            "Implementation status": "implemented with slope/proximity/roughness active and illumination/temperature marked as missing neutral placeholders",
        },
        {
            "Paper / filename": "1-s2.0-S2095927325001999-main.pdf",
            "Main scientific idea": "Radar CPR can support polar ice-content scenarios, but high CPR/radar response is ambiguous because roughness and multiple scattering can also elevate radar returns.",
            "What factor/module it supports": "Radar candidate screening, roughness ambiguity penalty, and scenario-based resource estimates.",
            "How it will be used in LunaQuest": "Uses CPR-style ratio proxy cautiously, adds roughness ambiguity risk, keeps radar candidates for validation, and reports planning-only resource scenarios.",
            "What not to claim": "Do not treat CPR-style proxy or high candidate score as compositional proof or measured resource amount.",
            "Implementation status": "implemented as proxy radar screening plus roughness ambiguity and scenario tables",
        },
        {
            "Paper / filename": "pnas.1802345115.sapp.pdf",
            "Main scientific idea": "Surface-exposed water ice evidence is strengthened with M3 spectral absorptions plus Diviner temperature, LOLA albedo, and LAMP H2O-proxy context.",
            "What factor/module it supports": "External validation layer framework.",
            "How it will be used in LunaQuest": "Creates validation layer status for Diviner maximum temperature, LOLA albedo, LAMP, M3, PSR/shadow, and illumination; missing layers are marked for future download.",
            "What not to claim": "Do not use surface-exposed ice validation as subsurface proof, and do not fabricate unavailable validation maps.",
            "Implementation status": "framework implemented; external layers marked future-required unless present",
        },
        {
            "Paper / filename": "1-s2.0-S0094576525004898-main.pdf",
            "Main scientific idea": "Rover planning should combine static obstacles with dynamic illumination/communication constraints and science waypoint priorities.",
            "What factor/module it supports": "Conceptual rover route planning and route comparison.",
            "How it will be used in LunaQuest": "Uses A* route variants for shortest, safest, science-priority, and energy-aware planning on available slope/roughness/science proxy costs; missing dynamic layers are explicit future inputs.",
            "What not to claim": "Do not call route variants operational traverses or certified operational paths.",
            "Implementation status": "implemented as conceptual A* route variants with missing dynamic constraints documented",
        },
    ]
    df = pd.DataFrame(rows)
    if not reference_inventory.empty:
        status = reference_inventory.set_index("filename")["used_in_pipeline"].to_dict()
        df["Reference file available"] = df["Paper / filename"].map(status).fillna(False)
    return df


def add_landing_context_columns(landing_sites: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    out["illumination_score_status"] = "real layer missing; neutral placeholder used in active score"
    out["thermal_score_status"] = "real layer missing; neutral placeholder used in active score"
    out["communication_score_status"] = "future line-of-sight layer needed"
    out["safe_wording"] = "preliminary landing candidate; validation required"
    return out


def enrich_candidate_patches_with_slope(patches: pd.DataFrame, candidate_mask: np.ndarray, slope: np.ndarray | None) -> pd.DataFrame:
    if patches.empty or slope is None:
        return patches
    labels, _ = ndi.label(candidate_mask)
    out = patches.copy()
    mean_slopes: list[float] = []
    max_slopes: list[float] = []
    std_slopes: list[float] = []
    dem_roughness: list[float] = []
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
            std_slope = float(np.nanstd(vals))
        else:
            mean_slope = np.nan
            max_slope = np.nan
            std_slope = np.nan
        mean_slopes.append(mean_slope)
        max_slopes.append(max_slope)
        std_slopes.append(std_slope)
        dem_roughness.append(float(np.clip(std_slope / 8.0, 0, 1)) if np.isfinite(std_slope) else np.nan)
        slope_conditions.append(describe_slope_condition(mean_slope, max_slope))
    out["mean_slope_deg"] = mean_slopes
    out["max_slope_deg"] = max_slopes
    out["std_slope_deg"] = std_slopes
    out["mean_dem_roughness_proxy"] = dem_roughness
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


def build_threshold_sensitivity(features: dict[str, np.ndarray], config: dict[str, Any], transform, baseline_mask: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
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
    thresholds = np.array([0.60, 0.65, 0.70, 0.75, 0.80, 0.85], dtype=float)
    pixel_area_m2 = abs(float(transform.a * transform.e))
    stability_count = np.zeros(score.shape, dtype="float32")
    rows: list[dict[str, Any]] = []
    valid_pixels = max(int(valid.sum()), 1)
    baseline = baseline_mask.astype(bool) if baseline_mask is not None else None
    for score_thr in thresholds:
        mask = valid & (ratio >= ratio_thr) & (intensity >= intensity_thr) & (texture <= texture_thr) & (score >= score_thr)
        mask = remove_small_candidate_components(mask, min_pixels)
        labels, n = ndi.label(mask)
        stability_count += mask.astype("float32")
        top_ids = []
        if n:
            counts = np.bincount(labels.ravel())
            order = np.argsort(counts[1:])[::-1][:5] + 1
            top_ids = [f"T-{int(label):03d}" for label in order if counts[label] > 0]
        if baseline is not None and np.logical_or(mask, baseline).any():
            overlap = float(np.logical_and(mask, baseline).sum() / np.logical_or(mask, baseline).sum())
        else:
            overlap = np.nan
        rows.append({
            "threshold": float(score_thr),
            "score_threshold": float(score_thr),
            "candidate_patch_count": int(n),
            "number_of_patches": int(n),
            "candidate_pixels": int(mask.sum()),
            "candidate_pixel_count": int(mask.sum()),
            "candidate_area_m2": float(mask.sum() * pixel_area_m2),
            "candidate_area_km2": float(mask.sum() * pixel_area_m2 / 1_000_000.0),
            "candidate_area_pct_of_valid_aoi": float(mask.sum() / valid_pixels * 100),
            "top_patch_ids": ", ".join(top_ids),
            "stable_patch_overlap_with_baseline": overlap,
        })
    return pd.DataFrame(rows), (stability_count / max(len(thresholds), 1)).astype("float32")


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
    confidence_scores_after_penalty: list[float] = []
    confidence_levels: list[str] = []
    confidence_before_penalty: list[str] = []
    confidence_after_penalty: list[str] = []
    roughness_penalty_scores: list[float] = []
    ambiguity_risks: list[str] = []
    ambiguity_flags: list[str] = []
    stability_classes: list[str] = []
    confidence_reasons: list[str] = []
    area = out["area_m2"].astype(float)
    area_norm = ((area - area.min()) / max(float(area.max() - area.min()), 1e-6)).fillna(0.0).to_numpy()
    texture_vals = out["mean_texture"].astype(float)
    texture_norm = ((texture_vals - texture_vals.min()) / max(float(texture_vals.max() - texture_vals.min()), 1e-6)).fillna(0.0).to_numpy()
    dem_vals = out["mean_dem_roughness_proxy"].astype(float) if "mean_dem_roughness_proxy" in out else pd.Series(np.nan, index=out.index)
    dem_vals = dem_vals.fillna(dem_vals.median() if np.isfinite(dem_vals).any() else 0.5)
    dem_norm = np.clip(dem_vals.to_numpy(dtype=float), 0, 1)
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
        before_level = "High" if confidence_score >= 0.78 else "Medium" if confidence_score >= 0.68 else "Low"
        slope_risk = np.clip(float(row.get("mean_slope_deg", 0.0)) / 15.0, 0, 1) if np.isfinite(float(row.get("mean_slope_deg", np.nan))) else 0.5
        roughness_risk_score = float(np.clip(0.55 * texture_norm[i] + 0.25 * dem_norm[i] + 0.20 * slope_risk, 0, 1))
        penalty = 0.22 * roughness_risk_score
        confidence_score_after = float(np.clip(confidence_score - penalty, 0, 1))
        after_level = "High" if confidence_score_after >= 0.78 else "Medium" if confidence_score_after >= 0.68 else "Low"
        risk_label = "High" if roughness_risk_score >= 0.66 else "Medium" if roughness_risk_score >= 0.38 else "Low"
        stability_label = "High" if stability >= 0.67 else "Medium" if stability >= 0.34 else "Low"
        if risk_label == "High" and float(row.get("mean_candidate_score", 0.0)) >= 0.70:
            flag = "strong radar score but roughness ambiguity high; needs OHRC/DEM/multi-frequency validation"
        elif risk_label == "High":
            flag = "roughness ambiguity high; validation priority depends on external layers"
        else:
            flag = "roughness ambiguity manageable in current proxy layers"
        reason = (
            f"{before_level} before penalty; {after_level} after roughness penalty; "
            f"{risk_label.lower()} roughness ambiguity and {stability_label.lower()} threshold stability."
        )
        max_scores.append(max_score)
        stabilities.append(stability)
        uncertainties.append(mean_uncertainty)
        confidence_scores.append(float(confidence_score))
        confidence_scores_after_penalty.append(confidence_score_after)
        confidence_levels.append(after_level)
        confidence_before_penalty.append(before_level)
        confidence_after_penalty.append(after_level)
        roughness_penalty_scores.append(float(penalty))
        ambiguity_risks.append(risk_label)
        ambiguity_flags.append(flag)
        stability_classes.append(stability_label)
        confidence_reasons.append(reason)
    out["area_km2"] = out["area_m2"].astype(float) / 1_000_000.0
    out["equivalent_candidate_patch_diameter_m"] = 2.0 * np.sqrt(out["area_m2"].astype(float) / np.pi)
    out["equivalent_diameter_m"] = out["equivalent_candidate_patch_diameter_m"]
    out["max_candidate_score"] = max_scores
    out["threshold_stability"] = stabilities
    out["mean_uncertainty_score"] = uncertainties
    out["confidence_score"] = confidence_scores
    out["confidence_score_after_roughness_penalty"] = confidence_scores_after_penalty
    out["confidence_level"] = confidence_levels
    out["confidence_before_penalty"] = confidence_before_penalty
    out["confidence_after_penalty"] = confidence_after_penalty
    out["candidate_confidence_after_roughness_penalty"] = confidence_after_penalty
    out["roughness_penalty_score"] = roughness_penalty_scores
    out["roughness_ambiguity_risk"] = ambiguity_risks
    out["radar_ambiguity_flag"] = ambiguity_flags
    out["threshold_stability_class"] = stability_classes
    out["reason_for_confidence"] = confidence_reasons
    out = out.sort_values(["confidence_score_after_roughness_penalty", "mean_candidate_score", "area_m2"], ascending=False).reset_index(drop=True)
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


def enrich_landing_sites_with_fuzzy_components(
    landing_sites: pd.DataFrame,
    landing_layers: dict[str, Any],
    candidate_mask: np.ndarray,
    slope: np.ndarray | None,
    aoi: dict[str, Any],
) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    slope_scores: list[float] = []
    proximity_scores: list[float] = []
    roughness_scores: list[float] = []
    illumination_scores: list[float] = []
    temperature_scores: list[float] = []
    inside_candidate: list[str] = []
    inside_f2: list[str] = []
    inside_aoi: list[str] = []
    for _, site in out.iterrows():
        r = int(site["row"])
        c = int(site["col"])
        slope_scores.append(sample_layer(landing_layers.get("slope_score"), r, c, np.nan))
        proximity_scores.append(sample_layer(landing_layers.get("candidate_proximity_score"), r, c, np.nan))
        roughness_scores.append(sample_layer(landing_layers.get("low_hazard_score"), r, c, np.nan))
        illumination_scores.append(sample_layer(landing_layers.get("illumination_score"), r, c, 0.5))
        temperature_scores.append(sample_layer(landing_layers.get("temperature_score"), r, c, 0.5))
        inside_candidate.append("yes" if 0 <= r < candidate_mask.shape[0] and 0 <= c < candidate_mask.shape[1] and bool(candidate_mask[r, c]) else "no")
        inside_aoi.append(
            "yes"
            if aoi["lat_min"] <= float(site["lat"]) <= aoi["lat_max"] and aoi["lon_min"] <= float(site["lon"]) <= aoi["lon_max"]
            else "no"
        )
        inside_f2.append("not_available_boundary_required")
    out["score_total"] = out["suitability_score"].astype(float)
    out["score_slope"] = slope_scores
    out["score_candidate_proximity"] = proximity_scores
    out["score_roughness"] = roughness_scores
    out["score_illumination"] = illumination_scores
    out["score_temperature"] = temperature_scores
    out["inside_candidate_mask_yes_no"] = inside_candidate
    out["inside_configured_aoi_yes_no"] = inside_aoi
    out["inside_f2_crater_estimate_yes_no"] = inside_f2
    out["inside_steep_unsafe_slope_zone_yes_no"] = [
        "yes" if slope is not None and np.isfinite(slope[int(row["row"]), int(row["col"])]) and slope[int(row["row"]), int(row["col"])] > 10 else "no"
        for _, row in out.iterrows()
    ]
    return out


def sample_layer(layer: Any, row: int, col: int, default: float) -> float:
    if isinstance(layer, np.ndarray) and 0 <= row < layer.shape[0] and 0 <= col < layer.shape[1]:
        val = layer[row, col]
        return float(val) if np.isfinite(val) else default
    return float(default)


def attach_landing_route_accessibility(landing_sites: pd.DataFrame, route_summary: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    ok_routes = route_summary[route_summary["status"].astype(str).str.lower().eq("ok")] if not route_summary.empty else pd.DataFrame()
    target_ids = set(ok_routes["target_candidate_id"].astype(str)) if not ok_routes.empty and "target_candidate_id" in ok_routes else set()
    start_site_ids = set(ok_routes["start_landing_site_id"].astype(str)) if not ok_routes.empty and "start_landing_site_id" in ok_routes else set()
    nearest_target_ok = out["nearest_candidate_id"].astype(str).isin(target_ids) if "nearest_candidate_id" in out else pd.Series(False, index=out.index)
    start_site_ok = out["site_id"].astype(str).isin(start_site_ids) if "site_id" in out else pd.Series(False, index=out.index)
    out["route_accessible"] = (nearest_target_ok | start_site_ok).map({True: "yes", False: "needs route check"})
    out["route_accessibility_to_candidate_patch"] = out["route_accessible"]
    return out


def build_landing_site_evaluation(landing_sites: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    slope = out["slope_deg"].astype(float)
    dist = out["distance_to_candidate_m"].astype(float)
    out["low_slope_score"] = out.get("score_slope", 1.0 - np.clip(slope / 8.0, 0, 1)).astype(float).round(4)
    out["candidate_proximity_score"] = (1.0 / (1.0 + dist / 1000.0)).round(4)
    out["roughness_avoidance_status"] = "included via SAR texture proxy"
    out["candidate_mask_clearance_status"] = "outside candidate mask with configured clearance"
    out["local_terrain_smoothness_status"] = "represented by low slope and texture proxy"
    out["illumination_layer_status"] = "real layer missing; neutral placeholder used"
    out["thermal_layer_status"] = "real layer missing; neutral placeholder used"
    out["communication_layer_status"] = "future line-of-sight layer needed"
    out["reason_selected"] = "high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance"
    out["validation_needed"] = "illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation"
    out["score_total"] = out.get("score_total", out["suitability_score"])
    out["route_accessible_yes_no"] = out.get("route_accessible", "needs route check")
    out["final_recommendation"] = np.where(
        (out["inside_candidate_mask_yes_no"].astype(str).eq("no")) & (slope < 8),
        "keep",
        "needs validation",
    )
    out["reason"] = np.where(
        out["final_recommendation"].eq("keep"),
        "near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer",
        "requires crater boundary, illumination/PSR, thermal, and hazard validation before use",
    )
    columns = [
        "site_id", "lat", "lon", "score_total", "suitability_score", "score_slope",
        "score_candidate_proximity", "score_roughness", "score_illumination", "score_temperature",
        "slope_deg", "distance_to_candidate_m", "distance_to_target_candidate_m",
        "nearest_candidate_id", "nearest_candidate_confidence_level", "route_accessible",
        "route_accessible_yes_no", "inside_candidate_mask_yes_no", "inside_configured_aoi_yes_no",
        "inside_f2_crater_estimate_yes_no",
        "inside_steep_unsafe_slope_zone_yes_no", "final_recommendation", "reason", "reason_selected",
        "validation_needed", "low_slope_score", "candidate_proximity_score", "roughness_avoidance_status",
        "candidate_mask_clearance_status", "illumination_layer_status", "thermal_layer_status", "communication_layer_status",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_fuzzy_landing_site_scores(landing_sites: pd.DataFrame) -> pd.DataFrame:
    if landing_sites.empty:
        return landing_sites
    out = landing_sites.copy()
    out["route_accessible_yes_no"] = out.get("route_accessible", "needs route check")
    out["final_recommendation"] = np.where(
        (out["inside_candidate_mask_yes_no"].astype(str).eq("no")) & (out["slope_deg"].astype(float) < 8),
        "keep",
        "needs validation",
    )
    out["reason"] = np.where(
        out["final_recommendation"].eq("keep"),
        "preliminary landing candidate near radar-based candidate patch while outside candidate mask and below 8 deg slope",
        "requires additional validation before retaining",
    )
    keep = [
        "site_id", "lat", "lon", "score_total", "score_slope", "score_candidate_proximity",
        "score_roughness", "score_illumination", "score_temperature", "nearest_candidate_id",
        "distance_to_candidate_m", "inside_candidate_mask_yes_no", "inside_configured_aoi_yes_no",
        "inside_f2_crater_estimate_yes_no",
        "route_accessible_yes_no", "final_recommendation", "reason",
    ]
    return out[[c for c in keep if c in out.columns]]


def build_landing_crater_boundary_check(
    landing_sites: pd.DataFrame,
    candidate_mask: np.ndarray,
    slope: np.ndarray | None,
    aoi: dict[str, Any],
) -> pd.DataFrame:
    if landing_sites.empty:
        return pd.DataFrame()
    rows = []
    for _, site in landing_sites.iterrows():
        r, c = int(site["row"]), int(site["col"])
        slope_val = float(slope[r, c]) if slope is not None and np.isfinite(slope[r, c]) else np.nan
        rows.append({
            "site_id": site["site_id"],
            "lat": float(site["lat"]),
            "lon": float(site["lon"]),
            "inside_configured_aoi_yes_no": "yes" if aoi["lat_min"] <= float(site["lat"]) <= aoi["lat_max"] and aoi["lon_min"] <= float(site["lon"]) <= aoi["lon_max"] else "no",
            "inside_approximate_f2_crater_boundary": "not_available_boundary_required",
            "inside_radar_candidate_mask": "yes" if bool(candidate_mask[r, c]) else "no",
            "inside_steep_unsafe_slope_zone": "yes" if np.isfinite(slope_val) and slope_val > 10 else "no",
            "near_rim_or_safe_terrain": "not_evaluated_crater_boundary_required",
            "reachable_by_rover_path": site.get("route_accessible", "needs route check"),
            "boundary_note": "Exact F2 crater boundary was not available in the workspace; no synthetic crater boundary was fabricated.",
        })
    return pd.DataFrame(rows)


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
        "candidate_id", "refinement_status", "refined_rank", "refined_candidate_score",
        "area_m2", "area_km2", "equivalent_candidate_patch_diameter_m", "equivalent_diameter_m",
        "centroid_lat", "centroid_lon", "mean_ice_probability", "max_ice_probability",
        "ice_confidence_score", "depth_likelihood_class", "shallow_ice_likelihood",
        "deep_ice_likelihood", "psr_stability_proxy",
        "mean_ratio_contribution", "mean_intensity_contribution",
        "mean_roughness_suitability_contribution", "mean_psr_proxy_contribution",
        "mean_candidate_score", "max_candidate_score", "mean_ratio_proxy",
        "mean_texture", "mean_slope_deg", "max_slope_deg", "mean_dem_roughness_proxy",
        "mean_terrain_roughness", "slope_safety_score", "roughness_hazard_score",
        "roughness_ambiguity_risk", "threshold_stability", "threshold_stability_class",
        "confidence_before_penalty", "confidence_after_penalty", "confidence_level",
        "candidate_confidence_after_roughness_penalty",
        "radar_ambiguity_flag", "reason_for_confidence", "distance_to_nearest_landing_candidate_m",
        "nearest_landing_site_id", "route_accessible_yes_no", "mean_uncertainty_score",
        "confidence_score", "confidence_score_after_roughness_penalty", "validation_priority_rank", "slope_condition",
    ]
    review = out[[c for c in keep if c in out.columns]].copy()
    if "refined_rank" in review.columns:
        return review.sort_values(["refined_rank", "validation_priority_rank"], na_position="last")
    return review.sort_values("validation_priority_rank")


def build_ice_candidate_patch_review(candidate_review: pd.DataFrame) -> pd.DataFrame:
    if candidate_review.empty:
        return candidate_review
    out = candidate_review.copy()
    out["distance_to_nearest_landing_site_m"] = out["distance_to_nearest_landing_candidate_m"]
    columns = [
        "candidate_id", "refinement_status", "refined_rank", "refined_candidate_score",
        "area_m2", "equivalent_candidate_patch_diameter_m", "centroid_lat", "centroid_lon",
        "mean_ice_probability", "max_ice_probability", "ice_confidence_score",
        "depth_likelihood_class", "shallow_ice_likelihood", "deep_ice_likelihood",
        "psr_stability_proxy", "mean_ratio_contribution", "mean_intensity_contribution",
        "mean_roughness_suitability_contribution", "mean_psr_proxy_contribution",
        "mean_candidate_score",
        "mean_texture", "mean_dem_roughness_proxy", "mean_terrain_roughness",
        "slope_safety_score", "roughness_hazard_score", "roughness_ambiguity_risk",
        "confidence_before_penalty", "confidence_after_penalty", "candidate_confidence_after_roughness_penalty",
        "confidence_level", "reason_for_confidence", "mean_slope_deg", "max_slope_deg",
        "nearest_landing_site_id", "distance_to_nearest_landing_site_m", "validation_priority_rank",
    ]
    if "candidate_confidence_after_roughness_penalty" not in out and "confidence_after_penalty" in out:
        out["candidate_confidence_after_roughness_penalty"] = out["confidence_after_penalty"]
    return out[[c for c in columns if c in out.columns]]


def build_resource_scenario_estimates(candidate_review: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidate_review.empty:
        return pd.DataFrame(rows)
    top = candidate_review.sort_values("validation_priority_rank").head(top_n)
    for _, patch in top.iterrows():
        for depth_m in [1, 3, 5]:
            for ice_fraction in [0.01, 0.03, 0.06, 0.10]:
                scenario_volume = float(patch["area_m2"]) * depth_m * ice_fraction
                rows.append({
                    "candidate_id": patch["candidate_id"],
                    "confidence_level": patch.get("confidence_level", ""),
                    "candidate_area_m2": float(patch["area_m2"]),
                    "assumed_depth_m": depth_m,
                    "assumed_ice_fraction": ice_fraction,
                    "scenario_volume_m3": scenario_volume,
                    "scenario_mass_kg_using_917kg_m3": scenario_volume * 917.0,
                    "interpretation": "scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate",
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
        five_to_8 = float(np.sum((slopes >= 5) & (slopes < 8)) / total * 100) if slopes.size else np.nan
        eight_to_10 = float(np.sum((slopes >= 8) & (slopes <= 10)) / total * 100) if slopes.size else np.nan
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
            "percent_under_5deg": under_5,
            "percent_5_to_8deg": five_to_8,
            "percent_8_to_10deg": eight_to_10,
            "percent_route_5_to_10deg": five_to_10,
            "percent_route_above_10deg": above_10,
            "percent_above_10deg": above_10,
            "blocked_cells_avoided": blocked_avoided,
            "science_reward_score": science_reward,
            "energy_cost_proxy": energy_proxy,
            "traverse_risk_score": risk,
            "route_accessibility_note": "conceptual rover navigation planning prototype; not operational rover command generation",
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    length_score = inverse_minmax_score(result["length_m"].astype(float))
    energy_score = inverse_minmax_score(result["energy_cost_proxy"].astype(float))
    risk_score = inverse_minmax_score(result["traverse_risk_score"].astype(float))
    science_score = minmax_score(result["science_reward_score"].astype(float))
    result["route_decision_score"] = (
        0.25 * length_score
        + 0.25 * energy_score
        + 0.25 * risk_score
        + 0.25 * science_score
    ).round(4)
    result["route_decision_rank"] = result["route_decision_score"].rank(method="first", ascending=False).astype(int)
    result["route_recommendation"] = np.where(result["route_decision_rank"].eq(1), "recommended", "alternative")
    return result


def path_values_for_profile(path: list[tuple[int, int]], arr: np.ndarray | None) -> np.ndarray:
    if arr is None or not path:
        return np.array([], dtype=float)
    return np.array([arr[r, c] for r, c in path if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]], dtype=float)


def recommended_route_type(route_summary: pd.DataFrame) -> str:
    if route_summary.empty:
        return ""
    if "route_decision_rank" in route_summary:
        rows = route_summary[route_summary["route_decision_rank"].astype(float).eq(1)]
        if not rows.empty:
            return str(rows.iloc[0]["route_type"])
    ok = route_summary[route_summary["status"].astype(str).str.lower().eq("ok")] if "status" in route_summary else route_summary
    if ok.empty:
        return str(route_summary.iloc[0]["route_type"])
    sort_cols = [c for c in ["traverse_risk_score", "energy_cost_proxy", "length_m"] if c in ok.columns]
    return str(ok.sort_values(sort_cols).iloc[0]["route_type"]) if sort_cols else str(ok.iloc[0]["route_type"])


def build_rover_traversal_simulation(
    routes: dict[str, list[tuple[int, int]]],
    route_summary: pd.DataFrame,
    features: dict[str, np.ndarray],
    slope: np.ndarray | None,
    transform,
) -> pd.DataFrame:
    route_type = recommended_route_type(route_summary)
    path = routes.get(route_type, [])
    if not path:
        return pd.DataFrame()
    pixel_size = abs(float(transform.a))
    texture = robust_normalize(features["texture"], features["valid"].astype(bool))
    science = robust_normalize(features["candidate_score"], features["valid"].astype(bool))
    rows: list[dict[str, Any]] = []
    cumulative_distance = 0.0
    cumulative_energy = 0.0
    prev = path[0]
    for i, (r, c) in enumerate(path):
        step_m = 0.0 if i == 0 else float(((r - prev[0]) ** 2 + (c - prev[1]) ** 2) ** 0.5 * pixel_size)
        slope_val = float(slope[r, c]) if slope is not None and np.isfinite(slope[r, c]) else 0.0
        texture_val = float(texture[r, c]) if np.isfinite(texture[r, c]) else 0.0
        science_val = float(science[r, c]) if np.isfinite(science[r, c]) else 0.0
        step_energy = step_m * (1.0 + 0.035 * max(slope_val, 0.0) + 0.18 * max(texture_val, 0.0))
        cumulative_distance += step_m
        cumulative_energy += step_energy
        rows.append({
            "route_type": route_type,
            "step": i,
            "row": int(r),
            "col": int(c),
            "distance_m": cumulative_distance,
            "step_distance_m": step_m,
            "slope_deg": slope_val,
            "texture_risk_proxy": texture_val,
            "science_reward_proxy": science_val,
            "step_energy_proxy": step_energy,
            "cumulative_energy_proxy": cumulative_energy,
        })
        prev = (r, c)
    return pd.DataFrame(rows)


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
        "acceptable_5_to_8deg_pct": float((((slope >= 5) & (slope < 8)) & mask).sum() / total * 100),
        "marginal_8_to_10deg_pct": float((((slope >= 8) & (slope <= 10)) & mask).sum() / total * 100),
        "moderate_5_10deg_pct": float((((slope >= 5) & (slope <= 10)) & mask).sum() / total * 100),
        "unsafe_gt_10deg_pct": float(((slope > 10) & mask).sum() / total * 100),
        "blocked_gt_15deg_pct": float(((slope > 15) & mask).sum() / total * 100),
    }


def build_data_coverage_status(
    inventory: pd.DataFrame,
    sar_scores: pd.DataFrame,
    ohrc_fp: pd.DataFrame,
    pair: dict[str, Any],
    aoi: dict[str, Any],
    tmc_coverage: pd.DataFrame,
) -> pd.DataFrame:
    ohrc_overlap = bool(
        not ohrc_fp.empty
        and ohrc_fp.get("coverage_fraction", pd.Series(dtype=float)).astype(float).gt(0).any()
    )
    selected_tmc = (
        tmc_coverage[tmc_coverage.get("selected_for_terrain", pd.Series(False, index=tmc_coverage.index)).astype(bool)]
        if not tmc_coverage.empty
        else pd.DataFrame()
    )
    tmc_detail = "No TMC-2 DTM intersects the configured AOI."
    tmc_status = "not available"
    if not selected_tmc.empty:
        row = selected_tmc.iloc[0]
        tmc_status = "usable"
        tmc_detail = (
            f"Selected {row.get('product_id')} with AOI coverage "
            f"{float(row.get('coverage_fraction', 0.0)):.3f}; used for terrain slope/roughness."
        )
    elif not tmc_coverage.empty and tmc_coverage["usable_for_analysis"].astype(bool).any():
        tmc_status = "usable_partial"
        tmc_detail = "TMC-2 DTM intersects the AOI but no selected terrain product was flagged."
    return pd.DataFrame([
        {"layer": "SAR/DFSAR", "status": "usable", "detail": f"Selected {pair['product_id']} with AOI coverage {float(pair['coverage_fraction']):.3f}"},
        {"layer": "TMC-2 DTM", "status": tmc_status, "detail": tmc_detail},
        {"layer": "DEM/LDEM/LDSM", "status": "usable_fallback", "detail": "LOLA/LDEM/LDSM terrain is retained as fallback/context and for TMC-2 sensitivity comparison when available."},
        {"layer": "OHRC", "status": "usable" if ohrc_overlap else "context only", "detail": "Current OHRC footprints overlap the AOI." if ohrc_overlap else "Available OHRC footprints do not overlap the Faustini AOI; excluded from hazard scoring."},
        {"layer": "Ground truth labels", "status": "not available", "detail": "U-Net uses rule-based pseudo-labels; metrics are agreement only."},
    ])


def compute_terrain_roughness(elevation: np.ndarray | None, slope: np.ndarray | None) -> np.ndarray | None:
    if elevation is not None and np.isfinite(elevation).any():
        filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation[np.isfinite(elevation)]))
        local_mean = ndi.uniform_filter(filled.astype("float32"), size=7)
        rough = np.abs(filled - local_mean)
        return robust_normalize(rough.astype("float32"), np.isfinite(rough)).astype("float32")
    if slope is not None and np.isfinite(slope).any():
        return robust_normalize(slope.astype("float32"), np.isfinite(slope)).astype("float32")
    return None


def compute_hillshade(elevation: np.ndarray | None, azimuth_deg: float = 315.0, altitude_deg: float = 35.0) -> np.ndarray | None:
    if elevation is None or not np.isfinite(elevation).any():
        return None
    filled = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation[np.isfinite(elevation)]))
    gy, gx = np.gradient(filled.astype("float32"))
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1.0) / 2.0, 0, 1).astype("float32")


def build_validation_layer_status(
    inventory: pd.DataFrame,
    ohrc_fp: pd.DataFrame,
    dem: dict[str, Any],
    pair: dict[str, Any],
    tmc_dem: dict[str, Any],
) -> pd.DataFrame:
    ohrc_overlap = bool(
        not ohrc_fp.empty
        and ohrc_fp.get("coverage_fraction", pd.Series(dtype=float)).astype(float).gt(0).any()
    )
    tmc_available = bool(isinstance(tmc_dem, dict) and tmc_dem.get("available"))
    rows = [
        {
            "validation_layer": "SAR/DFSAR LH/LV SRI",
            "status": "available",
            "used_now": "yes",
            "module_enabled": "radar candidate screening",
            "confidence_effect": "primary screening evidence, proxy-only",
            "next_action": f"Selected product {pair.get('product_id')}",
        },
        {
            "validation_layer": "LOLA/LDEM/LDSM terrain",
            "status": "available",
            "used_now": "fallback/context" if tmc_available else "yes",
            "module_enabled": "slope, roughness, landing, rover",
            "confidence_effect": "improves terrain safety and roughness ambiguity review",
            "next_action": "Retain as fallback and sensitivity layer beside TMC-2 DTM",
        },
        {
            "validation_layer": "TMC-2 DTM terrain",
            "status": "available" if tmc_available else "future_validation_layer_required",
            "used_now": "yes" if tmc_available else "no",
            "module_enabled": "slope, roughness, landing, rover",
            "confidence_effect": "local terrain/slope support for candidate ranking and route safety; not optical boulder confirmation",
            "next_action": f"Selected {tmc_dem.get('selected_product_id', '')}" if tmc_available else "Download or co-register a TMC-2 DTM covering the AOI",
        },
        {
            "validation_layer": "OHRC calibrated Faustini overlap",
            "status": "available_context_only" if ohrc_overlap else "future_validation_layer_required",
            "used_now": "yes" if ohrc_overlap else "no",
            "module_enabled": "hazard proxy / boulder review",
            "confidence_effect": "would reduce roughness ambiguity and landing hazard uncertainty",
            "next_action": "Download calibrated OHRC overlapping the configured AOI",
        },
        {
            "validation_layer": "Diviner maximum temperature",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "thermal/cold-trap suitability",
            "confidence_effect": "would support <110 K cold-trap suitability screening if available",
            "next_action": "Download Diviner maximum temperature map for AOI",
        },
        {
            "validation_layer": "PSR / shadow / illumination",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "PSR validation and landing power constraints",
            "confidence_effect": "would separate persistent shadow from rover power/landing constraints",
            "next_action": "Download PSR or illumination persistence layer",
        },
        {
            "validation_layer": "PSR stability proxy",
            "status": "available_proxy",
            "used_now": "yes",
            "module_enabled": "candidate stability context",
            "confidence_effect": "adds approximate polar/shadow context but does not replace real illumination modeling",
            "next_action": "Replace proxy with validated PSR/illumination layer when available",
        },
        {
            "validation_layer": "LOLA albedo",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "surface-exposure validation context",
            "confidence_effect": "would provide independent albedo context for surface ice screening",
            "next_action": "Download LOLA albedo layer for AOI",
        },
        {
            "validation_layer": "LAMP H2O proxy / band ratio",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "surface-exposure validation context",
            "confidence_effect": "would add ultraviolet H2O-proxy evidence",
            "next_action": "Download LAMP band-ratio product if available",
        },
        {
            "validation_layer": "M3 spectral ice evidence",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "surface ice validation",
            "confidence_effect": "would validate surface-exposed ice signatures only, not subsurface proof",
            "next_action": "Download suitable polar M3 spectral products if available",
        },
        {
            "validation_layer": "Complex/Stokes SAR products",
            "status": "future_validation_layer_required",
            "used_now": "no",
            "module_enabled": "true CPR/DOP derivation",
            "confidence_effect": "would replace CPR-style proxy with physically calibrated polarimetry",
            "next_action": "Download complex/Stokes products and convention documentation",
        },
    ]
    return pd.DataFrame(rows)


def build_candidate_validation_against_external_layers(candidate_review: pd.DataFrame, validation_status: pd.DataFrame) -> pd.DataFrame:
    if candidate_review.empty:
        return pd.DataFrame()
    rows = []
    external_available = set(validation_status.loc[validation_status["used_now"].astype(str).eq("yes"), "validation_layer"].astype(str))
    for _, patch in candidate_review.iterrows():
        rows.append({
            "candidate_id": patch["candidate_id"],
            "dem_slope_available": "yes" if np.isfinite(float(patch.get("mean_slope_deg", np.nan))) else "no",
            "roughness_proxy_available": "yes" if np.isfinite(float(patch.get("mean_texture", np.nan))) else "no",
            "diviner_temperature_status": "future validation layer required",
            "psr_shadow_status": "future validation layer required",
            "lola_albedo_status": "future validation layer required",
            "lamp_h2o_proxy_status": "future validation layer required",
            "m3_spectral_status": "future validation layer required",
            "current_external_validation_summary": "terrain context available; thermal, PSR/illumination, albedo, LAMP, and M3 layers not available locally",
            "validation_priority_rank": patch.get("validation_priority_rank", np.nan),
        })
    return pd.DataFrame(rows)


def build_hazard_proxy_summary(features: dict[str, np.ndarray], terrain_roughness: np.ndarray | None, ohrc_fp: pd.DataFrame) -> pd.DataFrame:
    valid = features["valid"].astype(bool)
    sar_texture = robust_normalize(features["texture"], valid)
    sar_hazard = sar_texture > np.nanquantile(sar_texture[valid], 0.90)
    rows = [
        {
            "hazard_layer": "SAR texture roughness proxy",
            "status": "available",
            "hazard_pixel_fraction_pct": float(sar_hazard[valid].mean() * 100) if valid.any() else np.nan,
            "interpretation": "proxy roughness/hazard layer; not direct boulder detection",
        }
    ]
    if terrain_roughness is not None:
        terrain_valid = np.isfinite(terrain_roughness)
        terrain_hazard = terrain_roughness > np.nanquantile(terrain_roughness[terrain_valid], 0.90)
        rows.append({
            "hazard_layer": "DEM terrain roughness proxy",
            "status": "available",
            "hazard_pixel_fraction_pct": float(terrain_hazard[terrain_valid].mean() * 100) if terrain_valid.any() else np.nan,
            "interpretation": "local terrain roughness proxy for planning",
        })
    rows.append({
        "hazard_layer": "OHRC boulder/hazard layer",
        "status": "context_only_or_missing",
        "hazard_pixel_fraction_pct": np.nan,
        "interpretation": "requires calibrated Faustini-overlapping OHRC before boulder claims",
    })
    return pd.DataFrame(rows)


def candidate_uncertainty(score: np.ndarray, threshold: float, valid: np.ndarray) -> np.ndarray:
    vals = score[valid & np.isfinite(score)]
    scale = max(float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5)), 1e-6) if vals.size else 1.0
    uncertainty = 1.0 - np.clip(np.abs(score - threshold) / (0.2 * scale), 0, 1)
    uncertainty[~valid | ~np.isfinite(uncertainty)] = np.nan
    return uncertainty.astype("float32")


def copy_legacy_figure_names(figures: Path) -> None:
    aliases = [
        ("04_sar_feature_panel.png", "sar_feature_panel.png"),
        ("04_sar_feature_panel.png", "radar_feature_panel.png"),
        ("05_radar_candidate_overlay.png", "sar_candidate_overlay.png"),
        ("05_radar_candidate_overlay.png", "radar_candidate_overlay.png"),
        ("06_candidate_patch_area_histogram.png", "candidate_patch_area_histogram.png"),
        ("07_dem_elevation_map.png", "dem_elevation_map.png"),
        ("08_dem_slope_map.png", "dem_slope_map.png"),
        ("08_dem_slope_map.png", "dtm_slope_map.png"),
        ("08b_slope_classification_map.png", "slope_classification_map.png"),
        ("10_top_landing_candidates_overlay.png", "landing_suitability_overlay.png"),
        ("11_rover_route_comparison.png", "route_cost_comparison.png"),
        ("11_rover_route_comparison.png", "rover_route_comparison_chart.png"),
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


def save_research_backed_pipeline_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        "Data audit\nregion validation",
        "SAR/DFSAR\nfeature extraction",
        "Radar candidate\nmap",
        "Patch evaluation\nuncertainty",
        "External validation\nstatus",
        "DEM/DTM\nterrain",
        "Fuzzy landing\nscoring",
        "Boundary\nvalidation",
        "Rover route\nplanning",
        "Weak U-Net\npseudo-labels",
        "Research outputs\nvalidation report",
    ]
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_axis_off()
    colors = ["#e8f4f8", "#eef5e8", "#fff4d6", "#f7e9f2", "#eae7ff"]
    coords: list[tuple[float, float]] = []
    for x in np.linspace(0.10, 0.90, 6):
        coords.append((float(x), 0.62))
    for x in np.linspace(0.16, 0.84, 5):
        coords.append((float(x), 0.30))

    box_w = 0.128
    box_h = 0.18
    for i, ((x, y), label) in enumerate(zip(coords, steps)):
        ax.add_patch(plt.Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, facecolor=colors[i % len(colors)], edgecolor="#263238", linewidth=1.1))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.0, weight="bold")
        if i < len(steps) - 1:
            nx, ny = coords[i + 1]
            if abs(ny - y) < 1e-6:
                ax.annotate("", xy=(nx - box_w / 2 - 0.012, ny), xytext=(x + box_w / 2 + 0.012, y), arrowprops=dict(arrowstyle="->", color="#263238", lw=1.25))
            else:
                ax.annotate("", xy=(nx, ny + box_h / 2 + 0.018), xytext=(x, y - box_h / 2 - 0.018), arrowprops=dict(arrowstyle="->", color="#263238", lw=1.25, connectionstyle="angle3,angleA=-90,angleB=180"))
    ax.text(0.5, 0.88, "LunaQuest / Lunar IceNav Research-Backed Decision Pipeline", ha="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.08, "Radar-based candidate regions, preliminary landing candidates, conceptual rover routes, pseudo-label ML, and validation reports; independent validation required.", ha="center", fontsize=10.5)
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
    ax.legend(handles=[Line2D([0], [0], color="#00acc1", lw=7, alpha=0.6, label="candidate region")], loc="lower right", framealpha=0.9, fontsize=10)
    fig.text(0.01, 0.025, "Radar-based candidate regions; validation required.", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_ice_probability_map(
    base: np.ndarray,
    ice_probability: np.ndarray,
    candidate_mask: np.ndarray,
    patches: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    shown = np.ma.masked_invalid(ice_probability)
    im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=1, alpha=0.72)
    contour_mask = candidate_mask.astype(float)
    if np.nanmax(contour_mask) > 0:
        ax.contour(contour_mask, levels=[0.5], colors=["#ffffff"], linewidths=0.8)
    if not patches.empty:
        rank_col = "refined_rank" if "refined_rank" in patches else "validation_priority_rank"
        top = patches.sort_values(rank_col, na_position="last").head(8)
        for _, row in top.iterrows():
            ax.text(
                row["centroid_col"] + 5,
                row["centroid_row"] + 5,
                str(row["candidate_id"]),
                color="black",
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="#2e7d32", alpha=0.82, pad=1.5),
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    cbar.set_label("Weighted candidate probability score")
    ax.set_title("Radar/Terrain Ice Candidate Probability Map", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    ax.legend(handles=[Line2D([0], [0], color="#ffffff", lw=1.8, label="refined candidate outline")], loc="lower right", framealpha=0.88, fontsize=9)
    fig.text(0.01, 0.025, "Weighted SAR + CPR-style proxy + roughness-penalized candidate screening; compositional validation required.", fontsize=9.5)
    fig.subplots_adjust(left=0.01, right=0.94, top=0.86, bottom=0.10)
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
    x = df["threshold"] if "threshold" in df else df["score_quantile"]
    ax1.plot(x, df["candidate_patch_count"], marker="o", color="#2f80ed", label="candidate patches")
    ax2.plot(x, df["candidate_area_m2"], marker="s", color="#f4511e", label="candidate area")
    ax1.set_xlabel("Candidate score threshold")
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
    fig.text(0.02, 0.035, "Equivalent diameter is an area-derived candidate patch extent, not a compositional body diameter.", fontsize=8)
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


def save_candidate_score_vs_roughness(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    if not review.empty:
        risk_colors = review["roughness_ambiguity_risk"].map({"High": "#e74c3c", "Medium": "#ffcc00", "Low": "#1b9e77"}).fillna("#90a4ae")
        ax.scatter(review["mean_texture"], review["mean_candidate_score"], s=np.clip(review["area_m2"].astype(float) / 60, 28, 180), c=risk_colors, edgecolor="black", linewidth=0.4)
        for _, row in review.sort_values("validation_priority_rank").head(8).iterrows():
            ax.text(row["mean_texture"] * 1.01, row["mean_candidate_score"] + 0.002, row["candidate_id"], fontsize=8)
    ax.set_xlabel("Mean SAR texture / roughness proxy")
    ax.set_ylabel("Mean candidate score")
    ax.set_title("Candidate Score vs Roughness Ambiguity", fontsize=14, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#e74c3c", lw=0, label="High ambiguity"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#ffcc00", lw=0, label="Medium ambiguity"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="#1b9e77", lw=0, label="Low ambiguity"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.01, "High radar score with high roughness is retained as a validation target, not discarded.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_radar_roughness_ambiguity_map(base: np.ndarray, candidate_mask: np.ndarray, patches: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels, _ = ndi.label(candidate_mask)
    risk_map = np.zeros(candidate_mask.shape, dtype=int)
    risk_to_class = {"Low": 1, "Medium": 2, "High": 3}
    for _, row in patches.iterrows():
        risk_map[labels == candidate_label_from_id(row["candidate_id"])] = risk_to_class.get(str(row.get("roughness_ambiguity_risk", "Low")), 1)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    cmap = ListedColormap(["#00000000", "#1b9e77", "#ffcc00", "#e74c3c"])
    ax.imshow(np.ma.masked_where(risk_map == 0, risk_map), cmap=cmap, vmin=0, vmax=3, alpha=0.72)
    top = patches.sort_values("validation_priority_rank").head(10) if "validation_priority_rank" in patches else patches.head(10)
    for _, row in top.iterrows():
        ax.text(row["centroid_col"] + 5, row["centroid_row"] + 5, row["candidate_id"], color="black", fontsize=8, weight="bold", bbox=dict(facecolor="white", edgecolor="black", alpha=0.78, pad=1.4))
    ax.legend(handles=[
        Line2D([0], [0], color="#e74c3c", lw=7, label="High roughness ambiguity"),
        Line2D([0], [0], color="#ffcc00", lw=7, label="Medium ambiguity"),
        Line2D([0], [0], color="#1b9e77", lw=7, label="Low ambiguity"),
    ], loc="lower right", framealpha=0.9)
    ax.set_title("Radar Roughness Ambiguity Map", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.025, "High radar response can also arise from rough terrain, multiple scattering, crater walls, or blocky ejecta; validation required.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_top_candidate_review_panel(review: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = review.sort_values("validation_priority_rank").head(8) if not review.empty else review
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_axis_off()
    ax.set_title("Top Candidate Patch Scientific Review", fontsize=15, fontweight="bold", pad=12)
    if not top.empty:
        cols = ["candidate_id", "mean_candidate_score", "mean_slope_deg", "roughness_ambiguity_risk", "threshold_stability_class", "confidence_level"]
        table_data = top[cols].copy()
        for col in ["mean_candidate_score", "mean_slope_deg"]:
            table_data[col] = table_data[col].astype(float).map(lambda x: f"{x:.3f}")
        table = ax.table(cellText=table_data.values, colLabels=[c.replace("_", " ") for c in cols], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
    fig.text(0.02, 0.04, "Review ranks candidate patches for validation priority; patch extent is not a measured ice body.", fontsize=9)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_resource_scenario_bar_chart(scenarios: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    if not scenarios.empty:
        top = scenarios[(scenarios["assumed_depth_m"] == 3) & (np.isclose(scenarios["assumed_ice_fraction"], 0.10))].head(10)
        ax.bar(top["candidate_id"], top["scenario_volume_m3"], color="#2f80ed")
    ax.set_title("Scenario-Based Potential Resource Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Candidate patch")
    ax.set_ylabel("Scenario volume (m3)")
    ax.tick_params(axis="x", rotation=30)
    fig.text(0.01, 0.01, "Scenario shown: 3 m assumed depth and 10% assumed ice fraction; planning-only estimate.", fontsize=8)
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


def save_landing_vs_f2_boundary_map(base: np.ndarray, candidate_mask: np.ndarray, landing_sites: pd.DataFrame, aoi: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(candidate_mask, 0.34, 0)
    ax.imshow(rgba)
    if not landing_sites.empty:
        ax.scatter(landing_sites["col"], landing_sites["row"], s=90, c="#00ff66", edgecolor="black", linewidth=0.8, label="preliminary landing candidate")
        for _, site in landing_sites.iterrows():
            ax.text(site["col"] + 4, site["row"] + 4, site["site_id"], color="white", fontsize=9, weight="bold")
    ax.set_title("Landing Candidates vs F2 Boundary Status", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.025, "Exact F2 crater boundary is not available locally; map shows SAR AOI/candidate-mask relationship and records boundary validation as future-required.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_validation_layer_availability_matrix(status: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    if not status.empty:
        labels = status["validation_layer"].astype(str)
        value_map = {"available": 2, "available_proxy": 1, "available_context_only": 1, "future_validation_layer_required": 0}
        values = status["status"].map(value_map).fillna(0).to_numpy(dtype=float).reshape(-1, 1)
        cmap = ListedColormap(["#ef9a9a", "#ffcc80", "#81c784"])
        ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels(["Availability"])
        for i, row in status.iterrows():
            ax.text(0, i, str(row["status"]).replace("_", " "), ha="center", va="center", fontsize=8, color="black")
    ax.set_title("External Validation Layer Availability Matrix", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Missing validation layers are not fabricated; they are listed as future-required inputs.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_hazard_overlay_on_terrain(base: np.ndarray, hazard_mask: np.ndarray, terrain_roughness: np.ndarray | None, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    background = terrain_roughness if terrain_roughness is not None else robust_normalize(base, np.isfinite(base))
    ax.imshow(background, cmap="gray" if terrain_roughness is None else "magma")
    rgba = np.zeros((*hazard_mask.shape, 4), dtype=float)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.18
    rgba[..., 2] = 0.0
    rgba[..., 3] = np.where(hazard_mask, 0.42, 0)
    ax.imshow(rgba)
    ax.set_title("Hazard Proxy Overlay on Terrain", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    ax.legend(handles=[Line2D([0], [0], color="#ff2e00", lw=7, alpha=0.6, label="rough/edge-rich proxy")], loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.025, note, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_tmc2_vs_lola_slope_difference(tmc_dem: dict[str, Any], dem: dict[str, Any], path: Path) -> None:
    if not isinstance(tmc_dem, dict) or not tmc_dem.get("available"):
        return
    tmc_slope = tmc_dem.get("slope_deg")
    lola_slope = dem.get("slope_deg") if isinstance(dem, dict) else None
    if tmc_slope is None or lola_slope is None:
        return
    diff = np.where(np.isfinite(tmc_slope) & np.isfinite(lola_slope), tmc_slope - lola_slope, np.nan)
    if not np.isfinite(diff).any():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95))
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    im = ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("Slope difference (deg), TMC-2 minus LOLA/LDEM/LDSM")
    ax.set_title("TMC-2 vs LOLA/LDEM/LDSM Slope Difference", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.02, "Terrain-model sensitivity check only; co-registration and map-projection validation are still required.", fontsize=8.5)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.88, bottom=0.09)
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


def save_depth_likelihood_map(shallow: np.ndarray, deep: np.ndarray, depth_class: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    panels = [
        ("Shallow likelihood", shallow, "YlGnBu", 0, 1),
        ("Deep likelihood", deep, "magma", 0, 1),
        ("Depth class", depth_class, ListedColormap(["#000000", "#1b9e77", "#6a3d9a", "#bdbdbd"]), 0, 3),
    ]
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    fig.suptitle("Depth Likelihood Proxy", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Rule: high radar + low roughness suggests shallow-likelihood; high radar + moderate roughness suggests deeper-likelihood. Validation required.", fontsize=8.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_science_justification_overlay(
    base: np.ndarray,
    ice_probability: np.ndarray,
    candidate_mask: np.ndarray,
    patches: pd.DataFrame,
    landing_sites: pd.DataFrame,
    routes: dict[str, list[tuple[int, int]]],
    route_summary: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    ax.imshow(np.ma.masked_invalid(ice_probability), cmap="viridis", alpha=0.55, vmin=0, vmax=1)
    rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(candidate_mask, 0.32, 0)
    ax.imshow(rgba)
    top_patch = selected_top_patch(patches)
    top_site = landing_sites.iloc[0] if not landing_sites.empty else None
    route_type = recommended_route_type(route_summary)
    route = routes.get(route_type, [])
    if route:
        rr = [p[0] for p in route]
        cc = [p[1] for p in route]
        ax.plot(cc, rr, color="#ffff00", lw=3.0, label=f"{route_type} route")
    if top_patch is not None:
        ax.scatter([top_patch["centroid_col"]], [top_patch["centroid_row"]], s=150, c="#ffeb3b", edgecolor="black", marker="*", zorder=5)
        text = (
            f"{top_patch['candidate_id']}: top refined patch\n"
            f"P={float(top_patch.get('mean_ice_probability', np.nan)):.3f}, "
            f"conf={float(top_patch.get('ice_confidence_score', np.nan)):.3f}\n"
            f"depth={top_patch.get('depth_likelihood_class', 'n/a')}"
        )
        ax.annotate(text, xy=(top_patch["centroid_col"], top_patch["centroid_row"]), xytext=(top_patch["centroid_col"] + 55, top_patch["centroid_row"] - 55), color="black", fontsize=8.5, bbox=dict(facecolor="white", alpha=0.86), arrowprops=dict(arrowstyle="->", color="white"))
    if top_site is not None:
        ax.scatter([top_site["col"]], [top_site["row"]], s=110, c="#00ff66", edgecolor="black", zorder=6)
        ax.annotate(f"{top_site['site_id']}: slope {float(top_site.get('slope_deg', np.nan)):.1f} deg\nnear candidate, outside mask", xy=(top_site["col"], top_site["row"]), xytext=(top_site["col"] + 45, top_site["row"] + 45), color="black", fontsize=8.5, bbox=dict(facecolor="white", alpha=0.86), arrowprops=dict(arrowstyle="->", color="#00ff66"))
    if not route_summary.empty:
        r = route_summary[route_summary["route_type"].astype(str).eq(route_type)].iloc[0]
        ax.text(0.015, 0.04, f"Route: {route_type} | {float(r.get('length_m', np.nan)):.0f} m | energy {float(r.get('energy_cost_proxy', np.nan)):.0f} | risk {float(r.get('traverse_risk_score', np.nan)):.3f}", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.88))
    ax.set_title("Science Justification Overlay", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.02, "Human-readable explanation layer: why this patch, landing site, and route are selected. Screening outputs; validation required.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.09)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def selected_top_patch(patches: pd.DataFrame) -> pd.Series | None:
    if patches.empty:
        return None
    kept = patches[patches.get("refinement_status", pd.Series("", index=patches.index)).astype(str).eq("kept_for_planning")]
    source = kept if not kept.empty else patches
    sort_cols = [c for c in ["refined_rank", "validation_priority_rank"] if c in source.columns]
    return source.sort_values(sort_cols, na_position="last").iloc[0] if sort_cols else source.iloc[0]


def save_annotated_probability_map(base: np.ndarray, ice_probability: np.ndarray, candidate_mask: np.ndarray, patches: pd.DataFrame, path: Path) -> None:
    top = selected_top_patch(patches)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    im = ax.imshow(np.ma.masked_invalid(ice_probability), cmap="viridis", alpha=0.72, vmin=0, vmax=1)
    ax.contour(candidate_mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.8)
    if top is not None:
        ax.scatter([top["centroid_col"]], [top["centroid_row"]], s=150, c="#ffeb3b", edgecolor="black", marker="*", zorder=5)
        ax.annotate(
            f"Why {top['candidate_id']}?\nHighest refined mean probability\nlow roughness + low slope\nP={float(top.get('mean_ice_probability', np.nan)):.3f}",
            xy=(top["centroid_col"], top["centroid_row"]),
            xytext=(top["centroid_col"] + 45, top["centroid_row"] - 45),
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color="white"),
        )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, label="candidate probability score")
    ax.set_title("Annotated Ice Candidate Probability Map", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.025, "Annotation explains ranking logic for judges; map remains a screening/proxy output.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.94, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_annotated_landing_map(
    base: np.ndarray,
    candidate_mask: np.ndarray,
    hazard_mask: np.ndarray,
    patches: pd.DataFrame,
    landing_sites: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    for mask, color in [(candidate_mask, (0.0, 0.85, 1.0, 0.35)), (hazard_mask, (1.0, 0.20, 0.0, 0.25))]:
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color[:3]
        rgba[..., 3] = np.where(mask, color[3], 0)
        ax.imshow(rgba)
    if not landing_sites.empty:
        site = landing_sites.iloc[0]
        ax.scatter([site["col"]], [site["row"]], s=120, c="#00ff66", edgecolor="black")
        ax.annotate(
            f"Why {site['site_id']}?\ninside AOI, outside candidate mask\nslope={float(site.get('slope_deg', np.nan)):.1f} deg\ncandidate distance={float(site.get('distance_to_candidate_m', np.nan)):.0f} m",
            xy=(site["col"], site["row"]),
            xytext=(site["col"] + 45, site["row"] + 45),
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color="#00ff66"),
        )
    ax.set_title("Annotated Landing Site Justification", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.025, "Landing site is preliminary: low-slope, low-hazard proxy, outside candidate patch, close enough for traverse.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_annotated_route_map(
    base: np.ndarray,
    candidate_mask: np.ndarray,
    hazard_mask: np.ndarray,
    landing_sites: pd.DataFrame,
    routes: dict[str, list[tuple[int, int]]],
    route_summary: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    for mask, color in [(candidate_mask, (0.0, 0.85, 1.0, 0.35)), (hazard_mask, (1.0, 0.20, 0.0, 0.25))]:
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = color[:3]
        rgba[..., 3] = np.where(mask, color[3], 0)
        ax.imshow(rgba)
    route_type = recommended_route_type(route_summary)
    for name, route in routes.items():
        if not route:
            continue
        rr = [p[0] for p in route]
        cc = [p[1] for p in route]
        color = "#ffff00" if name == route_type else "#00ffff"
        lw = 3.2 if name == route_type else 1.6
        ax.plot(cc, rr, color=color, lw=lw, label=name)
    if not landing_sites.empty:
        ax.scatter([landing_sites.iloc[0]["col"]], [landing_sites.iloc[0]["row"]], s=120, c="#00ff66", edgecolor="black")
    if not route_summary.empty:
        r = route_summary[route_summary["route_type"].astype(str).eq(route_type)].iloc[0]
        ax.text(0.015, 0.045, f"Recommended: {route_type}\nDecision score={float(r.get('route_decision_score', np.nan)):.3f}\nLength={float(r.get('length_m', np.nan)):.0f} m\nEnergy={float(r.get('energy_cost_proxy', np.nan)):.0f}\nRisk={float(r.get('traverse_risk_score', np.nan)):.3f}", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.88))
    ax.set_title("Annotated Rover Route Justification", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.text(0.01, 0.025, "Route optimality is multi-objective: distance, energy proxy, terrain risk, and science reward.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_rover_energy_profile(traversal: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    if not traversal.empty:
        ax.plot(traversal["distance_m"], traversal["cumulative_energy_proxy"], color="#2f80ed", lw=2.0)
    ax.set_title("Rover Traversal Energy Proxy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Traverse distance (m)")
    ax.set_ylabel("Cumulative energy proxy")
    fig.text(0.01, 0.01, "Energy proxy accumulates distance, slope penalty, and texture-risk penalty; not a rover power model.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_rover_slope_distance_profile(traversal: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    if not traversal.empty:
        ax.plot(traversal["distance_m"], traversal["slope_deg"], color="#f4511e", lw=1.8)
    ax.axhline(5, color="#1b9e77", ls="--", lw=1.0)
    ax.axhline(8, color="#ffcc00", ls="--", lw=1.0)
    ax.set_title("Rover Slope vs Distance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Traverse distance (m)")
    ax.set_ylabel("Slope (deg)")
    fig.text(0.01, 0.01, "Route slope profile sampled from selected terrain layer.", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_rover_traversal_steps(
    base: np.ndarray,
    candidate_mask: np.ndarray,
    landing_sites: pd.DataFrame,
    routes: dict[str, list[tuple[int, int]]],
    route_summary: pd.DataFrame,
    path: Path,
) -> None:
    route_type = recommended_route_type(route_summary)
    route = routes.get(route_type, [])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
    rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
    rgba[..., 1] = 0.85
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(candidate_mask, 0.32, 0)
    ax.imshow(rgba)
    if route:
        rr = [p[0] for p in route]
        cc = [p[1] for p in route]
        ax.plot(cc, rr, color="#ffff00", lw=2.5)
        step_ids = np.linspace(0, len(route) - 1, min(12, len(route)), dtype=int)
        ax.scatter([cc[i] for i in step_ids], [rr[i] for i in step_ids], s=36, c="#ff2bbd", edgecolor="black")
        for n, i in enumerate(step_ids):
            ax.text(cc[i] + 4, rr[i] + 4, str(n), fontsize=7, color="white", weight="bold")
    if not landing_sites.empty:
        ax.scatter([landing_sites.iloc[0]["col"]], [landing_sites.iloc[0]["row"]], s=100, c="#00ff66", edgecolor="black")
    ax.set_title("Step-by-Step Rover Traversal", fontsize=15, fontweight="bold")
    ax.set_axis_off()
    fig.text(0.01, 0.025, "Numbered markers show sampled rover progress along the recommended conceptual route.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_rover_traversal_animation(
    base: np.ndarray,
    candidate_mask: np.ndarray,
    landing_sites: pd.DataFrame,
    routes: dict[str, list[tuple[int, int]]],
    route_summary: pd.DataFrame,
    path: Path,
) -> None:
    route_type = recommended_route_type(route_summary)
    route = routes.get(route_type, [])
    if not route:
        return
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        path.parent.mkdir(parents=True, exist_ok=True)
        step_ids = np.linspace(0, len(route) - 1, min(90, len(route)), dtype=int)
        rr = [p[0] for p in route]
        cc = [p[1] for p in route]
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.imshow(robust_normalize(base, np.isfinite(base)), cmap="gray")
        rgba = np.zeros((*candidate_mask.shape, 4), dtype=float)
        rgba[..., 1] = 0.85
        rgba[..., 2] = 1.0
        rgba[..., 3] = np.where(candidate_mask, 0.28, 0)
        ax.imshow(rgba)
        ax.plot(cc, rr, color="#ffff00", lw=1.5, alpha=0.65)
        dot, = ax.plot([], [], marker="o", color="#ff2bbd", markersize=7)
        trail, = ax.plot([], [], color="#ff2bbd", lw=2.0)
        ax.set_title(f"Rover Traversal Animation - {route_type}", fontsize=12, fontweight="bold")
        ax.set_axis_off()

        def update(frame_idx):
            idx = step_ids[frame_idx]
            dot.set_data([cc[idx]], [rr[idx]])
            trail.set_data(cc[: idx + 1], rr[: idx + 1])
            return dot, trail

        anim = FuncAnimation(fig, update, frames=len(step_ids), interval=140, blit=True)
        anim.save(path, writer=PillowWriter(fps=7))
        plt.close(fig)
    except Exception:
        fallback = path.with_suffix(".png")
        save_rover_traversal_steps(base, candidate_mask, landing_sites, routes, route_summary, fallback)


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
    fig.text(0.01, 0.02, "Error classes compare against rule-based pseudo-labels only; independent validation labels are unavailable.", fontsize=9)
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
        "- U-Net metrics are pseudo-label agreement only; independent validation labels are not available.",
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
        "- [ ] State that there is no direct compositional proof in the current outputs.",
        "- [ ] Do not present any output as direct compositional proof.",
        "- [ ] State that true CPR/DOP are not yet claimed from the selected SRI intensity rasters.",
        "- [ ] State that current SAR features are screening proxies: intensity, LH/LV ratio proxy, imbalance proxy, texture, and candidate score.",
        "- [ ] State that OHRC is context-only because current OHRC products do not co-register with Faustini/F2 AOI.",
        "- [ ] State that U-Net is weakly supervised with rule-based pseudo-labels only.",
        "- [ ] State that pseudo-IoU and pseudo-Dice measure agreement with pseudo-labels, not composition accuracy.",
        "- [ ] State that landing candidates are preliminary and not certified landing products.",
        "- [ ] State that rover route variants are conceptual planning products only.",
        "- [ ] State that illumination, thermal, communication, and higher-confidence hazard/boulder layers remain future work.",
        "- [ ] State that candidate ranking is for review priority, while resource scenarios remain planning-only.",
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
    research_traceability: pd.DataFrame,
    validation_layer_status: pd.DataFrame,
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
        "## 3. Which Papers Guided Which Modules?",
        "",
        dataframe_to_markdown(research_traceability if not research_traceability.empty else pd.DataFrame()),
        "",
        "## 4. Extracted SAR Features",
        "",
        "- SAR log intensity.",
        "- LH channel and LV channel.",
        "- CPR-style LH/LV ratio proxy and LV/LH ratio proxy.",
        "- Polarization imbalance proxy.",
        "- Local texture / roughness, local mean, and local standard deviation.",
        "- Candidate score and threshold uncertainty.",
        "",
        "Calibrated CPR/DOP are not derived from the selected SRI intensity rasters.",
        "",
        "## 5. Ice Candidate Detection Before Landing Site Search",
        "",
        "The workflow now explicitly generates the radar-based candidate ice map before landing analysis. Candidate screening starts from SAR proxy features, applies thresholded candidate-score and channel constraints, removes small connected components, and then evaluates connected candidate patches before any landing-site search.",
        "",
        f"- Candidate patches generated: {int(candidate_summary.iloc[0]['candidate_patch_count']) if not candidate_summary.empty else 'n/a'}.",
        f"- Candidate area percentage of valid AOI pixels: {float(candidate_summary.iloc[0]['candidate_area_pct_of_valid_aoi']):.3f}%." if not candidate_summary.empty else "- Candidate area percentage unavailable.",
        "- Threshold sensitivity is saved to `outputs/tables/threshold_sensitivity.csv` and `outputs/figures/threshold_sensitivity_curve.png`.",
        "- Candidate stability across thresholds contributes to the `confidence_level` column.",
        "- Landing candidates are selected near evaluated candidate patches but outside the candidate mask and risky/steep zones.",
        "- This is a candidate screening layer for planning and validation.",
        "",
        "## 6. How Candidate Patches Were Generated",
        "",
        "- Candidate pixels are selected using SAR proxy thresholds for ratio, intensity, texture, and candidate score.",
        "- Connected components define candidate patches.",
        "- Each patch is evaluated for area, equivalent candidate patch diameter, score, ratio proxy, texture, slope context, uncertainty, and threshold stability.",
        "- Confidence levels are High/Medium/Low based on score, extent, stability, uncertainty, and slope/traverse context.",
        "",
        "## 7. Strongest Scientific Candidate Patches",
        "",
        dataframe_to_markdown(top_candidates[[
            "candidate_id", "area_m2", "equivalent_candidate_patch_diameter_m", "mean_candidate_score",
            "confidence_level", "roughness_ambiguity_risk", "threshold_stability_class",
            "mean_slope_deg", "max_slope_deg", "nearest_landing_site_id",
            "distance_to_nearest_landing_candidate_m", "validation_priority_rank",
        ]] if not top_candidates.empty else top_candidates),
        "",
        "## 8. Candidates With Steep Terrain Or Roughness Tradeoff",
        "",
        "Some high-score patches are less attractive as direct landing/traverse targets because their local slope context is steep. These can still be useful validation targets, but they should not drive landing-site selection without terrain review.",
        "",
        dataframe_to_markdown(steep_candidates[[
            "candidate_id", "mean_candidate_score", "area_m2", "mean_slope_deg", "max_slope_deg", "roughness_ambiguity_risk", "confidence_level",
        ]] if not steep_candidates.empty else steep_candidates),
        "",
        "## 9. How Candidate Confidence Was Calculated",
        "",
        "Candidate confidence combines mean candidate score, candidate patch extent, threshold stability, uncertainty near threshold, slope context, and a roughness ambiguity penalty. High roughness does not remove candidates; it marks them for OHRC, DEM, or multi-frequency validation.",
        "",
        "## 10. Scenario-Based Resource Estimate",
        "",
        "Resource scenarios use candidate patch area, assumed depth, and assumed ice fraction. These values are planning-only estimates for later-validated patches, not measured resource quantities.",
        "",
        dataframe_to_markdown(resource_scenarios.head(12) if not resource_scenarios.empty else resource_scenarios),
        "",
        "## 11. Best Landing Candidates And Why",
        "",
        dataframe_to_markdown(landing_eval.head(5) if not landing_eval.empty else landing_eval),
        "",
        f"Best current preliminary landing candidate: `{best_landing.get('site_id', 'n/a')}` with suitability score {best_landing.get('suitability_score', 'n/a')}. It is selected because it is low slope, near a candidate patch, outside the candidate mask, and within the current proxy safety constraints.",
        "",
        "## 12. Best Rover Route And Why",
        "",
        dataframe_to_markdown(rover_eval[[
            "route_type", "target_candidate_id", "start_landing_site_id", "length_m", "total_cost",
            "mean_slope_deg", "max_slope_deg", "percent_under_5deg", "percent_5_to_8deg", "percent_8_to_10deg", "percent_above_10deg",
            "science_reward_score", "energy_cost_proxy", "traverse_risk_score",
        ]] if not rover_eval.empty else rover_eval),
        "",
        f"Best current route by low risk and energy proxy: `{best_route.get('route_type', 'n/a')}`. This remains a conceptual rover route for planning, not operational rover command generation.",
        "",
        "## 13. What The U-Net Proves And Does Not Prove",
        "",
        f"- Pseudo-IoU: {metrics.get('pseudo_iou', 'n/a')}.",
        f"- Pseudo-Dice: {metrics.get('pseudo_dice', 'n/a')}.",
        "- These metrics measure agreement with rule-based pseudo-labels only.",
        "- They do not measure independently validated composition performance.",
        "",
        dataframe_to_markdown(model_experiments if not model_experiments.empty else pd.DataFrame()),
        "",
        "## 14. Validation Still Needed",
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
        "### External Validation Layer Status",
        "",
        dataframe_to_markdown(validation_layer_status),
        "",
        "### Slope Safety Summary",
        "",
        dataframe_to_markdown(pd.DataFrame([slope_stats]) if slope_stats else pd.DataFrame()),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_technical_limitations(path: Path, aoi: dict[str, Any]) -> None:
    lines = [
        "# Technical Limitations",
        "",
        "- No direct compositional proof is claimed from current outputs.",
        "- Current SAR products are real-valued SRI intensity rasters; true CPR/DOP are future validated modules unless complex/Stokes products and conventions are available.",
        "- Candidate maps are radar-based screening outputs and require independent validation.",
        "- Equivalent candidate patch diameter is an area-derived screening extent, not a compositional body diameter.",
        "- Resource scenario estimates are planning scenarios, not measured quantities.",
        "- OHRC is context-only because current products do not co-register with the configured Faustini/F2 AOI.",
        "- Landing candidates are preliminary and require illumination/PSR, thermal, communications, boulder, hazard, and manual terrain review.",
        "- Rover routes are conceptual route variants on proxy cost maps, not operational command products.",
        "- U-Net outputs are weakly supervised pseudo-label agreement results; independent validation labels are not available.",
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
        "   - Why needed: resolves small-scale hazards, blocky terrain, and boulder-like features if resolution supports it.",
        "   - Improves: roughness ambiguity, landing safety, rover corridor review.",
        "   - Enables: optical_hazard_proxy_map with actual AOI co-registration.",
        "   - Confidence effect: reduces radar roughness ambiguity for top candidate patches.",
        "2. Additional TMC-2 DTM co-registration QA or backup DTM products.",
        "   - Why needed: a TMC-2 DTM is now processed when it overlaps the AOI, but co-registration and terrain-model sensitivity still require review.",
        "   - Improves: slope safety, local relief, route risk, and terrain-model robustness.",
        "   - Enables: stronger TMC-2 vs LOLA/DEM slope comparison and backup terrain selection for an AOI switch.",
        "   - Confidence effect: checks whether candidate ranking is terrain-model sensitive.",
        "3. Diviner maximum temperature map for the Faustini/F2 AOI.",
        "   - Why needed: cold-trap suitability should use real thermal data.",
        "   - Improves: thermal validation and candidate confidence.",
        "   - Enables: temperature_validation_map and candidate thermal status.",
        "   - Confidence effect: supports or weakens cold-trap plausibility; use the ~110 K rule only with real data.",
        "4. PSR / illumination / shadow persistence layer.",
        "   - Why needed: separates candidate access from persistent shadow and rover power constraints.",
        "   - Improves: validation, fuzzy landing score, rover route planning.",
        "   - Enables: psr_shadow_validation_map and active illumination scoring.",
        "   - Confidence effect: prevents treating SAR candidates outside stable shadow as equally strong.",
        "5. LAMP / LOLA albedo / M3 validation layers if available.",
        "   - Why needed: adds PNAS-style independent surface-evidence context.",
        "   - Improves: external validation layer framework.",
        "   - Enables: albedo_validation_map and candidate_validation_against_external_layers updates.",
        "   - Confidence effect: supports surface-exposure context, not subsurface proof.",
        "6. Complex/Stokes SAR/DFSAR products for true CPR/DOP computation.",
        "   - Why needed: current selected layers are SRI intensities only.",
        "   - Improves: polarimetric feature extraction.",
        "   - Enables: calibrated CPR/DOP module after convention validation.",
        "   - Confidence effect: replaces CPR-style ratio proxy with defensible polarimetric features.",
        "7. F2 crater boundary / crater catalog shapefile.",
        "   - Why needed: rectangular AOI is not a crater boundary.",
        "   - Improves: landing boundary validation and crater-interior/rim reasoning.",
        "   - Enables: landing_vs_f2_crater_boundary_map with real boundary geometry.",
        "   - Confidence effect: prevents accepting a landing point only because it lies inside the rectangular AOI.",
        "",
        "## AOI For Search",
        "",
        f"- Latitude min: {aoi['lat_min']}",
        f"- Latitude max: {aoi['lat_max']}",
        f"- Longitude min: {aoi['lon_min']}",
        f"- Longitude max: {aoi['lon_max']} E",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_report(
    path: Path,
    aoi: dict[str, Any],
    coverage_validation: pd.DataFrame,
    usable_datasets: pd.DataFrame,
    tmc_dem: dict[str, Any],
    ohrc_fp: pd.DataFrame,
) -> None:
    selected_tmc = tmc_dem.get("selected_product_id", "") if isinstance(tmc_dem, dict) and tmc_dem.get("available") else ""
    ohrc_overlap = bool(
        not ohrc_fp.empty
        and ohrc_fp.get("coverage_fraction", pd.Series(dtype=float)).astype(float).gt(0).any()
    )
    full_sar = coverage_validation[
        coverage_validation["dataset_type"].astype(str).eq("SAR/DFSAR SRI")
        & coverage_validation["coverage_class"].astype(str).eq("FULL")
    ] if not coverage_validation.empty else pd.DataFrame()
    alternative_note = (
        "Current OHRC scenes cluster closer to -89 to -90 deg latitude. If the team chooses an AOI switch, "
        "prioritize south-pole craters near those OHRC footprints and then require SAR + TMC-2 overlap validation before analysis."
    )
    chosen_path = "A) Continue Faustini/F2 with SAR + TMC-2 terrain only" if not ohrc_overlap else "Use SAR + OHRC + TMC-2 for the configured AOI"
    lines = [
        "LunaQuest Decision Report",
        "=========================",
        "",
        f"AOI: {aoi.get('name', 'configured AOI')}",
        f"Latitude: {aoi['lat_min']} to {aoi['lat_max']}",
        f"Longitude: {aoi['lon_min']} to {aoi['lon_max']} E",
        "",
        "Chosen branch",
        "-------------",
        chosen_path,
        "",
        "Justification",
        "-------------",
        f"- FULL SAR/DFSAR SRI products covering the AOI: {len(full_sar)}.",
        f"- OHRC overlap with configured Faustini AOI: {'yes' if ohrc_overlap else 'no'}; OHRC is excluded from Faustini hazard scoring when overlap is zero.",
        f"- Selected TMC-2 DTM: {selected_tmc or 'none'}; used for slope/roughness if available.",
        "- Current rectangular AOI is a working analysis window, not a validated F2 crater boundary.",
        "- Radar layers are treated as candidate screening inputs only; no compositional claim is made.",
        "",
        "Usable datasets selected",
        "------------------------",
        dataframe_to_markdown(usable_datasets if not usable_datasets.empty else pd.DataFrame()),
        "",
        "Coverage validation",
        "-------------------",
        dataframe_to_markdown(coverage_validation if not coverage_validation.empty else pd.DataFrame()),
        "",
        "AOI switch option",
        "-----------------",
        alternative_note,
        "",
        "Limitations to carry into presentation",
        "--------------------------------------",
        "- CPR-style ratio proxy is not calibrated CPR/DOP.",
        "- Thermal/Diviner validation and PSR/illumination layers are missing.",
        "- OHRC does not overlap the configured Faustini AOI.",
        "- Candidate patch scores are data-driven screening results mixed with clearly marked assumptions such as neutral PSR placeholder.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_justification_report(
    path: Path,
    aoi: dict[str, Any],
    patches: pd.DataFrame,
    landing_sites: pd.DataFrame,
    route_summary: pd.DataFrame,
    rover_traversal: pd.DataFrame,
    ice_probability_layers: dict[str, Any],
) -> None:
    top_patch = selected_top_patch(patches)
    top_site = landing_sites.iloc[0] if not landing_sites.empty else None
    route_type = recommended_route_type(route_summary)
    route_row = route_summary[route_summary["route_type"].astype(str).eq(route_type)].iloc[0] if route_type and not route_summary.empty else None
    weights = ice_probability_layers.get("weights", {})

    lines = [
        "Science Justification Report",
        "============================",
        "",
        f"AOI: {aoi.get('name', 'configured AOI')} ({aoi['lat_min']} to {aoi['lat_max']} lat, {aoi['lon_min']} to {aoi['lon_max']} E lon)",
        "",
        "Core answer for judges",
        "----------------------",
        "This system selects a radar-based candidate patch, a nearby preliminary landing candidate, and a conceptual rover route using a chain of explainable screening criteria: radar response, CPR-style ratio proxy, roughness penalty, slope safety, PSR stability proxy, distance, and route energy/risk tradeoff.",
        "",
        "Important scientific boundary",
        "-----------------------------",
        "These outputs are candidate-screening and mission-planning support products. Real PSR/illumination, Diviner thermal, OHRC hazard, and calibrated CPR/DOP validation are still required before compositional or operational claims.",
        "",
        "Ice probability model",
        "---------------------",
        "Weighted score = CPR-style ratio proxy + SAR backscatter intensity + roughness suitability + approximate PSR stability proxy.",
        "Model weights:",
        *[f"- {k}: {v:.2f}" for k, v in weights.items()],
        "",
    ]

    if top_patch is not None:
        lines.extend([
            f"Why {top_patch['candidate_id']} is the top selected patch",
            "--------------------------------",
            f"- Refined rank: {format_value(top_patch.get('refined_rank'))}; validation priority rank: {format_value(top_patch.get('validation_priority_rank'))}.",
            f"- Mean candidate probability: {format_value(top_patch.get('mean_ice_probability'))}.",
            f"- Ice confidence score: {format_value(top_patch.get('ice_confidence_score'))}, computed from high radar score + low roughness + low slope.",
            f"- CPR-style / ratio contribution: {format_value(top_patch.get('mean_ratio_contribution'))}.",
            f"- SAR intensity contribution: {format_value(top_patch.get('mean_intensity_contribution'))}.",
            f"- Roughness-suitability contribution: {format_value(top_patch.get('mean_roughness_suitability_contribution'))}.",
            f"- PSR-proxy contribution: {format_value(top_patch.get('mean_psr_proxy_contribution'))}.",
            f"- Mean slope: {format_value(top_patch.get('mean_slope_deg'))} deg; terrain roughness proxy: {format_value(top_patch.get('mean_terrain_roughness'))}.",
            f"- Roughness ambiguity risk: {top_patch.get('roughness_ambiguity_risk', 'not_available')}.",
            f"- Depth likelihood: {top_patch.get('depth_likelihood_class', 'not_available')} (shallow={format_value(top_patch.get('shallow_ice_likelihood'))}, deep={format_value(top_patch.get('deep_ice_likelihood'))}).",
            f"- Why this matters: it balances strong radar screening response with low terrain penalty, so it is a better mission-planning target than a high-radar but rough/ambiguous patch.",
            "",
        ])

    if top_site is not None:
        outside_candidate_mask = "yes" if str(top_site.get("inside_candidate_mask_yes_no", "")).lower() == "no" else "needs validation"
        lines.extend([
            f"Why {top_site['site_id']} is the safest current landing candidate",
            "---------------------------------------------",
            f"- Coordinates: lat {float(top_site.get('lat')):.6f}, lon {float(top_site.get('lon')):.6f}.",
            f"- Inside configured AOI: {top_site.get('inside_configured_aoi_yes_no', 'not_available')}.",
            f"- Outside radar candidate mask: {outside_candidate_mask}.",
            f"- Local slope: {format_value(top_site.get('slope_deg'))} deg; slope score: {format_value(top_site.get('score_slope'))}.",
            f"- Roughness/hazard score: {format_value(top_site.get('score_roughness'))}.",
            f"- Candidate proximity score: {format_value(top_site.get('score_candidate_proximity'))}; distance to nearest candidate: {format_value(top_site.get('distance_to_candidate_m'))} m.",
            "- Why this matters: landing is kept outside the candidate patch and risky terrain, but close enough that the rover can traverse to the science target.",
            "",
        ])

    if route_row is not None:
        lines.extend([
            f"Why the {route_type} rover route is recommended",
            "--------------------------------",
            f"- Target candidate patch: {route_row.get('target_candidate_id', 'not_available')}.",
            f"- Route decision score: {format_value(route_row.get('route_decision_score'))}; rank: {format_value(route_row.get('route_decision_rank'))}.",
            f"- Length: {format_value(route_row.get('length_m'))} m.",
            f"- Energy proxy: {format_value(route_row.get('energy_cost_proxy'))}.",
            f"- Traverse risk score: {format_value(route_row.get('traverse_risk_score'))}.",
            f"- Mean slope: {format_value(route_row.get('mean_slope_deg'))} deg; percent under 5 deg: {format_value(route_row.get('percent_under_5deg'))}%.",
            f"- Science reward proxy: {format_value(route_row.get('science_reward_score'))}.",
            "- Why this matters: the route recommendation is not simply shortest-distance; it balances distance, energy proxy, terrain risk, and science reward.",
            "- Note: if multiple route variants tie on geometry and risk, the first ranked route is reported while the alternatives remain available for review.",
            "",
        ])

    if not rover_traversal.empty:
        lines.extend([
            "Rover simulation summary",
            "------------------------",
            f"- Simulated steps: {len(rover_traversal)}.",
            f"- Final distance: {format_value(rover_traversal['distance_m'].iloc[-1])} m.",
            f"- Final cumulative energy proxy: {format_value(rover_traversal['cumulative_energy_proxy'].iloc[-1])}.",
            f"- Max sampled slope: {format_value(rover_traversal['slope_deg'].max())} deg.",
            "- Output figures: rover_energy_profile.png, rover_slope_vs_distance.png, rover_traversal_steps.png, rover_traversal_animation.gif.",
            "",
        ])

    lines.extend([
        "Data-driven results vs assumptions",
        "----------------------------------",
        "- Data-driven: SAR SRI intensity features, CPR-style ratio proxy, TMC-2 DTM slope/roughness, candidate patch geometry, landing distance, and route cost profiles.",
        "- Assumption/proxy: PSR stability proxy from latitude plus terrain-shadow context; this is not a substitute for a real illumination or PSR product.",
        "- Assumption/proxy: depth likelihood classes are rule-based planning labels from radar score and roughness, not measured depth.",
        "- Required validation: overlapping OHRC, Diviner thermal map, PSR/illumination layer, LAMP/LOLA/M3 context, and calibrated CPR/DOP products.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "not_available"
    if abs(val) >= 100:
        return f"{val:.1f}"
    return f"{val:.3f}"


def write_research_paper_method_map(path: Path, traceability: pd.DataFrame, reference_inventory: pd.DataFrame) -> None:
    lines = [
        "# Research Paper Method Map",
        "",
        "This report maps each uploaded method reference to an implementable LunaQuest module. Duplicate PDF copies are recorded but ignored for method integration.",
        "",
        "## Method Traceability",
        "",
        dataframe_to_markdown(traceability if not traceability.empty else pd.DataFrame()),
        "",
        "## Duplicate Handling",
        "",
        dataframe_to_markdown(reference_inventory[[
            "filename", "paper_title", "duplicate_group", "duplicate_status",
            "canonical_reference_filename", "sha256_12", "exists", "used_in_pipeline",
        ]] if not reference_inventory.empty else pd.DataFrame()),
        "",
        "## Language Guardrails",
        "",
        "- Radar output is a candidate screening result and requires validation.",
        "- CPR-style ratio proxy is not calibrated CPR/DOP.",
        "- Equivalent candidate patch diameter is a patch extent metric.",
        "- Resource scenarios are planning-only estimates.",
        "- U-Net metrics are pseudo-label agreement metrics.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_research_references_used(path: Path, reference_inventory: pd.DataFrame) -> None:
    used = reference_inventory[reference_inventory["used_in_pipeline"].astype(bool)] if not reference_inventory.empty else pd.DataFrame()
    ignored = reference_inventory[~reference_inventory["used_in_pipeline"].astype(bool)] if not reference_inventory.empty else pd.DataFrame()
    lines = [
        "# Research References Used",
        "",
        "## Used Method References",
        "",
        dataframe_to_markdown(used[["filename", "paper_title", "source", "sha256_12", "path", "exists"]] if not used.empty else used),
        "",
        "## Ignored Duplicate Or Missing Files",
        "",
        dataframe_to_markdown(ignored[[
            "filename", "paper_title", "duplicate_status",
            "canonical_reference_filename", "sha256_12", "path", "exists",
        ]] if not ignored.empty else ignored),
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
        md("## A. Problem Overview\n\nThis project screens Chandrayaan-2 SAR/DFSAR observations for radar-based candidate subsurface ice signatures near the lunar south pole, then connects those candidates to preliminary landing-site and conceptual rover-route planning. The configured focus region is Faustini/F2 because it is a high-priority south-polar candidate area. Outputs are screening results, not compositional proof, certified landing products, or operational rover routes."),
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
        md("## I. Landing Site Suitability\n\nLanding suitability combines low slope, terrain safety proxy, proximity to radar candidates, and neutral placeholders for missing external layers. Future illumination, thermal, and communication layers should replace those placeholders."),
        code("landing = pd.read_csv(tables / 'landing_sites.csv')\ndisplay(landing[['site_id','lat','lon','suitability_score','slope_deg','distance_to_candidate_m','safe_wording']])\nfor name in ['09_landing_suitability_map.png','10_top_landing_candidates_overlay.png','landing_score_comparison.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## J. Rover Traversal Planning\n\nThe cost map blocks slope >15 deg and generates shortest, safest, and science-priority conceptual route variants from the top landing candidate to a target candidate patch."),
        code("route_summary = pd.read_csv(routes_dir / 'route_summary.csv')\ndisplay(route_summary[['route_type','status','target_candidate_id','length_m','cost','mean_slope_deg','max_slope_deg']])\nfor name in ['11_rover_route_comparison.png','12_rover_route_overlay.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## K. OHRC Context and Footprint Issue\n\nThe current OHRC products are context-only. They do not directly co-register to the configured Faustini AOI, so they are not used as Faustini hazard/boulder maps."),
        code("ohrc = pd.read_csv(tables / 'ohrc_footprints.csv')\ndisplay(ohrc[['product_id','pixel_resolution_m','lat_min','lat_max','overlaps_faustini_lat','coverage_note']])\ndisplay(Image(filename=str(figdir / '13_ohrc_context_hazard_proxy.png')))\nprint((root / 'reports' / 'OHRC_DATA_DOWNLOAD_NEEDED.md').read_text())"),
        md("## L. Weakly Supervised U-Net Prototype\n\nThe U-Net uses five channels: SAR intensity, LH/LV ratio proxy, texture, polarization imbalance proxy, and rule candidate score. Labels are pseudo-labels from the rule-based mask. Spatial tile split is used, not random pixels."),
        code("metrics = pd.read_csv(tables / 'unet_pseudo_label_metrics.csv')\nhistory = pd.read_csv(tables / 'unet_training_history.csv')\ntiles_df = pd.read_csv(tables / 'unet_tile_inventory.csv')\ndisplay(metrics.T)\ndisplay(history.tail())\ndisplay(tiles_df.groupby('split')[['positive_pixels','positive_fraction']].agg(['count','sum','mean']))\nfor name in ['14_unet_training_curve.png','15_unet_prediction_overlay.png','pseudo_label_distribution.png']:\n    display(Image(filename=str(figdir / name)))\nprint('Limitation: these metrics measure agreement with pseudo-labels; independent validation labels are unavailable.')"),
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
        md("# LunaQuest / Lunar IceNav Research Notebook\n\nResearch notebook for BAH 2026 Problem Statement 8. Outputs are radar-based candidate screening products, scientific review tables, preliminary landing candidates, conceptual rover routes, pseudo-label ML experiments, and validation reports."),
        md("## A. Research Objective and C8 Problem\n\nDetect radar-based candidate subsurface ice regions in the lunar south polar Faustini/F2 AOI, characterize candidate patch extent and uncertainty, and connect candidates to landing and rover planning constraints."),
        code(root_cell),
        md("## B. Research Papers and Method Traceability\n\nMaps each uploaded method reference to the module it supports, with duplicate copies ignored."),
        code("import pandas as pd\nfrom IPython.display import display, Image\nimport json\ntrace = pd.read_csv(tables / 'research_method_traceability.csv')\ndisplay(trace)\nprint((root / 'reports' / 'RESEARCH_PAPER_METHOD_MAP.md').read_text()[:4000])"),
        md("## C. Region Validation: Faustini/F2 AOI\n\nConfigured AOI: lat -87.8 to -86.9, lon 80 to 85 E. Product selection is based on actual AOI coverage."),
        code("manifest = json.loads((root / 'reports' / 'run_manifest.json').read_text())\nprint('AOI:', manifest['aoi'])\ndisplay(pd.read_csv(tables / 'sar_product_selection_reason.csv')[['product_id','coverage_fraction','selected_for_main_map','selection_reason']])\ndisplay(Image(filename=str(figdir / 'region_validation_map.png')))"),
        md("## D. Dataset Inventory and Product Selection\n\nInventory, coverage, and data-status tables record what was used and what remains context-only or future-required."),
        code("inv = pd.read_csv(tables / 'product_inventory.csv')\ndisplay(inv['product_type'].value_counts().reset_index(name='rows'))\ndisplay(pd.read_csv(tables / 'data_coverage_status.csv'))\ndisplay(Image(filename=str(figdir / '01_dataset_inventory_chart.png')))\ndisplay(Image(filename=str(figdir / '02_selected_sar_coverage.png')))"),
        md("## E. SAR Feature Extraction\n\nFeatures include SAR log intensity, LH/LV and LV/LH ratio proxies, polarization imbalance proxy, local texture, local mean/std, and candidate score. Calibrated CPR/DOP require future complex/Stokes data."),
        code("sar_meta = pd.read_csv(tables / 'selected_sar_metadata.csv')\ndisplay(sar_meta.T)\nfeatures = __import__('numpy').load(root / 'outputs' / 'features' / 'sar_feature_stack.npz')\nprint('Feature arrays:', list(features.keys()))\ndisplay(Image(filename=str(figdir / 'radar_feature_panel.png')))"),
        md("## F. Radar-Based Candidate Ice Map\n\nThe candidate map is generated before landing analysis from SAR screening proxies and connected components."),
        code("candidate_summary = pd.read_csv(tables / 'candidate_patch_summary.csv')\ndisplay(candidate_summary)\ndisplay(Image(filename=str(figdir / 'ice_candidate_detection_map.png')))\ndisplay(Image(filename=str(figdir / 'radar_candidate_overlay.png')))"),
        md("## G. Candidate Patch Evaluation\n\nEach patch is evaluated for area, equivalent candidate patch diameter, score, ratio proxy, texture, slope, confidence, landing proximity, and route accessibility."),
        code("review = pd.read_csv(tables / 'candidate_scientific_review.csv')\ndisplay(review.head(15))\nfor name in ['ice_candidate_confidence_map.png','candidate_area_vs_score_scatter.png','candidate_diameter_distribution.png','top_candidate_review_panel.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## G2. Science Justification Layer\n\nJudge-facing explanations connect the selected patch, landing site, and rover route to the actual screening metrics."),
        code("patch_table = pd.read_csv(tables / 'candidate_patch_table.csv')\ndisplay(patch_table.head(10))\nprint((root / 'reports' / 'justification_report.txt').read_text())\nfor name in ['science_justification_overlay.png','ice_probability_map_annotated.png','landing_site_map_annotated.png','rover_routes_annotated.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## H. Radar Ambiguity and Roughness Penalty\n\nHigh radar response can be caused by rough terrain or multiple scattering, so candidates are retained but flagged with roughness ambiguity risk."),
        code("patch_review = pd.read_csv(tables / 'ice_candidate_patch_review.csv')\ndisplay(patch_review.head(15))\nfor name in ['radar_roughness_ambiguity_map.png','candidate_score_vs_roughness.png','candidate_score_vs_slope.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## I. Threshold Sensitivity and Confidence\n\nCandidate stability is evaluated across score thresholds from 0.60 to 0.85 and contributes to confidence."),
        code("thresh = pd.read_csv(tables / 'threshold_sensitivity.csv')\ndisplay(thresh)\nfor name in ['threshold_sensitivity_curve.png','candidate_stability_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## I2. PSR Proxy and Depth Likelihood\n\nThe PSR layer here is an approximation from poleward latitude plus terrain-shadow context. Depth likelihood is a rule-based planning label from radar score and roughness."),
        code("for name in ['psr_stability_proxy_map.png','ice_confidence_score_map.png','depth_likelihood_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## J. Scenario-Based Resource Estimate\n\nResource scenarios are planning-only estimates for later-validated candidate patches."),
        code("resource = pd.read_csv(tables / 'resource_scenario_estimates.csv')\ndisplay(resource.head(24))\ndisplay(Image(filename=str(figdir / 'resource_scenario_bar_chart.png')))"),
        md("## K. DEM/DTM Slope and Roughness\n\nDEM-derived slope, roughness, and hillshade support terrain safety and route screening."),
        code("slope_summary = pd.read_csv(tables / 'slope_safety_summary.csv')\ndisplay(slope_summary)\nfor name in ['dem_elevation_map.png','dtm_slope_map.png','slope_classification_map.png','terrain_roughness_map.png','hillshade_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## L. Fuzzy Landing Site Scoring\n\nThe landing model uses candidate proximity, slope safety, roughness avoidance, and neutral placeholders for missing illumination/temperature layers."),
        code("landing = pd.read_csv(tables / 'fuzzy_landing_site_scores.csv')\ndisplay(landing)\nfor name in ['fuzzy_landing_score_map.png','landing_score_components.png','landing_candidate_decision_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## M. Landing Site vs Crater/Candidate-Mask Validation\n\nLanding candidates are checked against configured AOI, candidate mask, unsafe slope, and F2 boundary availability."),
        code("boundary = pd.read_csv(tables / 'landing_crater_boundary_check.csv')\ndisplay(boundary)\nfor name in ['landing_vs_f2_crater_boundary_map.png','landing_to_candidate_distance_map.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## N. Rover Navigation and Route Evaluation\n\nA* route variants compare shortest, safest, science-priority, and energy-aware planning concepts using available slope/roughness/science proxy layers."),
        code("rover = pd.read_csv(tables / 'rover_navigation_evaluation.csv')\ndisplay(rover[['route_type','route_recommendation','route_decision_score','target_candidate_id','start_landing_site_id','length_m','total_cost','percent_under_5deg','science_reward_score','energy_cost_proxy','traverse_risk_score']])\nfor name in ['rover_route_decision_map.png','rover_route_slope_profile.png','rover_route_risk_profile.png','rover_route_comparison_chart.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## N2. Rover Traversal Simulation\n\nThe recommended route is sampled step-by-step with cumulative energy proxy and slope-vs-distance profiles."),
        code("traversal = pd.read_csv(tables / 'rover_traversal_simulation.csv')\ndisplay(traversal.head())\ndisplay(traversal.tail())\nfor name in ['rover_energy_profile.png','rover_slope_vs_distance.png','rover_traversal_steps.png']:\n    display(Image(filename=str(figdir / name)))\nprint('Animation:', figdir / 'rover_traversal_animation.gif')"),
        md("## O. U-Net Pseudo-Label Experiment\n\nThe U-Net is weakly supervised with rule-based pseudo-labels. Metrics are pseudo-label agreement metrics."),
        code("metrics = pd.read_csv(tables / 'unet_pseudo_label_metrics.csv')\nexperiments = pd.read_csv(tables / 'model_experiment_comparison.csv')\ndisplay(metrics.T)\ndisplay(experiments)\nfor name in ['unet_training_curve.png','model_experiment_comparison.png','unet_prediction_overlay.png','unet_error_map_against_pseudolabel.png']:\n    display(Image(filename=str(figdir / name)))"),
        md("## P. External Validation Layers Status\n\nDiviner, PSR/illumination, LOLA albedo, LAMP, and M3 layers are not fabricated; missing data are listed as future-required."),
        code("validation = pd.read_csv(tables / 'validation_layer_status.csv')\ncandidate_validation = pd.read_csv(tables / 'candidate_validation_against_external_layers.csv')\ndisplay(validation)\ndisplay(candidate_validation.head(15))\ndisplay(Image(filename=str(figdir / 'validation_layer_availability_matrix.png')))"),
        md("## Q. Final Research Evaluation Summary\n\nThe evaluation report links data, methods, candidate patches, landing scoring, route selection, ML agreement, limitations, and validation needs."),
        code("print((root / 'reports' / 'MODEL_EVALUATION_REPORT.md').read_text()[:7000])\ndisplay(Image(filename=str(figdir / 'combined_research_decision_map.png')))"),
        md("## R. Technical Limitations and Next Data\n\nLimitations and next-download priorities are explicit so the prototype does not overclaim."),
        code("print((root / 'reports' / 'TECHNICAL_LIMITATIONS.md').read_text())\nprint('\\n' + '='*80 + '\\n')\nprint((root / 'reports' / 'NEXT_DATA_TO_DOWNLOAD.md').read_text())"),
    ]


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)
