# Model Evaluation Report

This report evaluates the current LunaQuest / Lunar IceNav research prototype. It focuses on radar-based candidate patch detection, candidate patch scientific review, landing-site dependency on the candidate map, rover navigation evaluation, and pseudo-label ML agreement.

## 1. What Data Was Processed?

- Inventory rows processed: 245.
- Product classes include SAR/DFSAR, DEM/topography, OHRC bundles, and documents.
- OHRC remains context-only until a Faustini/F2-overlapping calibrated product is downloaded and co-registered.

## 2. Selected SAR Product And Reason

- Selected SAR product: `ch2_sar_ncls_20200808t201154198_d_cp_d18`.
- Configured AOI coverage fraction: 1.000.
- AOI: lat -87.8 to -86.9, lon 80.0 to 85.0 E.
- Partial-overlap products are supporting only; no-overlap products are excluded from the main candidate map.

## 3. Extracted SAR Features

- SAR log intensity.
- LH channel and LV channel.
- CPR-style LH/LV ratio proxy and LV/LH ratio proxy.
- Polarization imbalance proxy.
- Local texture / roughness, local mean, and local standard deviation.
- Candidate score and threshold uncertainty.

True CPR/DOP are not claimed from the selected SRI intensity rasters.

## Ice Candidate Detection Before Landing Site Search

The workflow now explicitly generates the radar-based candidate ice map before landing analysis. Candidate screening starts from SAR proxy features, applies thresholded candidate-score and channel constraints, removes small connected components, and then evaluates connected candidate patches before any landing-site search.

- Candidate patches generated: 65.
- Candidate area percentage of valid AOI pixels: 0.255%.
- Threshold sensitivity is saved to `outputs/tables/threshold_sensitivity.csv` and `outputs/figures/threshold_sensitivity_curve.png`.
- Candidate stability across thresholds contributes to the `confidence_level` column.
- Landing candidates are selected near evaluated candidate patches but outside the candidate mask and risky/steep zones.
- This is a planning layer, not confirmed ice detection.

## 4. How Candidate Patches Were Generated

- Candidate pixels are selected using SAR proxy thresholds for ratio, intensity, texture, and candidate score.
- Connected components define candidate patches.
- Each patch is evaluated for area, equivalent candidate patch diameter, score, ratio proxy, texture, slope context, uncertainty, and threshold stability.
- Confidence levels are High/Medium/Low based on score, extent, stability, uncertainty, and slope/traverse context.

## 5. Strongest Scientific Candidate Patches

| candidate_id | area_m2 | equivalent_candidate_patch_diameter_m | mean_candidate_score | confidence_level | mean_slope_deg | max_slope_deg | nearest_landing_site_id | distance_to_nearest_landing_candidate_m | validation_priority_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-053 | 9375.0 | 109.25484305920791 | 0.7999980449676514 | High | 2.3299899101257324 | 3.6580698490142822 | L-04 | 948.5383492510991 | 1 |
| C-033 | 8750.0 | 105.5502061411188 | 0.7978206872940063 | High | 2.2419257164001465 | 2.5731451511383057 | L-02 | 406.5942419058838 | 2 |
| C-043 | 8750.0 | 105.5502061411188 | 0.775698184967041 | High | 2.23689341545105 | 2.5564935207366943 | L-01 | 241.34244025738184 | 3 |
| C-007 | 10000.0 | 112.83791670955127 | 0.6992507576942444 | High | 0.9593207240104675 | 2.3338022232055664 | L-03 | 5794.104796310212 | 4 |
| C-006 | 10000.0 | 112.83791670955127 | 0.667952835559845 | High | 1.9872174263000488 | 2.4750137329101562 | L-03 | 5533.962302865032 | 5 |
| C-009 | 8750.0 | 105.5502061411188 | 0.7354375720024109 | High | 3.8307299613952637 | 4.331850528717041 | L-02 | 2772.873334758053 | 6 |
| C-019 | 9375.0 | 109.25484305920791 | 0.6692438721656799 | High | 3.071352958679199 | 4.354031085968018 | L-03 | 3052.0052606624236 | 7 |
| C-027 | 8125.0 | 101.71072362820549 | 0.7512441277503967 | High | 1.2572503089904785 | 1.7135846614837646 | L-02 | 184.62540036933834 | 8 |

## 6. Candidates With Steep Terrain Tradeoff

Some high-score patches are less attractive as direct landing/traverse targets because their local slope context is steep. These can still be useful validation targets, but they should not drive landing-site selection without terrain review.

| candidate_id | mean_candidate_score | area_m2 | mean_slope_deg | max_slope_deg | confidence_level |
| --- | --- | --- | --- | --- | --- |
| C-040 | 0.8557230234146118 | 11250.0 | 17.776716232299805 | 21.397464752197266 | Medium |
| C-063 | 0.7351826429367065 | 10000.0 | 12.609016418457031 | 13.49665641784668 | Low |
| C-038 | 0.8308277130126953 | 5625.0 | 23.512035369873047 | 24.73196029663086 | Low |
| C-065 | 0.7794984579086304 | 5625.0 | 16.42576026916504 | 17.16067886352539 | Low |
| C-029 | 0.75523841381073 | 5625.0 | 16.22152328491211 | 17.442949295043945 | Low |
| C-001 | 0.6917432546615601 | 5625.0 | 10.70478630065918 | 13.889681816101074 | Low |
| C-005 | 0.6852017045021057 | 5625.0 | 14.175182342529297 | 14.610235214233398 | Low |

## 7. Best Landing Candidates And Why

| site_id | lat | lon | slope_deg | suitability_score | distance_to_target_candidate_m | nearest_candidate_id | nearest_candidate_confidence_level | route_accessible | reason_selected | validation_needed | low_slope_score | candidate_proximity_score | roughness_avoidance_status | candidate_mask_clearance_status | illumination_layer_status | thermal_layer_status | communication_layer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L-01 | -87.4194372574005 | 81.10705291605449 | 0.6107425689697266 | 0.9552358388900757 | 75.0 | C-045 | Medium | needs route check | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 0.9491 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | future data layer needed | future data layer needed | future line-of-sight layer needed |
| L-02 | -87.43773078418651 | 79.97776407787147 | 0.4877799153327942 | 0.94283527135849 | 100.0 | C-027 | High | needs route check | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 0.9594 | 0.9091 | included via SAR texture proxy | outside candidate mask with configured clearance | future data layer needed | future data layer needed | future line-of-sight layer needed |
| L-03 | -87.23732364901072 | 80.8502874234031 | 0.3548099994659424 | 0.9405280947685242 | 127.4754878398196 | C-031 | Medium | needs route check | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 0.9704 | 0.8869 | included via SAR texture proxy | outside candidate mask with configured clearance | future data layer needed | future data layer needed | future line-of-sight layer needed |
| L-04 | -87.3474897938974 | 81.54828764071902 | 0.7538388967514038 | 0.9397911429405212 | 75.0 | C-049 | Medium | needs route check | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 0.9372 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | future data layer needed | future data layer needed | future line-of-sight layer needed |
| L-05 | -87.42151244053373 | 81.65561964956177 | 1.0223958492279053 | 0.9377585053443909 | 75.0 | C-058 | Medium | needs route check | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 0.9148 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | future data layer needed | future data layer needed | future line-of-sight layer needed |

Best current preliminary landing candidate: `L-01` with suitability score 0.9552358388900757. It is selected because it is low slope, near a candidate patch, outside the candidate mask, and within the current proxy safety constraints.

## 8. Best Rover Route And Why

| route_type | target_candidate_id | start_landing_site_id | length_m | total_cost | mean_slope_deg | max_slope_deg | percent_route_under_5deg | percent_route_above_10deg | science_reward_score | energy_cost_proxy | traverse_risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shortest | C-040 | L-01 | 422.48737341529164 | 18.230670928955078 | 3.6776732444763183 | 14.685937881469727 | 80.0 | 13.333333333333334 | 0.6405978778998057 | 1222.2823294376994 | 0.3676793469799061 |
| safest | C-040 | L-01 | 422.48737341529164 | 32.46928787231445 | 3.6776732444763183 | 14.685937881469727 | 80.0 | 13.333333333333334 | 0.6405978778998057 | 1222.2823294376994 | 0.3676793469799061 |
| science_priority | C-040 | L-01 | 422.48737341529164 | 26.459871292114258 | 3.724302617708842 | 14.685937881469727 | 80.0 | 13.333333333333334 | 0.7469504117965698 | 1227.3169252941425 | 0.3678024684617089 |
| energy_efficient | C-040 | L-01 | 422.48737341529164 | 27.756969451904297 | 3.6776732444763183 | 14.685937881469727 | 80.0 | 13.333333333333334 | 0.6405978778998057 | 1222.2823294376994 | 0.3676793469799061 |

Best current route by low risk and energy proxy: `shortest`. This remains a conceptual rover route for planning, not operational rover command generation.

## 9. What The U-Net Proves And Does Not Prove

- Pseudo-IoU: 0.26524685382381413.
- Pseudo-Dice: 0.41928079571537874.
- These metrics measure agreement with rule-based pseudo-labels only.
- They do not measure independently validated ice detection accuracy.

| experiment | input_channels | pseudo_iou | pseudo_dice | pseudo_precision | pseudo_recall | prediction_fraction | pseudo_label_fraction | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyUNet_5channel_weighted_loss | SAR log intensity, ratio proxy, texture, polarization imbalance proxy, candidate score | 0.26524685382381413 | 0.41928079571537874 | 0.41896024464831805 | 0.41960183767228176 | 0.0013747351427706588 | 0.0013726331012679513 | weakly supervised pseudo-label agreement only |
| candidate_score_quantile_baseline | candidate score threshold baseline | 0.06606384941871193 | 0.12393976112168946 | 0.06986729117876658 | 0.5482388973966309 | 0.010770860659872868 | 0.0013726331012679513 | baseline pseudo-label agreement only |
| candidate_score_strict_threshold | candidate score threshold baseline | 0.01372627330628718 | 0.0270808277692531 | 0.01372627330628718 | 1.0 | 0.10000042040830054 | 0.0013726331012679513 | baseline pseudo-label agreement only |

## 10. Validation Still Needed

- Download and co-register calibrated OHRC for the configured Faustini/F2 AOI.
- Add illumination/PSR, thermal, and communication layers.
- Validate CPR/DOP only with correct product convention or complex/Stokes products.
- Replace pseudo-labels with stronger labels or multi-pass scientific consistency checks.
- Manually review top candidate patches and route corridors.

## Supporting Tables

### Data Coverage Status

| layer | status | detail |
| --- | --- | --- |
| SAR/DFSAR | usable | Selected ch2_sar_ncls_20200808t201154198_d_cp_d18 with AOI coverage 1.000 |
| DEM/LDEM/LDSM | usable | DEM rasters cover configured south-pole AOI and were resampled to SAR AOI for planning context. |
| OHRC | context only | Available OHRC footprints do not directly co-register to Faustini AOI; download calibrated overlapping product. |
| Ground truth labels | not available | U-Net uses rule-based pseudo-labels; metrics are agreement only. |

### Slope Safety Summary

| valid_slope_pixels | mean_slope_deg | median_slope_deg | safe_lt_5deg_pct | moderate_5_10deg_pct | unsafe_gt_10deg_pct | blocked_gt_15deg_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 256181.0 | 9.301920890808105 | 8.23563289642334 | 37.9852526143625 | 16.36499193929292 | 45.64975544634458 | 28.700801386519686 |

### Resource Scenario Summary

Scenario estimates are potential planning cases only if candidate patches are later validated.

| candidate_id | confidence_level | candidate_area_m2 | assumed_depth_m | assumed_ice_fraction | approx_candidate_volume_m3 | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| C-053 | High | 9375.0 | 1 | 0.05 | 468.75 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 1 | 0.1 | 937.5 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 1 | 0.2 | 1875.0 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 3 | 0.05 | 1406.25 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 3 | 0.1 | 2812.5 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 3 | 0.2 | 5625.0 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 5 | 0.05 | 2343.75 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 5 | 0.1 | 4687.5 | scenario-based potential resource volume if the candidate patch is later validated |
| C-053 | High | 9375.0 | 5 | 0.2 | 9375.0 | scenario-based potential resource volume if the candidate patch is later validated |
| C-033 | High | 8750.0 | 1 | 0.05 | 437.5 | scenario-based potential resource volume if the candidate patch is later validated |
| C-033 | High | 8750.0 | 1 | 0.1 | 875.0 | scenario-based potential resource volume if the candidate patch is later validated |
| C-033 | High | 8750.0 | 1 | 0.2 | 1750.0 | scenario-based potential resource volume if the candidate patch is later validated |