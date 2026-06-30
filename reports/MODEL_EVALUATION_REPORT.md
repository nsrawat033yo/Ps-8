# Model Evaluation Report

This report evaluates the current LunaQuest / Lunar IceNav research prototype. It focuses on radar-based candidate patch detection, candidate patch scientific review, landing-site dependency on the candidate map, rover navigation evaluation, and pseudo-label ML agreement.

## 1. What Data Was Processed?

- Inventory rows processed: 1223.
- Product classes include SAR/DFSAR, DEM/topography, OHRC bundles, and documents.
- OHRC remains context-only until a Faustini/F2-overlapping calibrated product is downloaded and co-registered.

## 2. Selected SAR Product And Reason

- Selected SAR product: `ch2_sar_ncls_20200808t201154198_d_cp_d18`.
- Configured AOI coverage fraction: 1.000.
- AOI: lat -87.8 to -86.9, lon 80.0 to 85.0 E.
- Partial-overlap products are supporting only; no-overlap products are excluded from the main candidate map.

## 3. Which Papers Guided Which Modules?

| Paper / filename | Main scientific idea | What factor/module it supports | How it will be used in LunaQuest | What not to claim | Implementation status | Reference file available |
| --- | --- | --- | --- | --- | --- | --- |
| remotesensing-14-04863.pdf | Fuzzy multi-factor landing site selection near PSRs using slope, rocks/roughness, illumination, maximum temperature, and PSR proximity. | Fuzzy landing candidate scoring and landing safety constraints. | Implements fuzzy membership scores for candidate proximity, slope safety, roughness hazard, and neutral placeholders for missing illumination/temperature layers. | Do not present preliminary landing candidates as certified landing products or use missing illumination/temperature layers as measured values. | implemented with slope/proximity/roughness active and illumination/temperature marked as missing neutral placeholders | True |
| 1-s2.0-S2095927325001999-main.pdf | Radar CPR can support polar ice-content scenarios, but high CPR/radar response is ambiguous because roughness and multiple scattering can also elevate radar returns. | Radar candidate screening, roughness ambiguity penalty, and scenario-based resource estimates. | Uses CPR-style ratio proxy cautiously, adds roughness ambiguity risk, keeps radar candidates for validation, and reports planning-only resource scenarios. | Do not treat CPR-style proxy or high candidate score as compositional proof or measured resource amount. | implemented as proxy radar screening plus roughness ambiguity and scenario tables | True |
| pnas.1802345115.sapp.pdf | Surface-exposed water ice evidence is strengthened with M3 spectral absorptions plus Diviner temperature, LOLA albedo, and LAMP H2O-proxy context. | External validation layer framework. | Creates validation layer status for Diviner maximum temperature, LOLA albedo, LAMP, M3, PSR/shadow, and illumination; missing layers are marked for future download. | Do not use surface-exposed ice validation as subsurface proof, and do not fabricate unavailable validation maps. | framework implemented; external layers marked future-required unless present | True |
| 1-s2.0-S0094576525004898-main.pdf | Rover planning should combine static obstacles with dynamic illumination/communication constraints and science waypoint priorities. | Conceptual rover route planning and route comparison. | Uses A* route variants for shortest, safest, science-priority, and energy-aware planning on available slope/roughness/science proxy costs; missing dynamic layers are explicit future inputs. | Do not call route variants operational traverses or certified operational paths. | implemented as conceptual A* route variants with missing dynamic constraints documented | True |

## 4. Extracted SAR Features

- SAR log intensity.
- LH channel and LV channel.
- CPR-style LH/LV ratio proxy and LV/LH ratio proxy.
- Polarization imbalance proxy.
- Local texture / roughness, local mean, and local standard deviation.
- Candidate score and threshold uncertainty.

Calibrated CPR/DOP are not derived from the selected SRI intensity rasters.

## 5. Ice Candidate Detection Before Landing Site Search

The workflow now explicitly generates the radar-based candidate ice map before landing analysis. Candidate screening starts from SAR proxy features, applies thresholded candidate-score and channel constraints, removes small connected components, and then evaluates connected candidate patches before any landing-site search.

- Candidate patches generated: 57.
- Candidate area percentage of valid AOI pixels: 0.375%.
- Threshold sensitivity is saved to `outputs/tables/threshold_sensitivity.csv` and `outputs/figures/threshold_sensitivity_curve.png`.
- Candidate stability across thresholds contributes to the `confidence_level` column.
- Landing candidates are selected near evaluated candidate patches but outside the candidate mask and risky/steep zones.
- This is a candidate screening layer for planning and validation.

## 6. How Candidate Patches Were Generated

- Candidate pixels are selected using SAR proxy thresholds for ratio, intensity, texture, and candidate score.
- Connected components define candidate patches.
- Each patch is evaluated for area, equivalent candidate patch diameter, score, ratio proxy, texture, slope context, uncertainty, and threshold stability.
- Confidence levels are High/Medium/Low based on score, extent, stability, uncertainty, and slope/traverse context.

## 7. Strongest Scientific Candidate Patches

| candidate_id | area_m2 | equivalent_candidate_patch_diameter_m | mean_candidate_score | confidence_level | roughness_ambiguity_risk | threshold_stability_class | mean_slope_deg | max_slope_deg | nearest_landing_site_id | distance_to_nearest_landing_candidate_m | validation_priority_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-025 | 13125.0 | 129.27207364566027 | 0.8460480570793152 | High | Low | High | 0.0 | 0.0 | L-05 | 105.49661201854936 | 1 |
| C-042 | 9375.0 | 109.25484305920791 | 0.7999980449676514 | High | Low | Medium | 0.0 | 0.0 | L-05 | 1806.301469854908 | 2 |
| C-011 | 11250.0 | 119.6826841204298 | 0.7799490094184875 | Medium | Low | Medium | 0.0 | 0.0 | L-02 | 186.34935000400063 | 3 |
| C-022 | 8750.0 | 105.5502061411188 | 0.834157407283783 | Medium | Low | Medium | 0.0 | 0.0 | L-05 | 549.3067291168584 | 4 |
| C-055 | 5000.0 | 79.78845608028654 | 0.8225793838500977 | Medium | Low | Medium | 0.0 | 0.0 | L-05 | 1561.8279804847268 | 5 |
| C-052 | 6250.0 | 89.20620580763855 | 0.7950607538223267 | Medium | Low | Medium | 0.0 | 0.0 | L-05 | 2549.9031844366164 | 6 |
| C-033 | 10625.0 | 116.31066229203195 | 0.6938923597335815 | Medium | Low | Low | 0.0 | 0.0 | L-05 | 461.31488019044144 | 7 |
| C-023 | 8750.0 | 105.5502061411188 | 0.7300246953964233 | Medium | Low | Medium | 0.0 | 0.0 | L-03 | 1971.172990679326 | 8 |

## 8. Candidates With Steep Terrain Or Roughness Tradeoff

Some high-score patches are less attractive as direct landing/traverse targets because their local slope context is steep. These can still be useful validation targets, but they should not drive landing-site selection without terrain review.

_No rows._

## 9. How Candidate Confidence Was Calculated

Candidate confidence combines mean candidate score, candidate patch extent, threshold stability, uncertainty near threshold, slope context, and a roughness ambiguity penalty. High roughness does not remove candidates; it marks them for OHRC, DEM, or multi-frequency validation.

## 10. Scenario-Based Resource Estimate

Resource scenarios use candidate patch area, assumed depth, and assumed ice fraction. These values are planning-only estimates for later-validated patches, not measured resource quantities.

| candidate_id | confidence_level | candidate_area_m2 | assumed_depth_m | assumed_ice_fraction | scenario_volume_m3 | scenario_mass_kg_using_917kg_m3 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-025 | High | 13125.0 | 1 | 0.01 | 131.25 | 120356.25 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 1 | 0.03 | 393.75 | 361068.75 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 1 | 0.06 | 787.5 | 722137.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 1 | 0.1 | 1312.5 | 1203562.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 3 | 0.01 | 393.75 | 361068.75 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 3 | 0.03 | 1181.25 | 1083206.25 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 3 | 0.06 | 2362.5 | 2166412.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 3 | 0.1 | 3937.5 | 3610687.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 5 | 0.01 | 656.25 | 601781.25 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 5 | 0.03 | 1968.75 | 1805343.75 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 5 | 0.06 | 3937.5 | 3610687.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |
| C-025 | High | 13125.0 | 5 | 0.1 | 6562.5 | 6017812.5 | scenario-based potential resource estimate if the candidate patch is later validated; planning-only estimate |

## 11. Best Landing Candidates And Why

| site_id | lat | lon | score_total | suitability_score | score_slope | score_candidate_proximity | score_roughness | score_illumination | score_temperature | slope_deg | distance_to_candidate_m | distance_to_target_candidate_m | nearest_candidate_id | nearest_candidate_confidence_level | route_accessible | route_accessible_yes_no | inside_candidate_mask_yes_no | inside_configured_aoi_yes_no | inside_f2_crater_estimate_yes_no | inside_steep_unsafe_slope_zone_yes_no | final_recommendation | reason | reason_selected | validation_needed | low_slope_score | candidate_proximity_score | roughness_avoidance_status | candidate_mask_clearance_status | illumination_layer_status | thermal_layer_status | communication_layer_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L-01 | -87.1467974308515 | 80.10291118780192 | 0.8407848477363586 | 0.8407848477363586 | 1.0 | 0.930232584476471 | 1.0 | 0.574999988079071 | 0.5 | 0.0 | 75.0 | 75.0 | C-001 | Low | yes | yes | no | yes | not_available_boundary_required | no | keep | near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 1.0 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | real layer missing; neutral placeholder used | real layer missing; neutral placeholder used | future line-of-sight layer needed |
| L-02 | -87.43876129006031 | 80.21703702179266 | 0.8407848477363586 | 0.8407848477363586 | 1.0 | 0.930232584476471 | 1.0 | 0.574999988079071 | 0.5 | 0.0 | 75.0 | 75.0 | C-013 | Low | needs route check | needs route check | no | yes | not_available_boundary_required | no | keep | near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 1.0 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | real layer missing; neutral placeholder used | real layer missing; neutral placeholder used | future line-of-sight layer needed |
| L-03 | -87.40858656968575 | 80.42451646120684 | 0.8407848477363586 | 0.8407848477363586 | 1.0 | 0.930232584476471 | 1.0 | 0.574999988079071 | 0.5 | 0.0 | 75.0 | 75.0 | C-010 | Low | needs route check | needs route check | no | yes | not_available_boundary_required | no | keep | near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 1.0 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | real layer missing; neutral placeholder used | real layer missing; neutral placeholder used | future line-of-sight layer needed |
| L-04 | -87.46588335849461 | 80.71610030112507 | 0.8407848477363586 | 0.8407848477363586 | 1.0 | 0.930232584476471 | 1.0 | 0.574999988079071 | 0.5 | 0.0 | 75.0 | 75.0 | C-026 | Low | needs route check | needs route check | no | yes | not_available_boundary_required | no | keep | near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 1.0 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | real layer missing; neutral placeholder used | real layer missing; neutral placeholder used | future line-of-sight layer needed |
| L-05 | -87.42872835037979 | 80.88866572563497 | 0.8407848477363586 | 0.8407848477363586 | 1.0 | 0.930232584476471 | 1.0 | 0.574999988079071 | 0.5 | 0.0 | 75.0 | 75.0 | C-025 | High | needs route check | needs route check | no | yes | not_available_boundary_required | no | keep | near candidate patch, outside candidate mask, and below landing slope cutoff in available DEM layer | high suitability score from low slope, roughness avoidance, candidate proximity, and candidate-mask clearance | illumination/PSR, thermal, communication, boulder/hazard, and manual terrain validation | 1.0 | 0.9302 | included via SAR texture proxy | outside candidate mask with configured clearance | real layer missing; neutral placeholder used | real layer missing; neutral placeholder used | future line-of-sight layer needed |

Best current preliminary landing candidate: `L-01` with suitability score 0.8407848477363586. It is selected because it is low slope, near a candidate patch, outside the candidate mask, and within the current proxy safety constraints.

## 12. Best Rover Route And Why

| route_type | target_candidate_id | start_landing_site_id | length_m | total_cost | mean_slope_deg | max_slope_deg | percent_under_5deg | percent_5_to_8deg | percent_8_to_10deg | percent_above_10deg | science_reward_score | energy_cost_proxy | traverse_risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shortest | C-055 | L-01 | 8949.011537017748 | 361.3563232421875 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.5830264264163939 | 10319.39330994918 | 0.06172890869060506 |
| safest | C-055 | L-01 | 8949.011537017748 | 407.7652587890625 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.5830264264163939 | 10319.39330994918 | 0.06172890869060506 |
| science_priority | C-055 | L-01 | 8949.011537017746 | 434.96160888671875 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.6749983016099479 | 10494.403037661492 | 0.06961222975872727 |
| energy_efficient | C-055 | L-01 | 8949.011537017748 | 391.9181823730469 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.5830264264163939 | 10319.39330994918 | 0.06172890869060506 |

Best current route by low risk and energy proxy: `shortest`. This remains a conceptual rover route for planning, not operational rover command generation.

## 13. What The U-Net Proves And Does Not Prove

- Pseudo-IoU: 0.2489406779661017.
- Pseudo-Dice: 0.3986429177268872.
- These metrics measure agreement with rule-based pseudo-labels only.
- They do not measure independently validated composition performance.

| experiment | input_channels | pseudo_iou | pseudo_dice | pseudo_precision | pseudo_recall | prediction_fraction | pseudo_label_fraction | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyUNet_5channel_weighted_loss | SAR log intensity, ratio proxy, texture, polarization imbalance proxy, candidate score | 0.2489406779661017 | 0.3986429177268872 | 0.3983050847457627 | 0.398981324278438 | 0.0012402044865973834 | 0.001238102445094676 | weakly supervised pseudo-label agreement only |
| candidate_score_quantile_baseline | candidate score threshold baseline | 0.10446877774489494 | 0.18917470525187566 | 0.1123130766783328 | 0.599320882852292 | 0.006606716443009451 | 0.001238102445094676 | baseline pseudo-label agreement only |
| candidate_score_strict_threshold | candidate score threshold baseline | 0.0123809724003111 | 0.02445911714629791 | 0.0123809724003111 | 1.0 | 0.10000042040830054 | 0.001238102445094676 | baseline pseudo-label agreement only |

## 14. Validation Still Needed

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
| TMC-2 DTM | usable | Selected ch2_tmc_ndn_20231026T1735025409_d_dtm_d18 with AOI coverage 1.000; used for terrain slope/roughness. |
| DEM/LDEM/LDSM | usable_fallback | LOLA/LDEM/LDSM terrain is retained as fallback/context and for TMC-2 sensitivity comparison when available. |
| OHRC | context only | Available OHRC footprints do not overlap the Faustini AOI; excluded from hazard scoring. |
| Ground truth labels | not available | U-Net uses rule-based pseudo-labels; metrics are agreement only. |

### External Validation Layer Status

| validation_layer | status | used_now | module_enabled | confidence_effect | next_action |
| --- | --- | --- | --- | --- | --- |
| SAR/DFSAR LH/LV SRI | available | yes | radar candidate screening | primary screening evidence, proxy-only | Selected product ch2_sar_ncls_20200808t201154198_d_cp_d18 |
| LOLA/LDEM/LDSM terrain | available | fallback/context | slope, roughness, landing, rover | improves terrain safety and roughness ambiguity review | Retain as fallback and sensitivity layer beside TMC-2 DTM |
| TMC-2 DTM terrain | available | yes | slope, roughness, landing, rover | local terrain/slope support for candidate ranking and route safety; not optical boulder confirmation | Selected ch2_tmc_ndn_20231026T1735025409_d_dtm_d18 |
| OHRC calibrated Faustini overlap | future_validation_layer_required | no | hazard proxy / boulder review | would reduce roughness ambiguity and landing hazard uncertainty | Download calibrated OHRC overlapping the configured AOI |
| Diviner maximum temperature | future_validation_layer_required | no | thermal/cold-trap suitability | would support <110 K cold-trap suitability screening if available | Download Diviner maximum temperature map for AOI |
| PSR / shadow / illumination | future_validation_layer_required | no | PSR validation and landing power constraints | would separate persistent shadow from rover power/landing constraints | Download PSR or illumination persistence layer |
| PSR stability proxy | available_proxy | yes | candidate stability context | adds approximate polar/shadow context but does not replace real illumination modeling | Replace proxy with validated PSR/illumination layer when available |
| LOLA albedo | future_validation_layer_required | no | surface-exposure validation context | would provide independent albedo context for surface ice screening | Download LOLA albedo layer for AOI |
| LAMP H2O proxy / band ratio | future_validation_layer_required | no | surface-exposure validation context | would add ultraviolet H2O-proxy evidence | Download LAMP band-ratio product if available |
| M3 spectral ice evidence | future_validation_layer_required | no | surface ice validation | would validate surface-exposed ice signatures only, not subsurface proof | Download suitable polar M3 spectral products if available |
| Complex/Stokes SAR products | future_validation_layer_required | no | true CPR/DOP derivation | would replace CPR-style proxy with physically calibrated polarimetry | Download complex/Stokes products and convention documentation |

### Slope Safety Summary

| valid_slope_pixels | mean_slope_deg | median_slope_deg | safe_lt_5deg_pct | acceptable_5_to_8deg_pct | marginal_8_to_10deg_pct | moderate_5_10deg_pct | unsafe_gt_10deg_pct | blocked_gt_15deg_pct | terrain_source | tmc_selected_product_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 157142 | 1.1772416830062866 | 0.0 | 91.30913441346044 | 3.1850173728220335 | 1.647554441206043 | 4.832571814028077 | 3.858293772511486 | 1.6329179977345332 | TMC-2 DTM: ch2_tmc_ndn_20231026T1735025409_d_dtm_d18 | ch2_tmc_ndn_20231026T1735025409_d_dtm_d18 |
