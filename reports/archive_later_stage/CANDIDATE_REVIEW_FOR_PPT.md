# Candidate Review For PPT

Purpose: review the strongest current evidence candidates before selecting final PPT visuals. All entries are screening outputs and require validation.

## Radar-Based Candidate Patches

### C-040

- Candidate type: radar-based candidate patch / screening result.
- Area: 11250.0 m2 (18 pixels).
- Mean candidate score: 0.856.
- Centroid latitude/longitude: -87.425872, 80.846214.
- Slope condition: steep local context; lower landing/traverse priority (mean 17.78 deg, max 21.40 deg).
- Uncertainty / limitation: proxy SAR features only; true CPR/DOP not claimed; contains steep pixels; requires validation.
- Good for PPT discussion: Yes - useful as a score-versus-terrain tradeoff example, not as a preferred planning target.
- Figure to use: `outputs/figures/05_radar_candidate_overlay.png plus outputs/figures/06b_top_candidate_patches.png`.

### C-059

- Candidate type: radar-based candidate patch / screening result.
- Area: 5000.0 m2 (8 pixels).
- Mean candidate score: 0.834.
- Centroid latitude/longitude: -87.403686, 81.858075.
- Slope condition: low mean slope; favorable screening context (mean 2.94 deg, max 3.81 deg).
- Uncertainty / limitation: proxy SAR features only; true CPR/DOP not claimed; small connected component; requires validation.
- Good for PPT discussion: Yes - useful if framed as a compact screening result that requires higher-resolution validation.
- Figure to use: `outputs/figures/05_radar_candidate_overlay.png plus outputs/figures/06b_top_candidate_patches.png`.

### C-038

- Candidate type: radar-based candidate patch / screening result.
- Area: 5625.0 m2 (9 pixels).
- Mean candidate score: 0.831.
- Centroid latitude/longitude: -87.443374, 80.632192.
- Slope condition: steep local context; lower landing/traverse priority (mean 23.51 deg, max 24.73 deg).
- Uncertainty / limitation: proxy SAR features only; true CPR/DOP not claimed; small connected component; contains steep pixels; requires validation.
- Good for PPT discussion: Yes - useful as a score-versus-terrain tradeoff example, not as a preferred planning target.
- Figure to use: `outputs/figures/05_radar_candidate_overlay.png plus outputs/figures/06b_top_candidate_patches.png`.

### C-062

- Candidate type: radar-based candidate patch / screening result.
- Area: 5000.0 m2 (8 pixels).
- Mean candidate score: 0.823.
- Centroid latitude/longitude: -87.406919, 81.923743.
- Slope condition: low mean slope; favorable screening context (mean 2.45 deg, max 3.07 deg).
- Uncertainty / limitation: proxy SAR features only; true CPR/DOP not claimed; small connected component; requires validation.
- Good for PPT discussion: Yes - useful if framed as a compact screening result that requires higher-resolution validation.
- Figure to use: `outputs/figures/05_radar_candidate_overlay.png plus outputs/figures/06b_top_candidate_patches.png`.

### C-060

- Candidate type: radar-based candidate patch / screening result.
- Area: 5000.0 m2 (8 pixels).
- Mean candidate score: 0.818.
- Centroid latitude/longitude: -87.362424, 82.024760.
- Slope condition: low mean slope; favorable screening context (mean 3.01 deg, max 3.49 deg).
- Uncertainty / limitation: proxy SAR features only; true CPR/DOP not claimed; small connected component; requires validation.
- Good for PPT discussion: Yes - useful if framed as a compact screening result that requires higher-resolution validation.
- Figure to use: `outputs/figures/05_radar_candidate_overlay.png plus outputs/figures/06b_top_candidate_patches.png`.

## Preliminary Landing Candidates

### L-01

- Candidate type: preliminary landing candidate, not a certified landing product.
- Suitability score: 0.955.
- Slope value: 0.61 deg.
- Distance to nearest radar-based candidate patch: 75.0 m.
- Why selected: high suitability score from low slope, terrain-safety proxy, and proximity to candidate patches while staying outside the candidate mask.
- Validation still needed: illumination, thermal, communication, higher confidence hazard/boulder layer, and manual terrain review.
- Show in PPT: Yes - show top 1-3 as planning concept.
- Figure to use: `outputs/figures/10_top_landing_candidates_overlay.png` and `outputs/figures/09_landing_suitability_map.png`.

### L-02

- Candidate type: preliminary landing candidate, not a certified landing product.
- Suitability score: 0.943.
- Slope value: 0.49 deg.
- Distance to nearest radar-based candidate patch: 100.0 m.
- Why selected: high suitability score from low slope, terrain-safety proxy, and proximity to candidate patches while staying outside the candidate mask.
- Validation still needed: illumination, thermal, communication, higher confidence hazard/boulder layer, and manual terrain review.
- Show in PPT: Yes - show top 1-3 as planning concept.
- Figure to use: `outputs/figures/10_top_landing_candidates_overlay.png` and `outputs/figures/09_landing_suitability_map.png`.

### L-03

- Candidate type: preliminary landing candidate, not a certified landing product.
- Suitability score: 0.941.
- Slope value: 0.35 deg.
- Distance to nearest radar-based candidate patch: 127.5 m.
- Why selected: high suitability score from low slope, terrain-safety proxy, and proximity to candidate patches while staying outside the candidate mask.
- Validation still needed: illumination, thermal, communication, higher confidence hazard/boulder layer, and manual terrain review.
- Show in PPT: Yes - show top 1-3 as planning concept.
- Figure to use: `outputs/figures/10_top_landing_candidates_overlay.png` and `outputs/figures/09_landing_suitability_map.png`.

### L-04

- Candidate type: preliminary landing candidate, not a certified landing product.
- Suitability score: 0.940.
- Slope value: 0.75 deg.
- Distance to nearest radar-based candidate patch: 75.0 m.
- Why selected: high suitability score from low slope, terrain-safety proxy, and proximity to candidate patches while staying outside the candidate mask.
- Validation still needed: illumination, thermal, communication, higher confidence hazard/boulder layer, and manual terrain review.
- Show in PPT: Optional supporting candidate.
- Figure to use: `outputs/figures/10_top_landing_candidates_overlay.png` and `outputs/figures/09_landing_suitability_map.png`.

### L-05

- Candidate type: preliminary landing candidate, not a certified landing product.
- Suitability score: 0.938.
- Slope value: 1.02 deg.
- Distance to nearest radar-based candidate patch: 75.0 m.
- Why selected: high suitability score from low slope, terrain-safety proxy, and proximity to candidate patches while staying outside the candidate mask.
- Validation still needed: illumination, thermal, communication, higher confidence hazard/boulder layer, and manual terrain review.
- Show in PPT: Optional supporting candidate.
- Figure to use: `outputs/figures/10_top_landing_candidates_overlay.png` and `outputs/figures/09_landing_suitability_map.png`.

## Conceptual Rover Route Variants

### shortest

- Route type: conceptual rover route variant.
- Length: 422.5 m.
- Total cost: 18.23.
- Mean slope: 3.68 deg.
- Max slope: 14.69 deg.
- Target candidate patch ID: C-040.
- PPT recommendation: Use for comparison.

### safest

- Route type: conceptual rover route variant.
- Length: 422.5 m.
- Total cost: 32.47.
- Mean slope: 3.68 deg.
- Max slope: 14.69 deg.
- Target candidate patch ID: C-040.
- PPT recommendation: Good PPT route if emphasizing terrain-risk minimization.

### science_priority

- Route type: conceptual rover route variant.
- Length: 422.5 m.
- Total cost: 26.46.
- Mean slope: 3.72 deg.
- Max slope: 14.69 deg.
- Target candidate patch ID: C-040.
- PPT recommendation: Best PPT route for balanced story.

Recommended route figure: `outputs/figures/12_rover_route_overlay.png` with `outputs/figures/11_rover_route_comparison.png` as backup.

## Weakly Supervised U-Net Review

- Input channels used: SAR log intensity, LH/LV ratio proxy, texture roughness proxy, polarization imbalance proxy, and rule candidate score.
- Pseudo-labels: rule-based candidate mask used as weak supervision, not independently validated lunar composition labels.
- Pseudo-IoU: 0.26524685382381413.
- Pseudo-Dice: 0.41928079571537874.
- Metric meaning: agreement with the pseudo-label mask only; not real composition accuracy.
- Visual usefulness: useful as a future ML module / weakly supervised baseline. Do not make it the main result unless clearly framed as pseudo-label agreement.
- Figures to use carefully: `outputs/figures/14_unet_training_curve.png`, `outputs/figures/15_unet_prediction_overlay.png`.

## Overall PPT Recommendation

- Lead with SAR feature/candidate workflow and combined decision map.
- Use landing and rover visuals as mission-planning concept outputs.
- Keep U-Net as a research prototype/future module, not the headline evidence.