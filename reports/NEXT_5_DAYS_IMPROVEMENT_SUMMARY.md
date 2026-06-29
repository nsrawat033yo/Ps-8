# Next 5 Days Improvement Summary

## Improved Compared To First Prototype

- Added numbered, 300 DPI research figures for inventory, coverage, SAR features, candidate patches, DEM terrain, landing, rover routes, OHRC context, U-Net curves, and combined decision support.
- Added candidate patch summary, data coverage status, slope safety summary, U-Net training history, U-Net tile inventory, and checkpoint output.
- Improved U-Net section with spatial tile split, augmentation, training/validation curves, pseudo-IoU, and pseudo-Dice.
- Improved route table with route length, cost, mean slope, max slope, and target candidate ID.
- Added OHRC download instruction report for the correct Faustini/F2 AOI.

## Stronger Outputs Now Available

- `outputs/figures/04_sar_feature_panel.png`
- `outputs/figures/05_radar_candidate_overlay.png`
- `outputs/figures/06b_top_candidate_patches.png`
- `outputs/figures/08b_slope_classification_map.png`
- `outputs/figures/12_rover_route_overlay.png`
- `outputs/figures/14_unet_training_curve.png`
- `outputs/figures/16_combined_decision_map.png`

## Current Evaluation-Style Summary

- Radar-based candidate patches: 57
- Preliminary landing candidates: 5
- Route variants: 4
- U-Net pseudo-IoU: 0.2489406779661017
- U-Net pseudo-Dice: 0.3986429177268872
- Safe slope area <5 deg: 91.30913441346044%

## Still Needs Validation

- CPR/DOP must be derived only after validating product convention or obtaining complex/Stokes layers.
- OHRC Faustini-overlapping product is still needed for real optical hazard analysis.
- Illumination, thermal, and communication layers are placeholders/future layers.
- U-Net metrics are pseudo-label agreement only; independent validation labels are not available.

## Recommended Next Technical Work

1. Download calibrated OHRC for the exact Faustini AOI and rerun coverage checks.
2. Add at least one illumination or PSR/shadow layer if available.
3. Manually review top candidate patches and mark 3-5 examples for field validation planning.
4. Convert notebook outputs into a dashboard or communication artifact only after technical review.