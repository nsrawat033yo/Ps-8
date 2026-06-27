# LunaQuest Prototype Summary

## What was built

This run creates a research-notebook-ready prototype for radar-based candidate screening, DEM terrain context, preliminary landing suitability, conceptual rover routing, and weakly supervised pseudo-label segmentation.

No output is presented as compositional proof. Candidate masks are screening outputs requiring independent validation.

## Actual data availability

- Inventory rows: 245
- Selected SAR product: `ch2_sar_ncls_20200808t201154198_d_cp_d18`
- Selected SAR AOI coverage fraction: 1.000
- Selected SAR pixel size: 25.0 m
- DEM files found and used for terrain/slope context.
- OHRC files are zip bundles with browse PNG and IMG data; current OHRC footprints are not directly co-registered to the configured Faustini AOI.

## Candidate screening result

- Candidate patches found: 65
- Candidate mask area: 0.255% of valid AOI pixels
- Top candidate: C-053 with area 9375.0 m2 and mean score 0.7999980449676514

The SAR features are intensity, LH/LV ratio proxy, polarization imbalance proxy, texture, and a combined candidate score. True CPR/DOP are not claimed because the available SRI rasters are real-valued intensity products rather than the complex/Stokes products needed for a defensible CPR/DOP derivation.

## Landing and route prototype

- Landing candidates found: 5
- Top landing candidate: L-01 with suitability score 0.9552358388900757
- Route variants: shortest, safest, science_priority, energy_efficient
- Blocked slope threshold for routing: >15 deg

## U-Net prototype

weakly supervised U-Net trained against rule-based pseudo-labels and SAR screening features; metrics measure pseudo-label agreement, not ground-truth ice accuracy

Pseudo-label metrics: `{
  "pseudo_iou": 0.26524685382381413,
  "pseudo_dice": 0.41928079571537874,
  "prediction_fraction": 0.0013747351427706588,
  "pseudo_label_fraction": 0.0013726331012679513,
  "training_loss_last": 1.1300250356611998,
  "validation_loss_last": 1.2297405004501343,
  "validation_pseudo_iou_last": 0.266199649737303,
  "validation_pseudo_dice_last": 0.42047026279391425,
  "training_tiles": 46,
  "validation_tiles": 18,
  "augmented_training_tiles": 184,
  "prediction_threshold": 0.9844234585762024
}`

## Best current research outputs

- `outputs/figures/04_sar_feature_panel.png`
- `outputs/figures/05_radar_candidate_overlay.png`
- `outputs/figures/08_dem_slope_map.png`
- `outputs/figures/12_rover_route_overlay.png`
- `outputs/figures/14_unet_training_curve.png`
- `outputs/figures/16_combined_decision_map.png`

## Limitations and next steps

- Replace CPR-style proxy with published CPR/DOP only after confirming compact-pol channel convention and availability of complex/Stokes products.
- Download a calibrated OHRC product overlapping the configured Faustini AOI before claiming pixel-level optical hazard/boulder analysis.
- Add illumination, thermal, and communication layers for stronger landing-site scoring.
- Treat U-Net metrics as pseudo-label agreement only, not lunar composition accuracy.

## How to run

```powershell
$env:PYTHONPATH='src'
python -m lunar_icenav.cli run --config configs/pipeline.json
python -m lunar_icenav.cli notebook --config configs/pipeline.json
```

## DEM slope safety summary

| valid_slope_pixels | mean_slope_deg | median_slope_deg | safe_lt_5deg_pct | moderate_5_10deg_pct | unsafe_gt_10deg_pct | blocked_gt_15deg_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 256181.0 | 9.301920890808105 | 8.23563289642334 | 37.9852526143625 | 16.36499193929292 | 45.64975544634458 | 28.700801386519686 |

## SAR AOI overlap table

| product_id | coverage_fraction | pixel_size_m | lh |
| --- | --- | --- | --- |
| ch2_sar_ncls_20200808t201154198_d_cp_d18 | 1.0 | 25.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncls_20200808t201154198_d_cp_d18\data\calibrated\20200808\ch2_sar_ncxl_20200808t201154198_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncls_20200808t201154198_d_cp_d18 | 1.0 | 25.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncls_20200808t201154198_d_cp_d18\data\calibrated\20200808\ch2_sar_ncxs_20200808t201154198_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20250919t055442925_d_cp_d18 | 0.18808156078442076 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20250919t055442925_d_cp_d18\data\calibrated\20250919\ch2_sar_ncxl_20250919t055442925_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20251003t123305739_d_cp_d18 | 0.0 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20251003t123305739_d_cp_d18\data\calibrated\20251003\ch2_sar_ncxl_20251003t123305739_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20251016t150420052_d_cp_d18 | 0.09741724430379513 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20251016t150420052_d_cp_d18\data\calibrated\20251016\ch2_sar_ncxl_20251016t150420052_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20251106t161627207_d_cp_d18 | 0.0 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20251106t161627207_d_cp_d18\data\calibrated\20251106\ch2_sar_ncxl_20251106t161627207_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20251106t181422757_d_cp_d18 | 0.0 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20251106t181422757_d_cp_d18\data\calibrated\20251106\ch2_sar_ncxl_20251106t181422757_d_sri_xx_cp_lh_d18.tif |
| ch2_sar_ncxl_20251106t201219233_d_cp_d18 | 0.0 | 4.0 | C:\Users\nsraw\Downloads\tantra-mantra\ch2_sar_ncxl_20251106t201219233_d_cp_d18\data\calibrated\20251106\ch2_sar_ncxl_20251106t201219233_d_sri_xx_cp_lh_d18.tif |

## OHRC footprint note

| product_id | lat_min | lat_max | coverage_note |
| --- | --- | --- | --- |
| ch2_ohr_ncp_20211228T2209123959_d_img_d18 | -89.923132 | -89.252796 | not co-registered to Faustini AOI |
| ch2_ohr_ncp_20241115T1326321339_d_img_d18 | -89.946885 | -89.19986 | not co-registered to Faustini AOI |
| ch2_ohr_ncp_20241115T1525004388_d_img_d18 | -89.950601 | -89.193425 | not co-registered to Faustini AOI |
| ch2_ohr_ncp_20251010T0942085687_d_img_d18 | -89.928812 | -89.219895 | not co-registered to Faustini AOI |