# LunaQuest / Lunar IceNav BAH 2026 Prototype

This repository contains a practical decision-support prototype for ISRO Bharatiya Antariksh Hackathon 2026 Problem Statement 8.

The pipeline inspects available Chandrayaan-2 SAR/DFSAR, OHRC, and DEM files, selects a usable Faustini AOI SAR pair, builds transparent radar screening proxies, generates preliminary candidate masks, adds DEM terrain context, scores preliminary landing candidates, plans conceptual rover routes, and runs a weakly supervised U-Net proof of concept against pseudo-labels.

Scientific wording rule: no output is presented as compositional proof. Outputs are radar-based candidate signatures, screening products, pseudo-label predictions, and preliminary planning products that require independent validation.

## Run

```powershell
$env:PYTHONPATH='src'
python -m lunar_icenav.cli run --config configs/pipeline.json
```

## Main Outputs

- `outputs/figures/sar_candidate_overlay.png`
- `outputs/figures/sar_feature_panel.png`
- `outputs/figures/dem_slope_map.png`
- `outputs/figures/landing_suitability_overlay.png`
- `outputs/figures/rover_route_overlay.png`
- `outputs/figures/unet_prediction_overlay.png`
- `outputs/figures/combined_decision_map.png`
- `outputs/tables/product_inventory.csv`
- `outputs/tables/candidate_patches.csv`
- `outputs/tables/landing_sites.csv`
- `outputs/routes/route_summary.csv`
- `reports/LUNAQUEST_PROTOTYPE_SUMMARY.md`
- `notebooks/LunaQuest_BAH2026_Workflow.ipynb`

## Current Data Decision

The configured Faustini F2 prototype AOI is lat -87.8 to -86.9 deg and lon 80 to 85 deg. The 2020 calibrated SRI SAR product provides full AOI coverage at 25 m/pixel and is selected automatically for the working prototype. DEM files cover the AOI. The available OHRC zip bundles are treated as contextual browse products because their current footprints are not directly co-registered to the Faustini AOI.
