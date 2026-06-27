# LunaQuest / Lunar IceNav - Complete Implementation Plan

> **Project goal:** Calibrated Chandrayaan-2 DFSAR/SAR and OHRC observations of the Faustini F2 lunar south-polar region ko use karke explainable **radar candidate regions** screen karna, unko terrain context ke saath assess karna, aur future landing/rover planning concept dikhana.
>
> **Scientific rule:** Is project mein koi output `confirmed water ice` nahi hoga. Har result ko `radar candidate`, `screening score`, `candidate evidence`, ya `requires validation` bola jayega.

---

## Actual Workspace Implementation Update - 2026-06-25

The first runnable prototype has now been implemented under `src/lunar_icenav/` and executed against the files in this workspace.

Actual data findings:

- Inventory generated: `outputs/tables/product_inventory.csv`
- SAR/DFSAR files found: calibrated SRI/SLI/GRI LH/LV products plus metadata/geometry/browse files.
- DEM files found: `DEM Data/LDEM_80S_80MPP_ADJ.tiff` and `DEM Data/LDSM_80S_80MPP_ADJ.tiff`.
- OHRC files found: four zipped OHRC bundles with IMG/XML/geometry/browse PNGs.
- Configured Faustini F2 prototype AOI: lat -87.8 to -86.9 deg, lon 80 to 85 deg.
- Selected working SAR product: `ch2_sar_ncls_20200808t201154198_d_cp_d18`, because its calibrated SRI product fully covers the configured AOI at 25 m/pixel.
- Newer 4 m SAR products partially clip or miss this configured AOI; their overlap decisions are recorded in `outputs/tables/sar_aoi_overlap.csv`.
- The DEM covers the AOI and is used for terrain/slope context.
- Current OHRC products do not directly co-register to the configured Faustini AOI, so the OHRC output is saved as contextual browse/hazard evidence only.

Implemented outputs:

- Radar candidate screening: `outputs/figures/sar_candidate_overlay.png`
- SAR feature panel: `outputs/figures/sar_feature_panel.png`
- DEM terrain/slope map: `outputs/figures/dem_slope_map.png`
- Landing suitability overlay: `outputs/figures/landing_suitability_overlay.png`
- Rover route overlay: `outputs/figures/rover_route_overlay.png`
- Weakly supervised U-Net pseudo-label overlay: `outputs/figures/unet_prediction_overlay.png`
- Combined research decision map: `outputs/figures/combined_decision_map.png`
- Final run summary: `reports/LUNAQUEST_PROTOTYPE_SUMMARY.md`
- Runnable notebook: `notebooks/LunaQuest_BAH2026_Workflow.ipynb`

Important correction from implementation:

- True CPR/DOP are not claimed from the selected SRI intensity rasters. The code uses intensity, an LH/LV ratio proxy, polarization imbalance proxy, texture, and a candidate score. These are screening features only and require validation against proper compact-pol/Stokes products or published conventions.

---

## 1. What We Are Building

LunaQuest ek black-box ice detector nahi hai. Ye ek decision-support pipeline hai.

```text
Calibrated DFSAR/SAR                  Calibrated OHRC + optional DEM
         |                                           |
         +----- Product audit and Faustini coverage -+
                               |
                         Common AOI/grid
                               |
          SAR features: intensity, CPR proxy, DOP, texture
                               |
                    Rule-based candidate screen
                               |
                Candidate patches + uncertainty ranking
                               |
             OHRC terrain / hazard / shadow context check
                               |
         Landing suitability + constrained rover route concept
                               |
            Maps, figures, tables, research reports, reproducible run folder
```

### Main questions our solution answers

1. Selected Faustini tile mein kaunse calibrated SAR patterns present hain?
2. `CPR proxy` aur `DOP` ke basis par candidate pixels/patches kahan hain?
3. OHRC terrain image un candidates ke aas-paas kya context dikhati hai?
4. Kya candidate ke paas relatively safer landing zone aur rover path identify kiya ja sakta hai?
5. Kaunse results valid hain, kaunse uncertain hain, aur kis data ki kami hai?

---

## 2. Expected Deliverable for the Hackathon

The first deliverable is a credible **SAR candidate-screening prototype**. Agar OHRC/DEM available aur well-registered hai to landing and rover modules add honge. Agar nahi, unko clearly `planned extension` bolenge.

### Minimum working result

- One calibrated SAR product inspected safely.
- Faustini overlap verified, or window explicitly labelled as only a technical test.
- SAR intensity, CPR-proxy, DOP, candidate-mask quicklooks generated.
- Candidate patches ranked and summarized.
- Honest limitations shown in the validation report.

### Strong final result (when more data is available)

- Multiple overlapping SAR product inventory.
- Faustini AOI crop on a known lunar CRS/grid.
- OHRC terrain overlay aligned to the same AOI.
- DEM/slope/hazard/illumination suitability layer.
- Landing-zone ranking and A* rover route concept.
- Spatially separated ML comparison only when defensible labels exist.

---

## 3. Required Data

### 3.1 SAR / DFSAR - mandatory

For every SAR product, preserve the **original extracted folder**. Do not rename, edit, or move internal files.

Required:

- Calibrated TIFFs or equivalent science raster files.
- XML/PDS labels / metadata.
- Geometry CSV/XML if supplied.
- Product identifier and acquisition date.
- Sensor mode, polarization/channel convention, and processing level.

Useful:

- SRI/map-projected product for geocoded AOI crop.
- Incidence-angle layers.
- Browse PNG/JPEG for quick visual checks.

### 3.2 OHRC / OHR - strongly recommended

- Calibrated image.
- XML/PDS label and footprint/projection information.
- Image resolution and acquisition time.
- Browse image, if supplied.

### 3.3 Terrain / operations - optional but needed for real planning claims

- DEM and its vertical units.
- Slope/aspect or raster from which slope can be calculated.
- Hazard/boulder catalogue, if available.
- Illumination / permanent-shadow layer.

### 3.4 Labels - only needed for supervised ML

SAR/OHRC imagery itself is **not a label**. For a trained model, use one of:

- Published/reviewed reference regions with source and coordinates.
- Manual terrain/hazard annotations with reviewer notes.
- Rule-based pseudo-labels only for an explicitly labelled *baseline imitation experiment*.

Never call pseudo-label model probabilities â€œice probability.â€

---

## 4. Repository Structure

Create the following project layout. Raw data stays separate from all derived output.

```text
tantra-mantra/
â”œâ”€â”€ README.md
â”œâ”€â”€ LUNAQUEST_IMPLEMENTATION_PLAN.md
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ configs/
â”‚   â””â”€â”€ pipeline.json
â”œâ”€â”€ src/
â”‚   â””â”€â”€ lunar_icenav/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ cli.py
â”‚       â”œâ”€â”€ config.py
â”‚       â”œâ”€â”€ pipeline.py
â”‚       â”œâ”€â”€ io/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ products.py
â”‚       â”œâ”€â”€ preprocessing/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ aoi.py
â”‚       â”‚   â”œâ”€â”€ sar.py
â”‚       â”‚   â””â”€â”€ registration.py
â”‚       â”œâ”€â”€ features/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ polarimetry.py
â”‚       â”‚   â””â”€â”€ texture.py
â”‚       â”œâ”€â”€ mapping/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ anomaly.py
â”‚       â”œâ”€â”€ planning/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â”œâ”€â”€ landing.py
â”‚       â”‚   â””â”€â”€ rover.py
â”‚       â”œâ”€â”€ models/
â”‚       â”‚   â”œâ”€â”€ __init__.py
â”‚       â”‚   â””â”€â”€ training.py
â”‚       â””â”€â”€ viz/
â”‚           â”œâ”€â”€ __init__.py
â”‚           â””â”€â”€ plots.py
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_products.py
â”‚   â”œâ”€â”€ test_polarimetry.py
â”‚   â”œâ”€â”€ test_anomaly.py
â”‚   â”œâ”€â”€ test_landing.py
â”‚   â””â”€â”€ test_rover.py
â”œâ”€â”€ data/                         # Never commit multi-GB raw products
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â”œâ”€â”€ sar/
â”‚   â”‚   â”œâ”€â”€ ohrc/
â”‚   â”‚   â”œâ”€â”€ dem/
â”‚   â”‚   â””â”€â”€ illumination/
â”‚   â”œâ”€â”€ metadata/
â”‚   â”‚   â””â”€â”€ product_inventory.csv
â”‚   â”œâ”€â”€ interim/
â”‚   â”‚   â”œâ”€â”€ quicklooks/
â”‚   â”‚   â””â”€â”€ registration_checks/
â”‚   â”œâ”€â”€ processed/
â”‚   â”‚   â”œâ”€â”€ aoi_crops/
â”‚   â”‚   â”œâ”€â”€ features/
â”‚   â”‚   â””â”€â”€ aligned_layers/
â”‚   â”œâ”€â”€ labels/
â”‚   â”‚   â”œâ”€â”€ label_registry.csv
â”‚   â”‚   â””â”€â”€ class_definition.md
â”‚   â””â”€â”€ splits/
â”œâ”€â”€ models/
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ maps/
â”‚   â”œâ”€â”€ tables/
â”‚   â””â”€â”€ runs/
â””â”€â”€ presentation/
    â”œâ”€â”€ assets/
    â””â”€â”€ LunaQuest_BAH2026.pptx
```

---

## 5. Core Configuration

All changing parameters go in one config file. Values code mein hard-code nahi honge.

`configs/pipeline.json`:

```json
{
  "aoi": {
    "name": "Faustini F2",
    "lat_min": -87.8,
    "lat_max": -86.9,
    "lon_min": 80.0,
    "lon_max": 85.0
  },
  "lunar_radius_m": 1737400.0,
  "processing": {
    "window_height": 1024,
    "window_width": 1024,
    "multilook_rows": 5,
    "multilook_cols": 5,
    "nodata_policy": "mask"
  },
  "candidate_screen": {
    "cpr_proxy_min": 1.0,
    "dop_max": 0.13,
    "min_patch_pixels": 20
  },
  "planning": {
    "max_slope_deg": 12.0,
    "min_hazard_clearance_m": 25.0,
    "candidate_proximity_weight": 0.25
  }
}
```

> Thresholds final science truth nahi hain. They are transparent initial screening settings and must be documented in each run.

---

## 6. Implementation Modules

### 6.1 `io/products.py` - Product discovery and safe audit

**Purpose:** Product folder ko discover karna without loading huge rasters into memory.

Tasks:

1. Recursively find TIFF, IMG, XML, CSV, JSON, browse files.
2. Identify likely data/geometry/browse folders.
3. Rasterio/GDAL se metadata read:
   - width, height, count, dtype
   - CRS, transform, nodata
   - compression/tiled state
   - bounded sample percentile/min/max
4. XML/PDS labels ka raw copy/save.
5. Audit JSON export.

Output example:

```json
{
  "product_id": "ch2_sar_...",
  "raster_count": 4,
  "metadata_files": 3,
  "projection_known": false,
  "sample_stats": {"p02": 0.1, "p50": 2.4, "p98": 874.8},
  "decision": "needs_geometry_check"
}
```

### 6.2 `preprocessing/aoi.py` - Faustini AOI and projection

**Purpose:** AOI coordinates ko lunar south-polar projection/grid mein handle karna.

Tasks:

- Define Faustini F2 bounds.
- Check source product footprint intersects AOI.
- For map-projected SRI/OHRC/DEM: crop exact AOI.
- Transform AOI edge points; do not assume a lat/lon box remains rectangular after projection.
- Save AOI bounds and target transform in run metadata.

### 6.3 `preprocessing/sar.py` - bounded reading and valid masking

**Purpose:** Large calibrated SAR rasters ko window/tiles mein safely read karna.

Tasks:

- Read only a user-selected window or geocoded crop.
- Read IQ/LH/LV values according to exact product documentation.
- Apply nodata and invalid-value mask.
- Multilook using configured 5 x 5 grouping.
- Save unmodified input identity and processing parameters.

### 6.4 `features/polarimetry.py` - transparent SAR features

**Purpose:** CPR proxy and DOP compute karna.

For compact-polarimetric channels `LH` and `LV`, the implementation first computes Stokes-like quantities:

```text
S0 = |LH|Â² + |LV|Â²
S1 = |LH|Â² - |LV|Â²
S2 = 2 Ã— Re(LH Ã— conjugate(LV))
S3 = -2 Ã— Im(LH Ã— conjugate(LV))

DOP = sqrt(S1Â² + S2Â² + S3Â²) / max(S0, epsilon)
CPR_proxy = (S0 + S1) / max(S0 - S1, epsilon)
```

Important implementation checks:

- Exact sign/convention depends on the supplied DFSAR documentation.
- This is a **project compact-pol CPR proxy**, not automatically a published calibrated circular-polarization ratio.
- `epsilon` avoids division by zero.
- Invalid/nodata pixels always remain invalid.

Outputs:

- `s0.npy` / GeoTIFF
- `s1.npy`, `s2.npy`, `s3.npy` (if required)
- `cpr_proxy.tif`
- `dop.tif`
- valid-pixel mask

### 6.5 `mapping/anomaly.py` - candidate screening and patch summary

**Purpose:** Feature maps ko explainable candidate regions mein convert karna.

Base rule:

```python
candidate = valid & (cpr_proxy > cpr_proxy_min) & (dop < dop_max)
```

Then:

1. Remove very small components (`min_patch_pixels`).
2. Connected-component labeling.
3. Calculate every patch ka:
   - patch ID
   - pixel area / map area where valid georeferencing exists
   - mean/median CPR proxy
   - mean/median DOP
   - bounding box / centroid
   - valid-pixel fraction
4. Rank candidates but preserve uncertainty.

Candidate ranking is not an ice-volume estimate. It simply prioritizes review.

### 6.6 `preprocessing/registration.py` - SAR/OHRC common grid

**Purpose:** Fusion/overlay se pehle ensure karna that both layers represent same lunar location.

Requirements:

- Known source CRS/geometry for both layers.
- Same target lunar CRS.
- Same target pixel grid, transform, resolution and nodata policy.
- Resampling method saved: nearest for labels/masks; bilinear/cubic only for continuous imagery where appropriate.
- At least visual crater-rim/terrain feature alignment check.

If this fails:

```text
No pixel-level fusion claim.
OHRC remains a separate contextual visual layer.
```

### 6.7 `planning/landing.py` - landing suitability

**Purpose:** A co-registered DEM/hazard/illumination layer ke basis par a conceptual suitability score.

Possible normalized score:

```text
suitability =
    0.35 Ã— low_slope_score
  + 0.25 Ã— hazard_clearance_score
  + 0.20 Ã— illumination_score
  + 0.20 Ã— candidate_proximity_score
```

Hard constraints:

- slope > configured maximum -> blocked
- hazard clearance below minimum -> blocked
- nodata -> blocked

Output: suitability GeoTIFF/PNG and top feasible landing cells/areas.

### 6.8 `planning/rover.py` - constrained A* route

**Purpose:** Selected landing point se candidate patch tak safe conceptual path.

Cost definition:

```text
cost = base_distance
     + slope_penalty
     + hazard_penalty
     + shadow_or_energy_penalty
```

Blocked pixels:

- steep slope
- hazard mask
- no-data
- prohibited illumination state, if available

Outputs:

- route polyline / raster
- route length
- route cost
- blocked area percentage
- failure reason if no safe path exists

### 6.9 `models/training.py` - optional model, only after labels

Start with a Random Forest, not a deep model. Features: CPR proxy, DOP, intensity, texture, optional terrain values.

Correct split:

```text
Train: 3 spatially distinct regions/products
Validate: 1 separate spatial region/product
Test: 1 never-tuned-on region/product
```

Never do random pixel split, because neighboring lunar pixels create spatial leakage.

Metrics: Precision, Recall, F1, PR-AUC, confusion matrix, spatial prediction map.

---

## 7. Pipeline Run Order

### Gate G0 - Product audit

Run only after data is received.

```powershell
python -m lunar_icenav.cli inspect data/raw/sar/<product_folder> `
  --output reports/runs/2026-06-24_audit_sar01
```

Pass conditions:

- Calibration/product metadata exists.
- Raster opens safely.
- Channel conventions can be identified.
- Footprint/geolocation is available or can be investigated.

### Gate G1 - Same area proof

```text
SAR and OHRC both overlap Faustini F2 on known lunar coordinates.
```

If false: do only SAR screening; do not claim fusion.

### Gate G2 - Data quality

Generate quicklooks and sample statistics.

```powershell
python -m lunar_icenav.cli quicklook data/raw/sar/<product_folder> `
  --output data/interim/quicklooks/sar01
```

Pass conditions:

- Valid pixel coverage is sufficient.
- No corrupt or all-zero raster.
- Sensible percentile stretch.
- Correct channel/band selected.

### Gate G3 - SAR prototype

```powershell
python -m lunar_icenav.cli prototype data/raw/sar/<product_folder> `
  --config configs/pipeline.json `
  --row 0 --col 0 --height 1024 --width 1024 `
  --output reports/runs/2026-06-26_sar_baseline_v01
```

Expected artifacts:

```text
reports/runs/2026-06-26_sar_baseline_v01/
â”œâ”€â”€ config_copy.json
â”œâ”€â”€ product_audit.json
â”œâ”€â”€ prototype_features.npz
â”œâ”€â”€ prototype_quicklook.png
â”œâ”€â”€ candidate_mask.png
â”œâ”€â”€ candidate_patches.csv
â”œâ”€â”€ prototype_summary.json
â””â”€â”€ run_notes.md
```

> If this is a row/column technical window, its figure must say `technical prototype window - geolocation pending`.

### Gate G4 - Registered OHRC contextual check

```powershell
python -m lunar_icenav.cli register `
  --sar data/processed/aoi_crops/sar_faustini.tif `
  --ohrc data/raw/ohrc/<product>.tif `
  --output reports/runs/2026-06-27_registration_v01
```

Pass only if CRS/grid/visual alignment checks pass.

### Gate G5 - Mission planning

```powershell
python -m lunar_icenav.cli plan `
  --candidate-mask data/processed/features/candidate_mask.tif `
  --slope data/processed/aligned_layers/slope.tif `
  --hazards data/processed/aligned_layers/hazards.tif `
  --illumination data/processed/aligned_layers/illumination.tif `
  --output reports/runs/2026-06-29_planning_v01
```

Run only if each input has valid co-registration and known units.

---

## 8. Mandatory Data Tables

### `data/metadata/product_inventory.csv`

```csv
product_id,sensor,processing_level,calibrated,date_utc,mode_polarization,resolution,crs,footprint,overlaps_faustini,keep,reason
TBD,SAR,TBD,yes,TBD,TBD,TBD,TBD,TBD,TBD,TBD,TBD
```

### `data/labels/label_registry.csv` - create only if labels exist

```csv
label_id,class,geometry_tile,source,reviewer,date,confidence,uncertainty_reason,permitted_use
TBD,TBD,TBD,TBD,TBD,TBD,TBD,TBD,train_validate_or_visual_only
```

### `reports/tables/candidate_summary.csv`

```csv
candidate_id,product_id,centroid_or_tile,area_pixels,mean_cpr_proxy,mean_dop,valid_fraction,rank,interpretation
C-001,TBD,TBD,TBD,TBD,TBD,TBD,TBD,radar_candidate_requires_validation
```

---

## 9. Visual Outputs Required for Research Evaluation

Every map must include title, product ID, date where available, scale/coordinates when known, legend, and uncertainty statement.

1. **Study area / Faustini map** - selected AOI and product footprints.
2. **Product inventory slide** - SAR/OHRC product IDs, modes, dates, coverage decision.
3. **SAR quicklook** - intensity/valid-pixel inspection.
4. **Feature panel** - S0 intensity, CPR proxy, DOP, candidate mask.
5. **Candidate ranking table** - patch-level evidence and uncertainty.
6. **SAR + OHRC overlay** - only after registration passes.
7. **Landing suitability map** - only after terrain layers are valid.
8. **Rover route map** - path, landing site, candidate, blocked zones.
9. **Limitations panel** - false positives/ambiguous terrain/missing layers.

---

## 10. Research Narrative for Later Communication

| Slide | Message | Evidence to show |
|---|---|---|
| 1 | LunaQuest: explainable lunar candidate screening | Team, challenge, one-line solution |
| 2 | Why the lunar south pole is hard | PSR/DSC context, Faustini AOI |
| 3 | Data and audit discipline | Inventory table, SAR/OHRC/DEM roles |
| 4 | Architecture | End-to-end workflow diagram |
| 5 | Safe SAR preprocessing | Metadata, bounded window, feature equations |
| 6 | Explainable M0 candidate screen | CPR proxy, DOP, candidate map + caveat |
| 7 | Candidate interpretation | Patch table and terrain ambiguity explanation |
| 8 | OHRC / terrain / fusion context | Registered overlay or clearly labelled planned module |
| 9 | Landing + rover concept | Suitability/routing logic and constraints |
| 10 | Impact, limits, next steps | No confirmation claim; validation plan |

Recommended slide language:

- `potential radar candidate region`
- `screening feature`
- `candidate evidence requiring validation`
- `co-registered terrain context`
- `conceptual route under stated constraints`

Avoid:

- direct compositional confirmation claims
- `exact water volume`
- `mission-ready landing site`
- Accuracy metrics from synthetic data, random pixels, or undocumented pseudo-labels

---

## 11. 1 July Execution Plan

### Day 1: Data intake and product audit

- Receive data and preserve originals under `data/raw/`.
- Run inspect command per product.
- Fill product inventory.
- Make browse/quicklook images.

**Deliverable:** `product_inventory.csv`, audit JSONs, quicklooks.

### Day 2: Faustini AOI and SAR read validation

- Confirm product footprint/AOI overlap.
- Identify correct SAR inputs/channel convention.
- Read one bounded window / crop.
- Validate data statistics and valid mask.

**Deliverable:** SAR validation note and first valid intensity plot.

### Day 3: Explainable SAR baseline

- Multilook data.
- Generate S0, CPR proxy, DOP.
- Produce candidate mask and patch summary.

**Deliverable:** Main results figure for the research output package.

### Day 4: Candidate review and OHRC preparation

- Rank candidate patches.
- Audit OHRC data.
- Check projection and overlap.
- Start registration check.

**Deliverable:** Candidate table and OHRC quicklook.

### Day 5: Context and mission-planning module

- Create SAR/OHRC overlay if registration passes.
- Build terrain/hazard/slope suitability only when valid inputs exist.
- Generate route concept where valid.

**Deliverable:** Context/mission figure or transparent â€œfuture stageâ€ slide.

### Day 6: Research figures and reproducibility

- Save final maps/tables in `reports/maps` and `reports/tables`.
- Build limitation examples.
- Complete research output package.

**Deliverable:** First full research output package.

### Day 7: Final review and submission

- Check every scientific claim has saved evidence.
- Check all labels and caveats.
- Practice 2-minute and 5-minute explanation.

**Deliverable:** Submission-ready research outputs + reproducible run folder.

---

## 12. Team Work Split

| Person / role | Work |
|---|---|
| Data lead | Download, preserve, inventory SAR/OHRC/DEM products; validate files |
| SAR lead | Product audit, channel identification, feature extraction, candidate mask |
| GIS/OHRC lead | AOI, projection, registration, terrain/hazard context |
| Mission-planning lead | Landing suitability and A* rover path |
| Research communication lead | Figures, story, citations, limitations, rehearsal |

For a small team, priority should be: **SAR baseline first -> figures -> research outputs -> OHRC context -> planning extension -> ML last.**

---

## 13. Reproducibility Rules

Every `reports/runs/<run_id>/` folder must contain:

- Exact copied `pipeline.json`.
- Input product ID/path list.
- Date/time and code version/commit if available.
- Package versions.
- Processing settings: window, multilook, thresholds, resampling.
- Figures and numeric summary.
- Known errors/limitations.

Raw data is never modified. Derived files always go in `data/interim/`, `data/processed/`, or `reports/`.

---

## 14. Final Scientific Limitations Slide

Use these points directly in the validation report:

- CPR proxy and DOP are radar-screening features, not direct proof of water ice.
- Rough terrain, fresh crater ejecta, acquisition geometry and noise can create ambiguous radar signatures.
- OHRC imagery has severe illumination/shadow limits near the south pole.
- Pixel-level SAR/OHRC fusion is not valid until co-registration is demonstrated.
- A small number of products limits generalisation.
- Pseudo-label performance, if used, measures agreement with a rule - not lunar composition truth.
- Landing/rover outputs remain conceptual unless slope, hazard and illumination layers are validated on a common grid.

---

## 15. First Action When Data Arrives

Place source folders unchanged here:

```text
data/raw/sar/<original_calibrated_sar_product>/
data/raw/ohrc/<original_calibrated_ohrc_product>/
```

Then run the product audit. Do **not** train a model first. The correct first output is a product inventory and a valid SAR quicklook.

Once the first SAR product is inspected, update this plan with its exact file names, channel convention, product ID, coverage decision, and the selected first experiment.

