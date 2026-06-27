# Slide-Wise Text For PPT

Copy-paste-ready text for a 12-slide LunaQuest / Lunar IceNav deck. This is not a PPT file.

## Slide 1

**Title:** LunaQuest / Lunar IceNav - Candidate Lunar Ice Detection and Mission Planning for Faustini/F2

**Bullets:**
- BAH 2026 Problem Statement 8
- Chandrayaan-2 SAR + DEM-based screening workflow
- Radar-based candidate patches, preliminary landing candidates, and conceptual rover routes
- Screening output; validation required

**Figure:** `outputs/figures/16_combined_decision_map.png`

**Caption:** Preliminary decision-support map for the Faustini/F2 prototype AOI; validation required.

**Speaker Notes:**  
Yeh deck LunaQuest ka full prototype story dikhata hai. Hum radar screening se planning tak ja rahe hain: candidate patches, landing candidates, rover route variants, aur future ML module. Current outputs decision-support level par hain, validation required.

## Slide 2

**Title:** Problem Statement - From Candidate Ice Screening To Mission Planning

**Bullets:**
- Detect and characterize candidate subsurface ice indicators
- Connect candidate patches to safe landing and traverse planning
- Build reproducible geospatial workflow
- Keep claims limited to screening and planning outputs

**Figure:** `outputs/figures/workflow_diagram.png`

**Caption:** Prototype workflow linking radar screening, terrain context, preliminary landing candidates, conceptual rover routes, and weakly supervised ML.

**Speaker Notes:**  
Problem sirf anomaly map banana nahi hai. Real value tab aati hai jab candidate area ko landing and rover access ke saath connect karte hain. LunaQuest isi end-to-end decision-support workflow ko demonstrate karta hai.

## Slide 3

**Title:** Why Faustini/F2 Prototype AOI

**Bullets:**
- Lunar South Pole prototype region
- Latitude: -87.8 to -86.9
- Longitude: 80.0 to 85.0 E
- Used for candidate screening and planning demonstration

**Figure:** `outputs/figures/16_combined_decision_map.png`

**Caption:** Faustini/F2 prototype AOI used for candidate-screening and planning demonstration; validation required.

**Speaker Notes:**  
Faustini/F2 ko focused AOI ke roop mein select kiya gaya. Coordinates clearly define hain, aur yeh lunar South Pole context ke liye relevant region hai. Is slide par safe message: candidate screening region, not a validated water-ice map.

## Slide 4

**Title:** Dataset and Product Audit

**Bullets:**
- Inventory: 158 SAR/DFSAR rows, 4 OHRC bundles, 2 DEM/topography entries
- Selected SAR: `ch2_sar_ncls_20200808t201154198_d_cp_d18`
- Selected SAR coverage fraction: 1.000 for configured AOI
- Partial/no-overlap SAR products kept out of main map

**Figure:** `outputs/figures/01_dataset_inventory_chart.png` and `outputs/figures/02_selected_sar_coverage.png`

**Caption:** Source inventory and SAR coverage audit for the Faustini/F2 prototype AOI.

**Speaker Notes:**  
Pipeline pehle source audit karti hai. Inventory se pata chalta hai kaunse products available hain, phir AOI overlap check hota hai. Main map ke liye 2020 SAR product use hua because configured Faustini/F2 AOI fully cover hota hai.

## Slide 5

**Title:** System Architecture

**Bullets:**
- SAR/DFSAR to proxy feature extraction
- Feature score to radar-based candidate patches
- DEM slope to landing suitability screening
- Route planner and weakly supervised pseudo-label prototype

**Figure:** `outputs/figures/workflow_diagram.png`

**Caption:** LunaQuest prototype architecture from SAR screening to preliminary mission-planning outputs.

**Speaker Notes:**  
Architecture modular hai: SAR data se features nikalte hain, candidate score banate hain, DEM slope add karte hain, landing suitability compute karte hain, rover route variants plan karte hain, aur U-Net ko future ML module ke roop mein include karte hain.

## Slide 6

**Title:** SAR Feature Extraction

**Bullets:**
- Selected SAR LH/LV SRI intensity pair
- Features: intensity, LH/LV ratio proxy, imbalance proxy, texture proxy
- Candidate score creates transparent screening layer
- True CPR/DOP are not claimed yet

**Figure:** `outputs/figures/04_sar_feature_panel.png`

**Caption:** Proxy SAR feature panel for radar-based candidate screening; validation required.

**Speaker Notes:**  
Yahan feature panel explain karna hai. Har panel ek proxy view deta hai: intensity, ratio proxy, imbalance, texture, candidate score, aur candidate mask. True CPR/DOP abhi claim nahi kar rahe, kyunki current rasters intensity-based hain.

## Slide 7

**Title:** Radar-Based Candidate Results

**Bullets:**
- 65 radar-based candidate patches generated
- Candidate pixels: about 0.255% of valid AOI pixels
- C-040: high score and larger top-patch area, but steep local slope context
- C-059, C-060, C-062: useful low-slope discussion candidates

**Figure:** `outputs/figures/05_radar_candidate_overlay.png`; optional `outputs/figures/06b_top_candidate_patches.png`

**Caption:** Radar-based candidate patch overlay and top-patch ranking from proxy SAR screening; validation required.

**Speaker Notes:**  
This is the main radar result. 65 candidate patches generated hain. C-040 strong score example hai, but slope context steep hai. C-059, C-060, C-062 low-slope context ke karan discussion ke liye stronger examples hain. Sab candidate patches validation require karte hain.

## Slide 8

**Title:** DEM/Slope-Based Landing Safety

**Bullets:**
- Low slope under 5 deg: about 37.99%
- Moderate slope 5-10 deg: about 16.36%
- Higher-risk slope above 10 deg: about 45.65%
- Slope helps screen landing and traverse feasibility

**Figure:** `outputs/figures/08b_slope_classification_map.png`

**Caption:** DEM-derived slope classes for preliminary landing and traverse planning.

**Speaker Notes:**  
Candidate patch useful ho sakta hai, but nearby safe access bhi chahiye. Slope layer batata hai ki low-slope zones kahan hain aur terrain risk kahan high hai. Yeh preliminary planning constraint hai.

## Slide 9

**Title:** Preliminary Landing Candidate Selection

**Bullets:**
- Top 5 preliminary landing candidates: L-01 to L-05
- Main examples: L-01, L-02, L-03
- L-01: score 0.955, slope 0.61 deg, 75.0 m to nearest candidate patch
- L-02 and L-03 are additional high-score, low-slope access examples

**Figure:** `outputs/figures/10_top_landing_candidates_overlay.png`

**Caption:** Top preliminary landing candidates near radar-based candidate patches; validation required.

**Speaker Notes:**  
Landing module top 5 candidate zones shortlist karta hai. L-01, L-02, L-03 ko main examples ke roop mein use kar sakte hain because scores high hain aur slopes low hain. Inko preliminary landing candidate hi bolna hai.

## Slide 10

**Title:** Rover Traverse Planning

**Bullets:**
- Three conceptual rover route variants
- Shortest, safest, and science-priority
- Route length: about 422.5 m in current prototype
- Science-priority route gives balanced story: science access + terrain cost

**Figure:** `outputs/figures/12_rover_route_overlay.png` and `outputs/figures/11_rover_route_comparison.png`

**Caption:** Conceptual rover route variants to a radar-based candidate patch; validation required.

**Speaker Notes:**  
Rover planning slide route tradeoff dikhata hai. Shortest simple distance optimize karta hai, safest terrain cost ko emphasize karta hai, aur science-priority balanced story deta hai. Yeh conceptual rover route variant hai, not an operational route plan.

## Slide 11

**Title:** Weakly Supervised U-Net Prototype

**Bullets:**
- Trained using rule-based pseudo-labels
- Inputs: SAR intensity and proxy feature channels
- Pseudo-IoU: 0.265; pseudo-Dice: 0.419
- Best framed as future ML module

**Figure:** `outputs/figures/14_unet_training_curve.png`; optional `outputs/figures/15_unet_prediction_overlay.png`

**Caption:** Weakly supervised U-Net pseudo-label agreement curves; not an independently validated composition model.

**Speaker Notes:**  
U-Net ka role future ML module hai. It learns from pseudo-labels generated by the screening rules. Pseudo-IoU and pseudo-Dice agreement metrics hain, independent detection accuracy nahi. Isko headline result nahi banana.

## Slide 12

**Title:** Limitations and Future Work

**Bullets:**
- No direct water-ice confirmation claim in current outputs
- True CPR/DOP not yet claimed
- Faustini-overlapping calibrated OHRC product still needed
- Add illumination/PSR, thermal, communication, and stronger hazard layers
- Improve ML with stronger labels and multi-pass SAR context

**Figure/Table:** `reports/SCIENTIFIC_LIMITATION_CHECKLIST.md`; optional `outputs/figures/16_combined_decision_map.png`

**Caption:** Current LunaQuest outputs are preliminary decision-support products; validation required before stronger scientific or operational claims.

**Speaker Notes:**  
Closing slide honest hona chahiye. Prototype ka strength reproducible workflow hai, but stronger science ke liye validation layers chahiye: OHRC co-registration, illumination/PSR, thermal, communications, CPR/DOP validation, aur better ML labels. Storyboard approve hone ke baad deck banana next step hoga.
