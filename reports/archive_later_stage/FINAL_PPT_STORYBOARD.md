# Final PPT Storyboard

Project: LunaQuest / Lunar IceNav  
Problem statement: BAH 2026 Problem Statement 8 - Detection and characterization of candidate subsurface ice indicators in lunar South Polar regions using Chandrayaan-2 radar and imagery data for landing and rover traverse planning.

This storyboard is a slide plan only. It does not create the PPT.

## Slide 1: LunaQuest / Lunar IceNav - Candidate Lunar Ice Detection and Mission Planning for Faustini/F2

**Main Message:** LunaQuest converts Chandrayaan-2 radar and terrain data into a preliminary decision-support workflow for candidate patch review, landing selection, and rover traverse planning.

**Bullets:**
- BAH 2026 Problem Statement 8.
- Study area: Faustini/F2 prototype AOI near the lunar South Pole.
- Workflow connects radar screening, DEM terrain context, preliminary landing candidates, and conceptual rover routes.
- All current outputs are screening results and require validation.

**Figure/Table To Use:** `outputs/figures/16_combined_decision_map.png`

**Exact Safe Caption:** Preliminary decision-support map for the Faustini/F2 prototype AOI; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Yeh deck LunaQuest ka end-to-end prototype dikhata hai. Hum Chandrayaan-2 SAR data, DEM slope, landing scoring, rover route planning, aur weak ML prototype ko ek workflow mein combine kar rahe hain. Important point: yeh decision-support output hai, validation required.

**What Not To Claim:**
- Do not claim water-ice presence has been independently verified.
- Do not present any route or landing output as operational approval.
- Do not overstate the U-Net result.

## Slide 2: Problem Statement - From Candidate Ice Screening To Mission Planning

**Main Message:** The challenge is not only to detect candidate radar anomalies, but to translate them into usable landing and traverse planning products.

**Bullets:**
- C8 asks for candidate subsurface ice detection and characterization in lunar South Polar regions.
- Exploration value depends on linking candidate patches to safe access.
- LunaQuest builds a reproducible pipeline from data audit to planning outputs.
- Outputs are designed for review, not direct mission certification.

**Figure/Table To Use:** `outputs/figures/workflow_diagram.png`

**Exact Safe Caption:** Prototype workflow linking radar screening, terrain context, preliminary landing candidates, conceptual rover routes, and weakly supervised ML.

**Speaker Notes - Simple Hinglish/English:**  
Problem sirf anomaly map banana nahi hai. Agar koi radar-based candidate patch milta hai, hume dekhna hota hai ki lander kahan safely aa sakta hai aur rover kaise reach karega. Isliye deck ka story detection se planning tak jaata hai.

**What Not To Claim:**
- Do not claim the workflow replaces scientific validation.
- Do not imply current outputs are complete mission products.
- Do not claim composition from proxy SAR features.

## Slide 3: Why Faustini/F2 Prototype AOI

**Main Message:** Faustini/F2 was selected as a focused lunar South Pole prototype AOI where radar screening and planning layers can be tested together.

**Bullets:**
- AOI latitude range: -87.8 to -86.9.
- AOI longitude range: 80.0 to 85.0 E.
- The region is relevant for lunar South Pole volatile-preservation studies.
- The current pipeline treats this as a candidate-screening region, not a verified water-ice map.

**Figure/Table To Use:** `outputs/figures/16_combined_decision_map.png` or AOI coordinate callout using `configs/pipeline.json`

**Exact Safe Caption:** Faustini/F2 prototype AOI used for candidate-screening and planning demonstration; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Faustini/F2 ko humne focused AOI ke roop mein liya hai. Coordinates clearly define hain: lat -87.8 se -86.9, lon 80 se 85 E. Yeh South Pole candidate region hai, but slide par hamesha safe wording rakhenge: screening and planning demonstration.

**What Not To Claim:**
- Do not claim the AOI map verifies water-ice presence.
- Do not imply every candidate patch is scientifically validated.
- Do not present coordinates as a selected operational target.

## Slide 4: Dataset and Product Audit

**Main Message:** The workflow starts with source-data discipline: inventory, overlap checking, and selecting the SAR product that covers the configured AOI.

**Bullets:**
- Inventory includes 158 SAR/DFSAR rows, 4 OHRC zip bundles, 2 DEM/topography entries, and supporting documents.
- The selected SAR product is `ch2_sar_ncls_20200808t201154198_d_cp_d18`.
- Selected SAR AOI coverage fraction: 1.000 for the configured Faustini/F2 AOI.
- Partial/no-overlap SAR products are not used for the main map.

**Figure/Table To Use:** `outputs/figures/01_dataset_inventory_chart.png` and `outputs/figures/02_selected_sar_coverage.png`

**Exact Safe Caption:** Source inventory and SAR coverage audit for the Faustini/F2 prototype AOI.

**Speaker Notes - Simple Hinglish/English:**  
Is slide mein hum dikha rahe hain ki pipeline random product use nahi karti. Pehle inventory banaya, phir AOI overlap check kiya. 2020 SAR product selected hai because configured AOI ko fully cover karta hai.

**What Not To Claim:**
- Do not imply all downloaded products are co-registered.
- Do not imply all SAR products have equal AOI coverage.
- Do not use OHRC as a core Faustini hazard layer yet.

## Slide 5: System Architecture

**Main Message:** LunaQuest is an end-to-end geospatial ML and planning pipeline, not just a single image-processing script.

**Bullets:**
- SAR/DFSAR intensity products are converted into proxy screening features.
- Radar-based candidate patches are filtered and summarized.
- DEM slope context supports landing and traverse safety screening.
- Rover route variants and weakly supervised pseudo-label ML are generated as planning modules.

**Figure/Table To Use:** `outputs/figures/workflow_diagram.png`

**Exact Safe Caption:** LunaQuest prototype architecture from SAR screening to preliminary mission-planning outputs.

**Speaker Notes - Simple Hinglish/English:**  
Architecture ka idea simple hai: data audit se start, SAR features nikalna, candidate patch screen karna, DEM slope add karna, landing suitability score banana, rover routes plan karna, aur ML prototype ko future module ke tarah include karna.

**What Not To Claim:**
- Do not claim this architecture is a complete mission operations system.
- Do not say current proxy layers replace illumination, thermal, or communications analysis.
- Do not imply U-Net output is independently labeled science truth.

## Slide 6: SAR Feature Extraction

**Main Message:** SAR intensity and derived proxy features are used to create a transparent radar-based candidate screening score.

**Bullets:**
- Inputs: selected Chandrayaan-2 SAR LH/LV SRI intensity pair.
- Features: SRI intensity, LH/LV ratio proxy, polarization imbalance proxy, texture roughness proxy.
- Candidate score combines proxy radar behavior and texture constraints.
- True CPR/DOP are not claimed from the current SRI intensity rasters.

**Figure/Table To Use:** `outputs/figures/04_sar_feature_panel.png`

**Exact Safe Caption:** Proxy SAR feature panel for radar-based candidate screening; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Yahan hum SAR feature panel explain karenge. Hum intensity, ratio proxy, imbalance proxy, texture, score, aur mask dikha rahe hain. Sabse important line: true CPR/DOP current rasters se claim nahi kar rahe. Yeh transparent screening layer hai.

**What Not To Claim:**
- Do not call the LH/LV ratio proxy true CPR/DOP.
- Do not present candidate score as composition measurement.
- Do not imply texture alone indicates volatile content.

## Slide 7: Radar-Based Candidate Results

**Main Message:** The pipeline found 65 radar-based candidate patches; the best discussion examples balance candidate score with slope context.

**Bullets:**
- Candidate patch count: 65; candidate pixels cover about 0.255% of valid AOI pixels.
- C-040 has the highest score and largest top-patch area, but steep local slope context.
- C-059, C-060, and C-062 are stronger discussion candidates for low-slope context.
- All radar-based candidate patches require higher-confidence validation.

**Figure/Table To Use:** `outputs/figures/05_radar_candidate_overlay.png` plus optional `outputs/figures/06b_top_candidate_patches.png`; table source `outputs/tables/candidate_patches.csv`

**Exact Safe Caption:** Radar-based candidate patch overlay and top-patch ranking from proxy SAR screening; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Yeh main result slide hai, but wording careful rahegi. 65 radar-based candidate patches mile. C-040 score high hai, area bhi larger hai, but slope steep hai. Isliye C-059, C-060, C-062 low-slope context ke liye better discussion examples hain.

**What Not To Claim:**
- Do not state that radar-based patches verify water-ice presence.
- Do not say candidate ranking estimates subsurface resource quantity.
- Do not ignore slope tradeoffs when discussing C-040.

## Slide 8: DEM/Slope-Based Landing Safety

**Main Message:** DEM-derived slope screening helps separate scientifically interesting candidate areas from safer access zones.

**Bullets:**
- Slope classes: low slope under 5 deg, moderate 5-10 deg, higher-risk above 10 deg.
- Low-slope area under 5 deg is about 37.99% of valid slope pixels.
- Moderate slope area is about 16.36%; higher-risk slope area is about 45.65%.
- Slope is a core constraint for landing and rover traverse screening.

**Figure/Table To Use:** `outputs/figures/08b_slope_classification_map.png`; table source `outputs/tables/slope_safety_summary.csv`

**Exact Safe Caption:** DEM-derived slope classes for preliminary landing and traverse planning.

**Speaker Notes - Simple Hinglish/English:**  
Is slide ka message hai: candidate patch interesting ho sakta hai, but landing safe zone nearby chahiye. DEM slope layer se hum low, moderate, aur high-risk slope areas separate karte hain. About 37.99% area low-slope class mein hai.

**What Not To Claim:**
- Do not call DEM slope alone a complete hazard assessment.
- Do not claim boulder, shadow, thermal, or communications safety is solved here.
- Do not present slope classes as certification.

## Slide 9: Preliminary Landing Candidate Selection

**Main Message:** The landing module identifies five preliminary landing candidates near radar-based candidate patches while avoiding steep/rough proxy zones.

**Bullets:**
- Top five preliminary landing candidates: L-01 to L-05.
- Main discussion examples: L-01, L-02, and L-03.
- L-01 score: 0.955, slope 0.61 deg, distance to nearest candidate patch 75.0 m.
- L-02 and L-03 provide additional low-slope access examples with high suitability scores.

**Figure/Table To Use:** `outputs/figures/10_top_landing_candidates_overlay.png`; table source `outputs/tables/landing_sites.csv`

**Exact Safe Caption:** Top preliminary landing candidates near radar-based candidate patches; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Yahan hum landing candidates ko planning output ke roop mein dikhayenge. L-01, L-02, L-03 good examples hain because score high hai, slope low hai, aur candidate patches ke paas hain. But inko preliminary landing candidate hi bolna hai.

**What Not To Claim:**
- Do not call any landing candidate certified or selected for operations.
- Do not imply illumination, thermal, or communication constraints are already complete.
- Do not present suitability score as absolute lander safety.

## Slide 10: Rover Traverse Planning

**Main Message:** Three conceptual rover route variants show how science access and terrain cost can be compared.

**Bullets:**
- Route variants: shortest, safest, and science-priority.
- Each route is about 422.5 m to target patch C-040 in the current prototype.
- Science-priority route cost: 26.46; mean slope: 3.72 deg; max slope: 14.69 deg.
- Recommend science-priority route for a balanced deck story, with shortest/safest as tradeoff context.

**Figure/Table To Use:** `outputs/figures/12_rover_route_overlay.png` and `outputs/figures/11_rover_route_comparison.png`; table source `outputs/routes/route_summary.csv`

**Exact Safe Caption:** Conceptual rover route variants to a radar-based candidate patch; validation required.

**Speaker Notes - Simple Hinglish/English:**  
Route slide mein hum teen options compare karenge: shortest, safest, science-priority. Balanced story ke liye science-priority route achha hai because it connects science access with terrain cost. Yeh conceptual rover route hai, operational route plan nahi.

**What Not To Claim:**
- Do not call route variants approved traverse plans.
- Do not imply route costs include full rover mobility, thermal, or communications constraints.
- Do not claim C-040 is the preferred science target without discussing steep local context.

## Slide 11: Weakly Supervised U-Net Prototype

**Main Message:** The U-Net module is a future ML pathway trained on pseudo-labels from rule-based screening, not independent science labels.

**Bullets:**
- Input channels: SAR intensity, LH/LV ratio proxy, texture roughness proxy, polarization imbalance proxy, and candidate score.
- Training uses rule-based pseudo-labels generated by the screening workflow.
- Pseudo-IoU: 0.265; pseudo-Dice: 0.419.
- Use this slide as future ML module / prototype extension, not headline evidence.

**Figure/Table To Use:** `outputs/figures/14_unet_training_curve.png`; optional `outputs/figures/15_unet_prediction_overlay.png`; table source `outputs/tables/unet_pseudo_label_metrics.csv`

**Exact Safe Caption:** Weakly supervised U-Net pseudo-label agreement curves; not an independently validated composition model.

**Speaker Notes - Simple Hinglish/English:**  
U-Net slide ko carefully frame karna hai. Model pseudo-labels pe train hua hai, jo rule-based candidate mask se aaye. Metrics pseudo-label agreement dikhate hain, independent science validation nahi. Isko future ML module ke roop mein show karna best hai.

**What Not To Claim:**
- Do not present pseudo-IoU or pseudo-Dice as independent detection accuracy.
- Do not make U-Net the main evidence result.
- Do not imply pseudo-labels are ground-truth labels.

## Slide 12: Limitations and Future Work

**Main Message:** The prototype is strong as a reproducible screening and planning workflow, but scientific validation layers are still required before stronger claims.

**Bullets:**
- No direct water-ice confirmation claim in current outputs.
- True CPR/DOP are not yet claimed from the selected SRI intensity rasters.
- OHRC Faustini-overlapping calibrated product is still needed for co-registered optical hazard analysis.
- Illumination/PSR, thermal, communication, and stronger hazard layers remain future work.
- Future work: validate CPR/DOP with complex/Stokes products, improve U-Net labels, and build a mission-planning dashboard/deck.

**Figure/Table To Use:** `reports/SCIENTIFIC_LIMITATION_CHECKLIST.md` and optional `outputs/figures/16_combined_decision_map.png`

**Exact Safe Caption:** Current LunaQuest outputs are preliminary decision-support products; validation required before stronger scientific or operational claims.

**Speaker Notes - Simple Hinglish/English:**  
Last slide honest closing hai. Prototype useful hai because workflow complete and reproducible hai. But stronger science claim ke liye OHRC co-registration, illumination/PSR, thermal, communications, aur better SAR/CPR/DOP validation chahiye. Next stage final deck tab banayenge jab storyboard approve ho jaye.

**What Not To Claim:**
- Do not present screening outputs as direct composition proof.
- Do not claim current OHRC products solve Faustini/F2 hazard mapping.
- Do not imply preliminary landing or route outputs are complete mission products.
