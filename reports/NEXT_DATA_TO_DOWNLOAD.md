# Next Data To Download

Priority data needed to move from screening prototype toward stronger scientific validation:

1. Calibrated OHRC product overlapping the configured Faustini/F2 AOI.
   - Why needed: resolves small-scale hazards, blocky terrain, and boulder-like features if resolution supports it.
   - Improves: roughness ambiguity, landing safety, rover corridor review.
   - Enables: optical_hazard_proxy_map with actual AOI co-registration.
   - Confidence effect: reduces radar roughness ambiguity for top candidate patches.
2. Additional TMC-2 DTM co-registration QA or backup DTM products.
   - Why needed: a TMC-2 DTM is now processed when it overlaps the AOI, but co-registration and terrain-model sensitivity still require review.
   - Improves: slope safety, local relief, route risk, and terrain-model robustness.
   - Enables: stronger TMC-2 vs LOLA/DEM slope comparison and backup terrain selection for an AOI switch.
   - Confidence effect: checks whether candidate ranking is terrain-model sensitive.
3. Diviner maximum temperature map for the Faustini/F2 AOI.
   - Why needed: cold-trap suitability should use real thermal data.
   - Improves: thermal validation and candidate confidence.
   - Enables: temperature_validation_map and candidate thermal status.
   - Confidence effect: supports or weakens cold-trap plausibility; use the ~110 K rule only with real data.
4. PSR / illumination / shadow persistence layer.
   - Why needed: separates candidate access from persistent shadow and rover power constraints.
   - Improves: validation, fuzzy landing score, rover route planning.
   - Enables: psr_shadow_validation_map and active illumination scoring.
   - Confidence effect: prevents treating SAR candidates outside stable shadow as equally strong.
5. LAMP / LOLA albedo / M3 validation layers if available.
   - Why needed: adds PNAS-style independent surface-evidence context.
   - Improves: external validation layer framework.
   - Enables: albedo_validation_map and candidate_validation_against_external_layers updates.
   - Confidence effect: supports surface-exposure context, not subsurface proof.
6. Complex/Stokes SAR/DFSAR products for true CPR/DOP computation.
   - Why needed: current selected layers are SRI intensities only.
   - Improves: polarimetric feature extraction.
   - Enables: calibrated CPR/DOP module after convention validation.
   - Confidence effect: replaces CPR-style ratio proxy with defensible polarimetric features.
7. F2 crater boundary / crater catalog shapefile.
   - Why needed: rectangular AOI is not a crater boundary.
   - Improves: landing boundary validation and crater-interior/rim reasoning.
   - Enables: landing_vs_f2_crater_boundary_map with real boundary geometry.
   - Confidence effect: prevents accepting a landing point only because it lies inside the rectangular AOI.

## AOI For Search

- Latitude min: -87.8
- Latitude max: -86.9
- Longitude min: 80.0
- Longitude max: 85.0 E