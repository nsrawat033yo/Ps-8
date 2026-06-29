# Research Paper Method Map

This report maps each uploaded method reference to an implementable LunaQuest module. Duplicate PDF copies are recorded but ignored for method integration.

## Method Traceability

| Paper / filename | Main scientific idea | What factor/module it supports | How it will be used in LunaQuest | What not to claim | Implementation status | Reference file available |
| --- | --- | --- | --- | --- | --- | --- |
| remotesensing-14-04863.pdf | Fuzzy multi-factor landing site selection near PSRs using slope, rocks/roughness, illumination, maximum temperature, and PSR proximity. | Fuzzy landing candidate scoring and landing safety constraints. | Implements fuzzy membership scores for candidate proximity, slope safety, roughness hazard, and neutral placeholders for missing illumination/temperature layers. | Do not present preliminary landing candidates as certified landing products or use missing illumination/temperature layers as measured values. | implemented with slope/proximity/roughness active and illumination/temperature marked as missing neutral placeholders | True |
| 1-s2.0-S2095927325001999-main.pdf | Radar CPR can support polar ice-content scenarios, but high CPR/radar response is ambiguous because roughness and multiple scattering can also elevate radar returns. | Radar candidate screening, roughness ambiguity penalty, and scenario-based resource estimates. | Uses CPR-style ratio proxy cautiously, adds roughness ambiguity risk, keeps radar candidates for validation, and reports planning-only resource scenarios. | Do not treat CPR-style proxy or high candidate score as compositional proof or measured resource amount. | implemented as proxy radar screening plus roughness ambiguity and scenario tables | True |
| pnas.1802345115.sapp.pdf | Surface-exposed water ice evidence is strengthened with M3 spectral absorptions plus Diviner temperature, LOLA albedo, and LAMP H2O-proxy context. | External validation layer framework. | Creates validation layer status for Diviner maximum temperature, LOLA albedo, LAMP, M3, PSR/shadow, and illumination; missing layers are marked for future download. | Do not use surface-exposed ice validation as subsurface proof, and do not fabricate unavailable validation maps. | framework implemented; external layers marked future-required unless present | True |
| 1-s2.0-S0094576525004898-main.pdf | Rover planning should combine static obstacles with dynamic illumination/communication constraints and science waypoint priorities. | Conceptual rover route planning and route comparison. | Uses A* route variants for shortest, safest, science-priority, and energy-aware planning on available slope/roughness/science proxy costs; missing dynamic layers are explicit future inputs. | Do not call route variants operational traverses or certified operational paths. | implemented as conceptual A* route variants with missing dynamic constraints documented | True |

## Duplicate Handling

| filename | paper_title | duplicate_group | duplicate_status | canonical_reference_filename | sha256_12 | exists | used_in_pipeline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| remotesensing-14-04863.pdf | Selection of Lunar South Pole Landing Site Based on Constructing and Analyzing Fuzzy Cognitive Maps | remote_sensing_fcm_2022 | unique_used | remotesensing-14-04863.pdf | 2f108fc4c7e9 | True | True |
| 1-s2.0-S2095927325001999-main.pdf | Upper limit of ice content at the lunar south pole as revealed by the Earth-based SYISR-FAST bistatic radar system | science_bulletin_sysisr_fast_2025 | unique_used | 1-s2.0-S2095927325001999-main.pdf | dc97f260845e | True | True |
| 1-s2.0-S2095927325001999-main (1).pdf | Upper limit of ice content at the lunar south pole as revealed by the Earth-based SYISR-FAST bistatic radar system | science_bulletin_sysisr_fast_2025 | duplicate_ignored | 1-s2.0-S2095927325001999-main.pdf | 3347fdf96b32 | True | False |
| pnas.1802345115.sapp.pdf | Supporting information for direct evidence of surface-exposed water ice in the lunar polar regions | pnas_surface_exposed_ice_si | unique_used | pnas.1802345115.sapp.pdf | 3d87057df74f | True | True |
| 1-s2.0-S0094576525004898-main.pdf | Path planning algorithm for a South Pole lunar rover mission | acta_rover_path_planning_2025 | unique_used | 1-s2.0-S0094576525004898-main.pdf | 3e84f133178e | True | True |

## Language Guardrails

- Radar output is a candidate screening result and requires validation.
- CPR-style ratio proxy is not calibrated CPR/DOP.
- Equivalent candidate patch diameter is a patch extent metric.
- Resource scenarios are planning-only estimates.
- U-Net metrics are pseudo-label agreement metrics.