# OHRC Data Download Needed

Current OHRC products are not co-registered to the configured Faustini/F2 AOI. They are useful as context-only browse examples, but should not be used as Faustini hazard/boulder maps.

Download calibrated OHRC products from PRADAN for:

- Lat Min: -87.8
- Lat Max: -86.9
- Lon Min: 80.0
- Lon Max: 85.0

Selection notes:

- Choose calibrated OHRC products.
- Preserve the full product bundle with data, geometry, browse, and XML metadata.
- After download, rerun `python -m lunar_icenav.cli run --config configs/pipeline.json`.
- Only claim OHRC hazard/boulder context after footprint and grid alignment are demonstrated.