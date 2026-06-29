"""
Generate SAR Dataset Footprint Map
===================================
Visualizes the geographic footprints of all discovered SAR datasets
to show which ones cover the target AOI ("work ok") and which don't.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lunar_icenav.preprocessing.aoi import aoi_lon_lat_corners, map_to_lonlat

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "illumination_landing"
INVENTORY_CSV = OUTPUT_DIR / "sar_inventory.csv"
DEM_PATH = ROOT / "DEM Data" / "LDEM_80S_80MPP_ADJ.tiff"

ICE_AOI = {
    "name": "Faustini F2 ice candidate AOI",
    "lat_min": -87.8,
    "lat_max": -86.9,
    "lon_min": 80.0,
    "lon_max": 85.0,
}

DARK_BG = "#0a0a0f"
PANEL_BG = "#111118"
GOLD = "#ffd700"
GREEN = "#00e676"
RED = "#ff1744"
WHITE = "#f0f0f0"

def get_sar_bounds(lh_path: str):
    try:
        with rasterio.open(lh_path) as ds:
            # We need the lon/lat bounds of the dataset
            # ds.bounds gives bounds in the dataset's CRS
            crs_wkt = ds.crs.to_wkt()
            xs = np.array([ds.bounds.left, ds.bounds.right, ds.bounds.right, ds.bounds.left, ds.bounds.left])
            ys = np.array([ds.bounds.bottom, ds.bounds.bottom, ds.bounds.top, ds.bounds.top, ds.bounds.bottom])
            lons, lats = map_to_lonlat(crs_wkt, xs, ys)
            return lons, lats
    except Exception as e:
        print(f"Error reading bounds for {lh_path}: {e}")
        return None, None

def main():
    print("Generating SAR Dataset Footprint Map...")
    
    if not INVENTORY_CSV.exists():
        print("SAR inventory not found. Please run the main pipeline first.")
        return

    inventory = pd.read_csv(INVENTORY_CSV)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Plot the DEM as background context if possible
    # We will just plot it in lon/lat space approximately
    try:
        with rasterio.open(DEM_PATH) as ds:
            # Subsample for speed
            factor = 10
            elev = ds.read(1, out_shape=(ds.height // factor, ds.width // factor))
            # Just get the bounds in lon/lat
            # LDEM is polar stereographic
            xs = np.linspace(ds.bounds.left, ds.bounds.right, elev.shape[1])
            ys = np.linspace(ds.bounds.top, ds.bounds.bottom, elev.shape[0])
            # We won't reproject the whole grid, just use a simple background
            # Actually, plotting footprints in Polar Stereographic is much better than lon/lat!
            dem_crs = ds.crs.to_wkt()
            extent = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
            # Normalize elevation for display
            elev = np.where(elev == ds.nodata, np.nan, elev)
            lo, hi = np.nanpercentile(elev, [2, 98])
            elev_norm = np.clip((elev - lo) / (hi - lo), 0, 1)
            ax.imshow(elev_norm, cmap="gray", extent=extent, alpha=0.5)
            
            # Now we need to plot everything in DEM CRS
            from pyproj import CRS, Transformer
            dem_crs_obj = CRS.from_wkt(dem_crs)
            geo_crs = dem_crs_obj.geodetic_crs
            transformer = Transformer.from_crs(geo_crs, dem_crs_obj, always_xy=True)
            
            # Plot AOI
            aoi_lons, aoi_lats = aoi_lon_lat_corners(ICE_AOI)
            # close the polygon
            aoi_lons.append(aoi_lons[0])
            aoi_lats.append(aoi_lats[0])
            aoi_xs, aoi_ys = transformer.transform(aoi_lons, aoi_lats)
            ax.plot(aoi_xs, aoi_ys, color=GOLD, linewidth=3, linestyle="--", label="Target AOI", zorder=10)
            
            # Plot footprints
            working_count = 0
            failed_count = 0
            
            for _, row in inventory.iterrows():
                try:
                    with rasterio.open(row["lh_path"]) as sar_ds:
                        sar_crs = CRS.from_wkt(sar_ds.crs.to_wkt())
                        # Bounds in SAR CRS
                        xs = np.array([sar_ds.bounds.left, sar_ds.bounds.right, sar_ds.bounds.right, sar_ds.bounds.left, sar_ds.bounds.left])
                        ys = np.array([sar_ds.bounds.bottom, sar_ds.bounds.bottom, sar_ds.bounds.top, sar_ds.bounds.top, sar_ds.bounds.bottom])
                        
                        # SAR CRS -> GEO -> DEM CRS
                        sar_to_geo = Transformer.from_crs(sar_crs, geo_crs, always_xy=True)
                        lons, lats = sar_to_geo.transform(xs, ys)
                        dxs, dys = transformer.transform(lons, lats)
                        
                        usable = row["usable"]
                        color = GREEN if usable else RED
                        alpha = 0.4 if usable else 0.15
                        lw = 2 if usable else 1
                        
                        ax.plot(dxs, dys, color=color, linewidth=lw, alpha=0.8, zorder=5)
                        ax.fill(dxs, dys, color=color, alpha=alpha, zorder=4)
                        
                        if usable:
                            working_count += 1
                        else:
                            failed_count += 1
                except Exception as e:
                    print(f"Skipping {row['product_id']}: {e}")
                    
            # Add legends
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], color=GOLD, linewidth=3, linestyle="--", label="Target AOI (Faustini)"),
                mpatches.Patch(facecolor=GREEN, alpha=0.4, edgecolor=GREEN, linewidth=2, 
                             label=f"Works OK - Covers AOI ({working_count} datasets)"),
                mpatches.Patch(facecolor=RED, alpha=0.15, edgecolor=RED, linewidth=1, 
                             label=f"Fails - Misses AOI ({failed_count} datasets)"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=11,
                      facecolor=PANEL_BG, edgecolor=GOLD, labelcolor=WHITE)

    except Exception as e:
        print(f"Could not plot DEM background: {e}")
        
    ax.set_title("SAR Dataset Footprints vs Target AOI", color=GOLD, fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(colors=WHITE)
    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(WHITE)
        
    out_path = OUTPUT_DIR / "09_sar_footprint_map.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
