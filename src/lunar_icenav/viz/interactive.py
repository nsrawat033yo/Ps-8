"""
interactive.py - Generates interactive HTML maps for lunar landing and rover planning
"""
import pandas as pd
from pathlib import Path
import folium

def generate_interactive_rover_map(
    aoi: dict,
    landing_sites: pd.DataFrame,
    rover_routes_df: pd.DataFrame,
    output_path: Path
):
    """
    Generates an interactive Folium HTML map showing the AOI, landing sites, and rover routes.
    """
    center_lat = (aoi["lat_min"] + aoi["lat_max"]) / 2.0
    center_lon = (aoi["lon_min"] + aoi["lon_max"]) / 2.0
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )
    
    # Add AOI bounding box
    aoi_bounds = [
        [aoi["lat_min"], aoi["lon_min"]],
        [aoi["lat_max"], aoi["lon_min"]],
        [aoi["lat_max"], aoi["lon_max"]],
        [aoi["lat_min"], aoi["lon_max"]],
        [aoi["lat_min"], aoi["lon_min"]]
    ]
    folium.PolyLine(
        aoi_bounds,
        color="#ffd700",
        weight=2,
        dash_array="5, 5",
        tooltip="Target AOI (Faustini)"
    ).add_to(m)
    
    # Add landing sites
    for _, row in landing_sites.iterrows():
        site_id = row.get("site_id", "Landing Site")
        score = row.get("suitability_score", 0.0)
        
        popup_html = f"<b>{site_id}</b><br>Suitability Score: {score:.3f}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}"
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=site_id,
            icon=folium.Icon(color="green", icon="rocket", prefix="fa")
        ).add_to(m)

    # Add rover routes
    colors = {
        "safest": "#00e676",
        "science_priority": "#b388ff",
        "energy_efficient": "#ff9100",
        "shortest": "#9e9e9e"
    }
    
    if rover_routes_df is not None and not rover_routes_df.empty:
        for route_type in rover_routes_df["route_type"].unique():
            df_mode = rover_routes_df[rover_routes_df["route_type"] == route_type]
            # Folium expects lat/lon pairs
            path_coords = list(zip(df_mode["lat"], df_mode["lon"]))
            color = colors.get(route_type, "white")
            
            folium.PolyLine(
                path_coords,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"Route: {route_type}"
            ).add_to(m)
            
    m.save(str(output_path))
    return str(output_path)
