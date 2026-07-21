import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
import os

if hasattr(ox, 'settings'):
    ox.settings.timeout = 2000
    ox.settings.use_cache = True
    # Reroute from the congested default server to the Kumi Systems or LZ4 server
    ox.settings.overpass_endpoint = "https://lz4.overpass-api.de/api/interpreter"

# 2. Force settings for older osmnx versions (v1.x)
try:
    ox.config(
        timeout=2000, 
        use_cache=True, 
        overpass_endpoint="https://lz4.overpass-api.de/api/interpreter"
    )
except AttributeError:
    pass

def generate_target_nodes(city_name, output_dir, max_points=1000):
    print(f"Generating topology for {city_name}...")
    import osmnx as ox
    ox.settings.timeout = 2000
    ox.settings.use_cache = True
    # 1. Download street network and building footprints
    graph = ox.graph_from_place(city_name, network_type='drive')
    buildings = ox.features_from_place(city_name, tags={'building': True})
    
    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)
    
    # 2. Interpolate points every 30-50m along the roads
    sampled_points = []
    for _, row in edges.iterrows():
        line = row['geometry']
        distance = 0
        # Random interval between 30 and 50 meters
        interval = random.randint(30, 50) 
        while distance < line.length:
            sampled_points.append(line.interpolate(distance))
            distance += interval
            
    points_gdf = gpd.GeoDataFrame(geometry=sampled_points, crs=edges.crs)
    
    # 3. Apply Strict Filtering (10m road corridor, 30m building buffer)
    # Note: in a real projection, you must project to a metric CRS (like UTM) before buffering
    points_gdf = points_gdf.to_crs(epsg=3857) 
    edges_metric = edges.to_crs(epsg=3857)
    buildings_metric = buildings.to_crs(epsg=3857)
    
    road_buffer = edges_metric.geometry.buffer(10)
    building_buffer = buildings_metric.geometry.buffer(30)
    
    # Keep points inside road buffer but outside building buffer
    valid_points = points_gdf[
        points_gdf.geometry.within(road_buffer.unary_union) & 
        ~points_gdf.geometry.within(building_buffer.unary_union)
    ]
    
    # Convert back to GPS coordinates (Lat/Lon)
    valid_points = valid_points.to_crs(epsg=4326)
    
    # 4. Quota Subsampling
    if len(valid_points) > max_points:
        valid_points = valid_points.sample(n=max_points, random_state=42)
        
    # Save to GeoJSON
    os.makedirs(f"{output_dir}/{city_name}", exist_ok=True)
    out_path = f"{output_dir}/{city_name}/target_nodes.geojson"
    valid_points.to_file(out_path, driver='GeoJSON')
    print(f"Saved {len(valid_points)} valid nodes for {city_name}.")

if __name__ == "__main__":
    # Example usage for a single city
    generate_target_nodes("Toulouse, France")