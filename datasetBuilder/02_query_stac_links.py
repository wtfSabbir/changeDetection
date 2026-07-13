import json
import requests
import geopandas as gpd
import os

PANORAMAX_API_URL = "https://api.panoramax.xyz/api/search"

def query_stac_metadata(city_name, output_dir):
    print(f"Scouting metadata for {city_name}...")
    file_path = f"{output_dir}/{city_name}/target_nodes.geojson"
    
    if not os.path.exists(file_path):
        return
        
    points_gdf = gpd.read_file(file_path)
    enriched_data = []

    for idx, row in points_gdf.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        
        # Query API for images within a small radius
        params = {
            "bbox": f"{lon-0.0001},{lat-0.0001},{lon+0.0001},{lat+0.0001}",
            "limit": 1
        }
        
        response = requests.get(PANORAMAX_API_URL, params=params)
        if response.status_code != 200:
            continue
            
        features = response.json().get('features', [])
        
        # 1. Strict filtering: equirectangular-only
        equi_features = [
            f for f in features 
            if f.get('properties', {}).get('exif', {}).get('Xmp.GPano.ProjectionType') == 'equirectangular'
        ]
        
        if not equi_features:
            continue
            
        # 2. Extract +/- 2 temporal neighbors via STAC links
        # (Assuming Panoramax STAC provides a 'collection' or temporal links)
        # For this blueprint, we simulate storing the valid download links
        base_image = equi_features[0]
        stac_links = base_image.get('links', [])
        
        temporal_neighbors = [
            link['href'] for link in stac_links 
            if link.get('rel') in ['prev', 'next'] # simplified STAC relation logic
        ]
        
        if temporal_neighbors:
            enriched_data.append({
                "node_id": idx,
                "lat": lat,
                "lon": lon,
                "base_image_url": base_image['assets']['hd']['href'],
                "temporal_urls": temporal_neighbors[:4] # up to +/- 2
            })
            
    # Save enriched metadata
    with open(f"{output_dir}/{city_name}/stac_metadata.json", "w") as f:
        json.dump(enriched_data, f, indent=4)
        
    print(f"Found {len(enriched_data)} valid temporal sequences for {city_name}.")

if __name__ == "__main__":
    query_stac_metadata("Toulouse, France")