import os
import json
import time
import requests
from multiprocessing import Pool

def download_with_backoff(url, save_path, max_retries=5):
    """Downloads an image using exponential backoff."""
    if os.path.exists(save_path):
        return # Idempotent: skip if already downloaded
        
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            elif response.status_code == 429: # Too Many Requests
                time.sleep((2 ** retries)) # Exponential backoff: 1, 2, 4, 8, 16s
                retries += 1
            else:
                break
        except requests.RequestException:
            time.sleep((2 ** retries))
            retries += 1
    return False

def process_node(node_data, city_name, output_dir):
    node_id = node_data['node_id']
    folder_path = f"{output_dir}/{city_name}/node_{node_id:04d}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Download base image
    download_with_backoff(node_data['base_image_url'], f"{folder_path}/base.jpg")
    
    # Download temporal neighbors
    for i, url in enumerate(node_data['temporal_urls']):
        download_with_backoff(url, f"{folder_path}/temporal_{i}.jpg")

def run_download_pool(city_name, output_dir):
    print(f"Starting parallel downloads for {city_name}...")
    metadata_path = f"{output_dir}/{city_name}/stac_metadata.json"
    
    if not os.path.exists(metadata_path):
        return
        
    with open(metadata_path, "r") as f:
        nodes = json.load(f)
        
    # Parallel pools for speed
    with Pool(processes=8) as pool:
        # Pass multiple arguments using starmap or a wrapper
        pool.starmap(process_node, [(node, city_name, output_dir) for node in nodes])
        
    print(f"Downloads complete for {city_name}.")

if __name__ == "__main__":
    run_download_pool("Toulouse, France")