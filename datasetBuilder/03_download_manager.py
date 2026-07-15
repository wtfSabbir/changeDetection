import os
import json
import time
import requests
from multiprocessing import Pool

def download_with_backoff(url, save_path, max_retries=5):
    """Downloads an image using exponential backoff, handling STAC JSON redirects."""
    if os.path.exists(save_path):
        return True # Idempotent: skip if already downloaded
        
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')

                # THE FIX: If the API hands us JSON metadata instead of an image...
                if 'json' in content_type:
                    stac_data = response.json()
                    # ...dig inside the JSON to find the real image URL!
                    real_image_url = stac_data['assets']['hd']['href']
                    
                    # Recursively call this function with the correct URL
                    return download_with_backoff(real_image_url, save_path, max_retries)
                else:
                    # It's an actual image! Save the bytes to the hard drive.
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                    
            elif response.status_code == 429: # Too Many Requests
                time.sleep((2 ** retries))
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