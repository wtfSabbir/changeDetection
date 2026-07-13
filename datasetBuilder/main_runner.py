import os
import argparse
from importlib import import_module
import osmnx as ox

ox.settings.timeout = 2000
ox.settings.use_cache = True
ox.settings.overpass_endpoint = "https://lz4.overpass-api.de/api/interpreter"

generate_topology = import_module("01_generate_topology").generate_target_nodes
query_stac = import_module("02_query_stac_links").query_stac_metadata
download_manager = import_module("03_download_manager").run_download_pool

print("!!!!! I AM RUNNING THE NEWEST VERSION OF THE SCRIPT !!!!!")

CITIES = ["Bordeaux, France"] # Example

#CITIES = [
#    "Toulouse, France", "Paris, France", "Lyon, France", "Marseille, France",
#    "Ajaccio, France", "Dijon, France", "Lille, France", 
    # ... up to 90 French cities ...
#    "Berlin, Germany", "Madrid, Spain", "Rome, Italy" 
    # ... 10 international cities ...
#]

def check_city_status(city_name, output_dir):
    # Dynamic path instead of hardcoded "data/"
    return os.path.exists(f"{output_dir}/{city_name}/stac_metadata.json")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run the Autonomous Panorama Scraper.")
    parser.add_argument(
        "--output", 
        type=str, 
        default="data", # Defaults to "data" if you forget the flag
        help="Path to the desired output directory (e.g., /Volumes/ExternalDrive/Scrape)"
    )
    args = parser.parse_args()
    output_dir = args.output

    print(f"Saving all data to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for city in CITIES:
        print(f"\n{'='*40}\nProcessing City: {city}\n{'='*40}")
        try:
            if not check_city_status(city, output_dir):
                # Pass the output_dir to the functions
                generate_topology(city, output_dir, max_points=1000)
                query_stac(city, output_dir)
            else:
                print(f"Metadata for {city} already exists. Resuming downloads...")
                
            # Pass the output_dir to the downloader
            download_manager(city, output_dir)
            
        except Exception as e:
            print(f"CRITICAL ERROR processing {city}: {e}")
            continue

if __name__ == "__main__":
    main()