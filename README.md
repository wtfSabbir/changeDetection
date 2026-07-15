# ChangeDetection: Autonomous 360° Urban Panorama Pipeline (Phase 1)

This repository contains an automated, high-performance ETL (Extract, Transform, Load) pipeline designed to scrape, filter, and organize strictly equirectangular (360°) urban panoramas. 

It is built to generate training data for 3D spatial change detection and synthetic view synthesis (inspired by state-of-the-art research like CYWS-3D). By downloading 1-second-apart panoramic frames, the dataset allows for mathematically precise 3D parallax calculation and generative AI synthesis.

## 🚀 Key Features

* **Geospatial Precision:** Uses OpenStreetMap (OSM) topology to generate uniform, unbiased GPS sampling nodes across entire cities, strictly filtered within 10m road corridors and 30m building buffers.
* **Smart STAC Scouting:** Queries the open-source Panoramax API via the SpatioTemporal Asset Catalog (STAC) standard. It parses deep EXIF metadata to drop flat images and guarantee 100% spherical geometry.
* **Temporal Grouping:** Automatically groups "Before" and "After" (1-second apart adjacent frames) by specific geographic locations to capture true camera translation (parallax).
* **Engineered for Scale:**
  * **Parallel Pools:** Downloads gigabytes of imagery using multi-core multiprocessing.
  * **Network Resilience:** Implements exponential backoff to handle API rate limits gracefully.
  * **Idempotent & Resumable:** Safely skips existing files. If the pipeline crashes on node 500, it resumes exactly where it left off without redownloading data.

## 📂 Architecture & Folder Structure

The pipeline is split into isolated modules to prevent catastrophic failures during massive scrapes.

```text
ChangeDetection/
│
├── datasetBuilder/
│   ├── main_runner.py             # The Orchestrator (CLI Entry Point)
│   ├── 01_generate_topology.py    # The Cartographer (OSM routing & spatial math)
│   ├── 02_query_stac_links.py     # The Scout (STAC metadata & EXIF filtering)
│   ├── 03_download_manager.py     # The Heavy Lifter (Parallel downloads & backoff)
│   │
│   └── data/                      # Auto-generated Output Directory
│       ├── Bordeaux, France/
│       │   ├── target_nodes.geojson
│       │   ├── stac_metadata.json
│       │   ├── node_0000/
│       │   │   ├── base.jpg       # Frame 1
│       │   │   ├── temporal_0.jpg # Frame 2 (1 second prior)
│       │   │   └── temporal_1.jpg # Frame 3 (2 seconds prior)
│       │   └── node_0001/
│       └── ...
```

## 🛠️ Installation

This project manages dependencies using `uv` for lightning-fast package resolution, which is highly recommended for geospatial C-library bindings.

```bash
# Install the required dependencies
uv pip install osmnx geopandas shapely requests
```
*(Note: Standard libraries like multiprocessing, json, and argparse are already included in standard Python).*

## 💻 Usage

Run the orchestrator from your terminal. You can optionally specify a custom output directory (e.g., an external hard drive) using the `--output` flag.

```bash
python datasetBuilder/main_runner.py --output "/path/to/save/data"
```

To change the target cities, edit the `CITIES` list inside `main_runner.py`:
```python
CITIES = ["Saint-Émilion, France", "Bordeaux, France", "Paris, France"]
```
