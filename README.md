# ChangeDetection: 3D Spatial Pipeline & Synthetic View Synthesis

This repository contains a complete, end-to-end pipeline for 3D spatial change detection and synthetic urban dataset generation, heavily inspired by the CYWS-3D architecture. 

The project is divided into three distinct phases: autonomous 360° data collection, geometric pose estimation, and synthetic change generation using diffusion models.

## 🚀 Project Architecture

### Phase 1: Autonomous Data Extraction (Completed)
* **OSM Topology Routing:** Generates uniform, unbiased GPS sampling nodes across urban environments, strictly constrained to 10m road corridors and 30m building buffers.
* **Panoramax & STAC Integration:** Scouts and filters strictly equirectangular (360°) images.
* **Time-Capsule Sampling:** Downloads 1-second-apart adjacent frames to guarantee mathematically calculable 3D parallax.
* **Resilient Infrastructure:** Features idempotent downloading, multiprocessing pools, and exponential API backoff.

### Phase 2: Geometry & Pose Estimation (In Progress)
* **Cubemap Unfolding:** Mathematically projects 360° spherical equirectangular images into 6 standard 2D perspective planes.
* **Camera Translation (MapAnything):** Utilizes pose-estimation models to extract the exact 3D translation vectors between adjacent frames.
* **Disocclusion Masking:** Calculates geometric depth masks to block out background pixels that become occluded as the camera moves forward.

### Phase 3: Synthetic Change Generation (Planned)
* **Generative Diffusion (ControlNet):** Injects realistic, synthetic structural changes (e.g., adding/removing street infrastructure) into the historical frames.
* **Parallax-Aware Synthesis:** Ensures synthetic objects perfectly respect the 3D geometry and camera movement calculated in Phase 2.

## 📂 Folder Structure

```text
ChangeDetection/
│
├── datasetBuilder/
│   ├── main_runner.py             # Phase 1: Orchestrator
│   ├── 01_generate_topology.py    # Phase 1: Spatial Map Generation
│   ├── 02_query_stac_links.py     # Phase 1: STAC Metadata Scout
│   ├── 03_download_manager.py     # Phase 1: Parallel Downloader
│   ├── 04_cubemap_transform.py    # Phase 2: Spherical Unfolding (WIP)
│   ├── 05_pose_estimation.py      # Phase 2: MapAnything Integration (WIP)
│   │
│   └── data/                      # Structured Time-Capsule Storage
│       ├── Bordeaux, France/
│       │   ├── target_nodes.geojson
│       │   ├── node_0000/
│       │   │   ├── base.jpg       
│       │   │   └── temporal_0.jpg 
│       └── ...
```

## 🛠️ Installation

Dependencies are managed using `uv` for high-performance geospatial C-library resolution.

```bash
uv pip install osmnx geopandas shapely requests
# Future Phase 2/3 dependencies (e.g., PyTorch, OpenCV) will be added here.
```

## 💻 Usage

To run the Phase 1 autonomous scraper:
```bash
python datasetBuilder/main_runner.py --output "/path/to/save/data"
```
