# Data Analysis Module

This module provides tools for analyzing the BDD100K object detection dataset.

## BDD100K Label Format

BDD100K labels are stored as **individual JSON files per image** (not a single consolidated file):
- Structure: `bdd100k_labels/100k/train/*.json`, `bdd100k_labels/100k/val/*.json`
- Each JSON file contains annotations for one image
- File naming: `<image_id>.json` (e.g., `0000f77c-6257a9fd.json`)

## Structure

- `parser.py`: BDD100K JSON annotation parser with comprehensive analysis methods
- `analyze.py`: Main analysis script that generates statistics and reports
- `dashboard.py`: Interactive Streamlit dashboard for visualizing analysis results
- `Dockerfile`: Self-contained Docker image for running analysis
- `requirements.txt`: Python dependencies

## Running with Docker

### Build the container:
```bash
docker build -t bdd-analysis ./data_analysis
```

### Run analysis:
```bash
# The script expects directories of JSON files (train/ and val/)
docker run -v /path/to/bdd100k_labels:/data/labels -v $(pwd)/results:/app/results bdd-analysis

# Windows PowerShell:
docker run `
  -v .\BBD_Detection_Task\bdd100k_labels:/data/labels `
  -v .\BBD_Detection_Task\results:/app/results `
  bdd-analysis
```

### Run interactive dashboard:
```bash
docker run -p 8501:8501 -v $(pwd)/results:/app/results bdd-analysis streamlit run dashboard.py --server.address=0.0.0.0
```

## Usage without Docker

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run analysis:
```bash
# Provide directories containing per-image JSON files
python analyze.py \
  --train-json bdd100k_labels/100k/train \
  --val-json bdd100k_labels/100k/val \
  --output-dir results/
```

### Run dashboard:
```bash
streamlit run dashboard.py
```

## Output

Analysis generates the following files in the results directory:
- `class_distribution.json`: Object counts per class
- `split_comparison.json`: Train/val distribution comparison
- `image_statistics.json`: Image-level statistics
- `bbox_statistics.json`: Bounding box size and shape statistics
- `occlusion_statistics.json`: Occlusion and truncation rates
- `scene_attributes.json`: Weather, time of day distributions
- `anomalies.json`: Detected anomalies in the dataset
