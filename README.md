# Crop Classification with Graph Convolutional Networks (GCN)

Pixel-level crop classification from Sentinel-2 satellite imagery using a Graph Convolutional Network built with PyTorch Geometric.

## Overview

This project classifies agricultural land into **5 crop/land-cover classes** using 23 spectral and vegetation index features derived from Sentinel-2 imagery:

| Class | Description |
|-------|-------------|
| Cotton | Cotton fields |
| Wheat | Wheat fields |
| Fallow | Bare / fallow land |
| Grass | Grassland / pasture |
| Water | Water bodies |

## Method

1. **Feature extraction** — 10 Sentinel-2 bands (B2–B12) + 13 spectral indices (NDVI, EVI, SAVI, etc.)
2. **Graph construction** — K-nearest neighbor graph (k=8) in feature space
3. **GCN training** — 3-layer GCN with batch normalization, dropout, and class-weighted loss
4. **Raster inference** — Tiled KNN-graph prediction over the full Sentinel-2 composite

## Project Structure

```
├── explore_data.py              # EDA and feature visualization
├── gcn_crop_classification.py   # GCN model training and evaluation
├── apply_gcn_to_raster.py       # Apply trained model to full raster
├── data/                        # Input data (not tracked in git)
│   ├── crop_training_data_5classes_2020.csv
│   ├── S2_composite_24bands_2020_Q1.tif
│   └── crop_classification_map.tif  (output)
└── figures/                     # Generated plots and maps
    ├── 01_class_distribution.png
    ├── 02_correlation_heatmap.png
    ├── gcn_training_curves.png
    ├── gcn_confusion_matrix.png
    ├── gcn_per_class_accuracy.png
    ├── gcn_tsne_embeddings.png
    └── crop_classification_map.png
```

## Requirements

- Python 3.9+
- PyTorch + PyTorch Geometric
- rasterio, scikit-learn, pandas, numpy, matplotlib, seaborn

Install with conda (recommended):

```bash
conda create -n geodl python=3.9
conda activate geodl
conda install pytorch torchvision -c pytorch
conda install pyg -c pyg
conda install rasterio scikit-learn pandas matplotlib seaborn -c conda-forge
```

## Usage

### 1. Explore data

```bash
python explore_data.py
```

### 2. Train the GCN

```bash
python gcn_crop_classification.py
```

Saves `best_gcn_model.pth` and evaluation plots to `figures/`.

### 3. Classify full raster

```bash
python apply_gcn_to_raster.py
```

Produces `data/crop_classification_map.tif` (GeoTIFF) and `figures/crop_classification_map.png`.

## Data

Training data is derived from Sentinel-2 imagery (2020 Q1) over an agricultural region (EPSG:32636, 10m resolution). The 24-band composite includes:

- **Spectral bands**: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- **Vegetation indices**: NDVI, EVI, SAVI, GNDVI, NDRE, NDRE2, NDWI, MNDWI, BSI, NDTI, CIgreen, CIrededge, MSAVI, GCVI

GCVI is dropped during training (duplicate of CIgreen), leaving 23 features.

## Results

The classification map produced by the GCN:

![Crop Classification Map](figures/crop_classification_map.png)

## License

MIT
