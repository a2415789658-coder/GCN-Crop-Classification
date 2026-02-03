# Crop Classification with Graph Convolutional Networks (GCN)

Pixel-level crop classification from **Sentinel-2** satellite imagery using a **Graph Convolutional Network** built with PyTorch Geometric. The model classifies agricultural land into 5 crop/land-cover classes at 10 m spatial resolution.

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Project Structure](#project-structure)
- [Results](#results)
  - [Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [Model Training](#2-model-training)
  - [Model Evaluation](#3-model-evaluation)
  - [Spatial Classification Map](#4-spatial-classification-map)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [License](#license)

---

## Overview

This project classifies agricultural land into **5 crop/land-cover classes** using 23 spectral and vegetation index features derived from Sentinel-2 imagery:

| Class | Description |
|:------|:------------|
| **Cotton** | Cotton crop fields |
| **Wheat** | Wheat crop fields |
| **Fallow** | Bare / fallow agricultural land |
| **Grass** | Grassland and pasture areas |
| **Water** | Rivers, canals, and water bodies |

## Method

```
Sentinel-2 Image (24 bands)
        |
        v
Feature Extraction (23 features: 10 spectral bands + 13 vegetation indices)
        |
        v
KNN Graph Construction (k=8 neighbors in feature space)
        |
        v
3-Layer GCN (128 hidden units, BatchNorm, Dropout, class-weighted loss)
        |
        v
Tiled Raster Inference (512x512 pixel tiles)
        |
        v
Classified Crop Map (GeoTIFF + PNG)
```

1. **Feature extraction** -- 10 Sentinel-2 bands (B2-B12) + 13 spectral indices (NDVI, EVI, SAVI, etc.)
2. **Graph construction** -- K-nearest neighbor graph (k=8) built in feature space to capture spectral similarity
3. **GCN training** -- 3-layer GCN with batch normalization, dropout (0.5), and inverse-frequency class weighting
4. **Raster inference** -- Tiled KNN-graph prediction over the full 2262x1424 Sentinel-2 composite

## Project Structure

```
.
|-- explore_data.py              # EDA and feature visualization
|-- gcn_crop_classification.py   # GCN model training and evaluation
|-- apply_gcn_to_raster.py       # Apply trained model to full raster
|-- requirements.txt             # Python dependencies
|-- LICENSE                      # MIT License
|-- data/                        # Input data (not tracked in git)
|   |-- crop_training_data_5classes_2020.csv
|   |-- S2_composite_24bands_2020_Q1.tif
|   +-- crop_classification_map.tif  (output)
+-- figures/                     # Generated plots and maps
    |-- 01_class_distribution.png
    |-- 02_correlation_heatmap.png
    |-- 03_bands_boxplot.png
    |-- 04_indices_boxplot.png
    |-- 05_key_indices_hist.png
    |-- 06_class_feature_profile.png
    |-- gcn_training_curves.png
    |-- gcn_confusion_matrix.png
    |-- gcn_confusion_matrix_norm.png
    |-- gcn_per_class_accuracy.png
    |-- gcn_tsne_embeddings.png
    +-- crop_classification_map.png
```

---

## Results

### 1. Exploratory Data Analysis

#### 1.1 Class Distribution

The training dataset contains ~24,000 labeled pixels across 5 classes. The distribution is imbalanced -- Fallow dominates at 45%, while Cotton (1.4%) and Water (0.6%) are minority classes. This imbalance is addressed during training using inverse-frequency class weighting.

<p align="center">
  <img src="figures/01_class_distribution.png" alt="Class Distribution" width="700">
</p>

---

#### 1.2 Feature Correlation Matrix

The correlation heatmap reveals the relationships between all 23 spectral and index features. Strong positive correlations exist among vegetation indices (NDVI, EVI, SAVI, GNDVI) and among red-edge bands (B5-B7). Negative correlations appear between vegetation indices and bare-soil indicators (BSI, MNDWI), confirming their complementary roles for discrimination.

<p align="center">
  <img src="figures/02_correlation_heatmap.png" alt="Feature Correlation Matrix" width="700">
</p>

---

#### 1.3 Spectral Band Distributions per Class

Box plots of the 10 Sentinel-2 spectral bands (B2-B12) and BSI broken down by crop class. Each class exhibits a distinct spectral signature -- Wheat shows consistently high reflectance in NIR bands (B7, B8), Water has low reflectance across all bands, and Fallow is characterized by high short-wave infrared (B11, B12) values relative to NIR.

<p align="center">
  <img src="figures/03_bands_boxplot.png" alt="Spectral Bands Distribution per Class" width="800">
</p>

---

#### 1.4 Vegetation Index Distributions per Class

Box plots of the 13 derived vegetation indices per class. Wheat stands out with high NDVI, EVI, and SAVI values (active vegetation), while Fallow and Water cluster near zero or negative ranges. CIgreen and CIrededge provide strong separability between vegetated crops (Wheat, Grass) and non-vegetated surfaces (Fallow, Water).

<p align="center">
  <img src="figures/04_indices_boxplot.png" alt="Vegetation Index Distributions per Class" width="800">
</p>

---

#### 1.5 Key Index Histograms by Class

Density histograms of 6 key indices (NDVI, EVI, NDWI, SAVI, MNDWI, BSI) overlaid by class. These reveal the degree of separability each index provides. NDVI and SAVI show clear bimodal patterns separating vegetated from non-vegetated classes. Water is distinctly separated by NDWI and MNDWI with values far from other classes.

<p align="center">
  <img src="figures/05_key_indices_hist.png" alt="Key Index Distributions by Class" width="800">
</p>

---

#### 1.6 Normalized Per-Class Feature Profiles

A grouped bar chart showing the normalized mean value of every feature for each class. This "spectral fingerprint" view highlights how each class has a unique profile across the 23 features. Wheat dominates in vegetation-sensitive features (NDVI, EVI, SAVI), Fallow peaks in bare-soil indicators (BSI, B11), and Water shows near-zero values across most features except MNDWI.

<p align="center">
  <img src="figures/06_class_feature_profile.png" alt="Normalized Per-Class Feature Profiles" width="900">
</p>

---

### 2. Model Training

#### 2.1 Training Loss and Validation Accuracy Curves

The training loss decreases rapidly in the first 10 epochs and converges near zero, indicating effective learning. Validation accuracy climbs from ~79% to ~99.9%, with early stopping triggered after epoch 55. The smooth convergence with no divergence between training and validation suggests the model generalizes well without overfitting.

<p align="center">
  <img src="figures/gcn_training_curves.png" alt="Training Loss and Validation Accuracy" width="800">
</p>

---

### 3. Model Evaluation

#### 3.1 Confusion Matrix (Test Set)

The confusion matrix on the held-out test set (15% of data) shows near-perfect classification. All 5 classes achieve close to 100% recall, with only 3 misclassified samples total (all from the Grass class: 1 predicted as Cotton, 2 as Wheat). Cotton (48 samples), Wheat (1149), Fallow (1619), and Water (21) are classified with zero errors.

<p align="center">
  <img src="figures/gcn_confusion_matrix.png" alt="Confusion Matrix" width="550">
</p>

---

#### 3.2 Normalized Confusion Matrix (Test Set)

The row-normalized confusion matrix confirms all classes achieve >= 99.6% recall. Cotton, Wheat, Fallow, and Water reach a perfect 1.000, while Grass achieves 0.996. This demonstrates the GCN's ability to handle class imbalance effectively through weighted loss.

<p align="center">
  <img src="figures/gcn_confusion_matrix_norm.png" alt="Normalized Confusion Matrix" width="550">
</p>

---

#### 3.3 Per-Class Accuracy

A bar chart summarizing per-class accuracy on the test set. All 5 classes exceed 99.6%, confirming consistently strong performance across both majority classes (Fallow, Wheat) and minority classes (Cotton, Water).

<p align="center">
  <img src="figures/gcn_per_class_accuracy.png" alt="Per-Class Accuracy" width="600">
</p>

---

#### 3.4 t-SNE Visualization of GCN Node Embeddings

A 2D t-SNE projection of the learned 128-dimensional node embeddings from the GCN's second-to-last layer. The 5 classes form well-separated clusters, confirming that the GCN learns discriminative feature representations. Cotton (blue) and Water (purple) form tight, isolated clusters, while the larger classes (Fallow, Wheat, Grass) occupy distinct regions with clear boundaries.

<p align="center">
  <img src="figures/gcn_tsne_embeddings.png" alt="t-SNE of GCN Node Embeddings" width="600">
</p>

---

### 4. Spatial Classification Map

The final classified crop map produced by applying the trained GCN to the full Sentinel-2 raster (2262 x 1424 pixels, 10 m resolution). Over 1 million valid pixels were classified using tiled KNN-graph inference. Fallow (tan) dominates bare agricultural areas, Wheat (yellow) and Grass (green) cover vegetated parcels, Water (blue) aligns with river and canal features, and Cotton (red) appears in scattered agricultural plots.

| Class | Classified Pixels | Percentage |
|:------|------------------:|-----------:|
| **Fallow** | 697,687 | 66.8% |
| **Grass** | 163,758 | 15.7% |
| **Wheat** | 141,385 | 13.5% |
| **Cotton** | 29,329 | 2.8% |
| **Water** | 11,696 | 1.1% |

<p align="center">
  <img src="figures/crop_classification_map.png" alt="GCN Crop Classification Map" width="900">
</p>

---

## Installation

**Prerequisites:** Python 3.9+, CUDA-capable GPU (recommended)

```bash
# Create conda environment (recommended)
conda create -n geodl python=3.9
conda activate geodl

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install remaining dependencies
conda install rasterio scikit-learn pandas matplotlib seaborn -c conda-forge
```

Or install from `requirements.txt` (PyTorch and PyG must be installed separately):

```bash
pip install -r requirements.txt
```

## Usage

### 1. Explore data

```bash
python explore_data.py
```

Generates EDA visualizations in `figures/`.

### 2. Train the GCN

```bash
python gcn_crop_classification.py
```

Trains the model with early stopping and saves `best_gcn_model.pth` along with evaluation plots.

### 3. Classify full raster

```bash
python apply_gcn_to_raster.py
```

Applies the trained GCN to the Sentinel-2 composite and produces:
- `data/crop_classification_map.tif` -- Classified GeoTIFF (same CRS/transform as input)
- `figures/crop_classification_map.png` -- Color-coded visualization

## Data

Training data is derived from Sentinel-2 imagery (2020 Q1) over an agricultural region (EPSG:32636, 10 m resolution). The 24-band composite includes:

| Category | Features |
|:---------|:---------|
| **Spectral bands** | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| **Vegetation indices** | NDVI, EVI, SAVI, GNDVI, NDRE, NDRE2, NDWI, MNDWI, BSI, NDTI, CIgreen, CIrededge, MSAVI, GCVI |

> GCVI is dropped during training (duplicate of CIgreen), leaving **23 features**.

## License

This project is licensed under the [MIT License](LICENSE).
