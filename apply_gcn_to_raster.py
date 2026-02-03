"""
Apply trained GCN model to Sentinel-2 raster for spatial crop classification.

Reads S2_composite_24bands_2020_Q1.tif (24-band, 2262x1424),
builds per-tile KNN graphs, runs the GCN, and produces:
  - data/crop_classification_map.tif  (classified GeoTIFF)
  - figures/crop_classification_map.png (colour-coded PNG)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import warnings
import time

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    raise ImportError("rasterio is required. Install with: pip install rasterio")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\New_Calssification _Methods")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TIFF_PATH = DATA_DIR / "S2_composite_24bands_2020_Q1.tif"
MODEL_PATH = BASE_DIR / "best_gcn_model.pth"
CSV_PATH = DATA_DIR / "crop_training_data_5classes_2020.csv"

OUTPUT_TIF = DATA_DIR / "crop_classification_map.tif"
OUTPUT_PNG = FIG_DIR / "crop_classification_map.png"

SEED = 42
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
K_NEIGHBORS = 8
TILE_SIZE = 512          # pixels per tile side
NODATA_VALUE = 255       # nodata in output GeoTIFF

# Band names in the TIFF (1-indexed → 0-indexed in array)
TIFF_BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
    "B11", "B12", "NDVI", "EVI", "SAVI", "GNDVI", "NDRE", "NDRE2",
    "NDWI", "MNDWI", "BSI", "NDTI", "CIgreen", "CIrededge", "MSAVI", "GCVI",
]

# Class mapping (must match training)
IDX_TO_CLASS = {0: "Cotton", 1: "Wheat", 2: "Fallow", 3: "Grass", 4: "Water"}
NUM_CLASSES = len(IDX_TO_CLASS)

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# GCN MODEL (same architecture as training)
# ============================================================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.convs[-1](x, edge_index)
        return out


# ============================================================
# RECONSTRUCT SCALER FROM TRAINING DATA
# ============================================================

def build_training_scaler():
    """Reproduce the exact StandardScaler used during training."""
    print("Reconstructing training scaler from CSV...")
    df = pd.read_csv(CSV_PATH)
    drop_cols = [c for c in [".geo", "system:index", "GCVI"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.drop_duplicates().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["class", "classname"]]
    X = df[feature_cols].values
    y = df["class"].values

    # Same split as training
    indices = np.arange(X.shape[0])
    train_idx, _ = train_test_split(
        indices, test_size=0.3, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    print(f"  Scaler fitted on {len(train_idx)} training samples, {len(feature_cols)} features")
    print(f"  Feature order: {feature_cols}")
    return scaler, feature_cols


# ============================================================
# MAP TIFF BANDS → TRAINING FEATURE ORDER
# ============================================================

def get_band_reorder_indices(feature_cols):
    """Return indices to reorder TIFF bands (minus GCVI) to match training feature order."""
    # Build map: band_name → 0-based index in the TIFF
    tiff_name_to_idx = {name: i for i, name in enumerate(TIFF_BAND_NAMES)}

    # We drop GCVI (index 23), keep the rest
    indices = []
    for feat in feature_cols:
        if feat not in tiff_name_to_idx:
            raise ValueError(f"Feature '{feat}' not found in TIFF band names")
        indices.append(tiff_name_to_idx[feat])
    return indices


# ============================================================
# TILE PROCESSING
# ============================================================

def classify_tile(pixel_features, model, device, k=K_NEIGHBORS):
    """
    Classify a set of valid pixels using the GCN.

    Parameters
    ----------
    pixel_features : np.ndarray, shape (N, num_features), already scaled
    model : GCN model (eval mode, on device)
    device : torch device
    k : number of KNN neighbors

    Returns
    -------
    predictions : np.ndarray, shape (N,), int class indices
    """
    n = pixel_features.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    # For very small tiles, reduce k
    k_actual = min(k, n - 1)
    if k_actual < 1:
        # Only 1 pixel — can't build a graph; just do a forward pass with self-loop
        x_t = torch.tensor(pixel_features, dtype=torch.float).to(device)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(x_t, edge_index)
        return out.argmax(dim=1).cpu().numpy()

    # Build KNN graph
    knn_adj = kneighbors_graph(
        pixel_features, n_neighbors=k_actual, mode="connectivity", include_self=False
    )
    knn_adj = knn_adj + knn_adj.T
    knn_adj[knn_adj > 1] = 1
    rows, cols = knn_adj.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long).to(device)

    x_t = torch.tensor(pixel_features, dtype=torch.float).to(device)

    with torch.no_grad():
        out = model(x_t, edge_index)
    preds = out.argmax(dim=1).cpu().numpy()
    return preds


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 60)
    print("  Apply GCN to Sentinel-2 Raster")
    print("=" * 60)
    print(f"Device: {device}")

    # ----------------------------------------------------------
    # 1. Rebuild scaler & get feature order
    # ----------------------------------------------------------
    scaler, feature_cols = build_training_scaler()
    num_features = len(feature_cols)
    band_indices = get_band_reorder_indices(feature_cols)
    print(f"  Band reorder indices (TIFF -> training): {band_indices}")

    # ----------------------------------------------------------
    # 2. Load model
    # ----------------------------------------------------------
    print("\nLoading GCN model...")
    model = GCN(
        in_dim=num_features,
        hidden_dim=HIDDEN_DIM,
        out_dim=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded from {MODEL_PATH}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------------------------------------
    # 3. Read raster metadata
    # ----------------------------------------------------------
    print("\nReading raster...")
    with rasterio.open(TIFF_PATH) as src:
        height = src.height
        width = src.width
        n_bands = src.count
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        print(f"  Size: {width}x{height}, Bands: {n_bands}, CRS: {crs}")

        # Read all bands: shape (bands, height, width)
        raster = src.read()  # shape: (24, H, W)

    # ----------------------------------------------------------
    # 4. Prepare output array
    # ----------------------------------------------------------
    class_map = np.full((height, width), NODATA_VALUE, dtype=np.uint8)

    # ----------------------------------------------------------
    # 5. Process tiles
    # ----------------------------------------------------------
    print(f"\nProcessing tiles ({TILE_SIZE}x{TILE_SIZE})...")

    n_tiles_y = int(np.ceil(height / TILE_SIZE))
    n_tiles_x = int(np.ceil(width / TILE_SIZE))
    total_tiles = n_tiles_y * n_tiles_x
    tile_count = 0
    total_classified = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_count += 1
            r0 = ty * TILE_SIZE
            r1 = min(r0 + TILE_SIZE, height)
            c0 = tx * TILE_SIZE
            c1 = min(c0 + TILE_SIZE, width)

            # Extract tile: shape (bands, tile_h, tile_w)
            tile = raster[:, r0:r1, c0:c1]
            tile_h, tile_w = tile.shape[1], tile.shape[2]

            # Reshape to (n_pixels, bands)
            pixels = tile.reshape(n_bands, -1).T  # (n_pixels, 24)

            # Valid pixel mask: not NaN and not all-zero
            valid = ~np.isnan(pixels).any(axis=1) & (pixels.sum(axis=1) != 0)
            n_valid = valid.sum()

            if n_valid == 0:
                if tile_count % 5 == 0 or tile_count == total_tiles:
                    print(f"  Tile {tile_count}/{total_tiles}: no valid pixels, skipping")
                continue

            # Reorder bands to match training feature order & drop GCVI
            valid_pixels = pixels[valid][:, band_indices]  # (n_valid, 23)

            # Scale
            valid_scaled = scaler.transform(valid_pixels)

            # Classify
            preds = classify_tile(valid_scaled, model, device, k=K_NEIGHBORS)
            total_classified += n_valid

            # Write back into class_map
            tile_map = np.full(tile_h * tile_w, NODATA_VALUE, dtype=np.uint8)
            tile_map[valid] = preds.astype(np.uint8)
            class_map[r0:r1, c0:c1] = tile_map.reshape(tile_h, tile_w)

            if tile_count % 5 == 0 or tile_count == total_tiles:
                print(
                    f"  Tile {tile_count}/{total_tiles}: "
                    f"{n_valid:,} valid px classified "
                    f"(total so far: {total_classified:,})"
                )

    print(f"\nTotal classified pixels: {total_classified:,}")

    # ----------------------------------------------------------
    # 6. Save classified GeoTIFF
    # ----------------------------------------------------------
    print("\nSaving classified GeoTIFF...")
    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=1,
        nodata=NODATA_VALUE,
        compress="lzw",
    )

    with rasterio.open(OUTPUT_TIF, "w", **out_profile) as dst:
        dst.write(class_map[np.newaxis, :, :])
    print(f"  Saved: {OUTPUT_TIF}")

    # ----------------------------------------------------------
    # 7. Generate colour-coded classification map (PNG)
    # ----------------------------------------------------------
    print("\nGenerating classification map PNG...")

    # Colours for each class
    class_colors = {
        0: "#e6194b",  # Cotton — red
        1: "#f5d742",  # Wheat — yellow
        2: "#c4a35a",  # Fallow — tan
        3: "#3cb44b",  # Grass — green
        4: "#4363d8",  # Water — blue
    }

    # Build RGBA image
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for idx, hex_color in class_colors.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        mask = class_map == idx
        rgba[mask] = [r, g, b, 255]
    # nodata pixels stay transparent (alpha=0)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(rgba)
    ax.set_title("GCN Crop Classification Map", fontsize=16, fontweight="bold")
    ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=class_colors[i], label=IDX_TO_CLASS[i])
        for i in range(NUM_CLASSES)
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=11,
        framealpha=0.9,
        title="Crop Class",
        title_fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_PNG}")

    # ----------------------------------------------------------
    # 8. Summary
    # ----------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Classified pixels: {total_classified:,} / {height * width:,}")
    print(f"  Output GeoTIFF: {OUTPUT_TIF}")
    print(f"  Output PNG:     {OUTPUT_PNG}")

    # Per-class pixel counts
    print("\n  Per-class pixel counts:")
    for i in range(NUM_CLASSES):
        count = (class_map == i).sum()
        pct = 100.0 * count / max(total_classified, 1)
        print(f"    {IDX_TO_CLASS[i]:>10s}: {count:>10,}  ({pct:5.1f}%)")
    nodata_count = (class_map == NODATA_VALUE).sum()
    print(f"    {'NoData':>10s}: {nodata_count:>10,}")


if __name__ == "__main__":
    main()
