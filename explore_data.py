import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load the CSV data
# ============================================================
DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\New_Calssification _Methods\data")
csv_path = DATA_DIR / "crop_training_data_5classes_2020.csv"

print("=" * 60)
print("  CROP CLASSIFICATION DATA EXPLORATION")
print("=" * 60)

df = pd.read_csv(csv_path)

# Drop the .geo column (JSON geometry, not needed for ML)
if ".geo" in df.columns:
    df = df.drop(columns=[".geo"])
if "system:index" in df.columns:
    df = df.drop(columns=["system:index"])

print(f"\n--- Basic Info ---")
print(f"Shape: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# ============================================================
# 2. Feature columns and target
# ============================================================
feature_cols = [c for c in df.columns if c not in ["class", "classname"]]
X = df[feature_cols]
y = df["class"]
class_names = df["classname"]

print(f"\n--- Features ({len(feature_cols)}) ---")
print(feature_cols)

# ============================================================
# 3. Class distribution
# ============================================================
print(f"\n--- Class Distribution ---")
class_dist = df.groupby(["class", "classname"]).size().reset_index(name="count")
class_dist["percentage"] = (class_dist["count"] / len(df) * 100).round(2)
print(class_dist.to_string(index=False))
print(f"\nTotal samples: {len(df)}")

# ============================================================
# 4. Descriptive statistics
# ============================================================
print(f"\n--- Feature Statistics ---")
print(X.describe().T.to_string())

# ============================================================
# 5. Missing values
# ============================================================
print(f"\n--- Missing Values ---")
missing = X.isnull().sum()
if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])

# ============================================================
# 6. Check for duplicates
# ============================================================
n_dup = df.duplicated().sum()
print(f"\n--- Duplicates ---")
print(f"Number of duplicate rows: {n_dup}")

# ============================================================
# 7. Check value ranges (detect potential outliers)
# ============================================================
print(f"\n--- Value Ranges per Feature ---")
for col in feature_cols:
    print(f"  {col:15s}  min={X[col].min():.4f}  max={X[col].max():.4f}  "
          f"mean={X[col].mean():.4f}  std={X[col].std():.4f}")

# ============================================================
# 8. Correlation between features
# ============================================================
print(f"\n--- Top Highly Correlated Feature Pairs (|r| > 0.95) ---")
corr = X.corr()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [(col, row, upper.loc[row, col])
             for col in upper.columns for row in upper.index
             if abs(upper.loc[row, col]) > 0.95]
high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, r in high_corr:
    print(f"  {f1:15s} <-> {f2:15s}  r = {r:.4f}")
if not high_corr:
    print("  None found.")

# ============================================================
# 9. Per-class feature means
# ============================================================
print(f"\n--- Per-Class Feature Means ---")
class_means = df.groupby("classname")[feature_cols].mean()
print(class_means.T.to_string())

# ============================================================
# PLOTS
# ============================================================
fig_dir = Path(r"D:\Udemy_Cour\Crops_Classification\New_Calssification _Methods\figures")
fig_dir.mkdir(exist_ok=True)

# Plot 1: Class distribution bar chart
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("Set2", len(class_dist))
bars = ax.bar(class_dist["classname"], class_dist["count"], color=colors)
for bar, pct in zip(bars, class_dist["percentage"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{pct}%", ha="center", fontsize=10)
ax.set_title("Class Distribution", fontsize=14)
ax.set_xlabel("Crop Type")
ax.set_ylabel("Sample Count")
plt.tight_layout()
plt.savefig(fig_dir / "01_class_distribution.png", dpi=150)
plt.close()
print(f"\nSaved: figures/01_class_distribution.png")

# Plot 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={"size": 6})
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "02_correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: figures/02_correlation_heatmap.png")

# Plot 3: Boxplots of spectral bands per class
band_cols = [c for c in feature_cols if c.startswith("B")]
index_cols = [c for c in feature_cols if c not in band_cols]

n_bands = len(band_cols)
ncols_b = 5
nrows_b = (n_bands + ncols_b - 1) // ncols_b
fig, axes = plt.subplots(nrows_b, ncols_b, figsize=(20, 4 * nrows_b))
axes = axes.flatten()
for i, col in enumerate(band_cols):
    df.boxplot(column=col, by="classname", ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45, labelsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Spectral Bands Distribution per Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "03_bands_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/03_bands_boxplot.png")

# Plot 4: Boxplots of vegetation indices per class
n_idx = len(index_cols)
ncols = 5
nrows = (n_idx + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
axes = axes.flatten()
for i, col in enumerate(index_cols):
    df.boxplot(column=col, by="classname", ax=axes[i])
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45, labelsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Vegetation Indices Distribution per Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "04_indices_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/04_indices_boxplot.png")

# Plot 5: Feature distributions (histograms) for key indices
key_indices = ["NDVI", "EVI", "NDWI", "SAVI", "MNDWI", "BSI"]
key_indices = [k for k in key_indices if k in feature_cols]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(key_indices):
    for cls_name in df["classname"].unique():
        subset = df[df["classname"] == cls_name][col]
        axes[i].hist(subset, bins=50, alpha=0.5, label=cls_name, density=True)
    axes[i].set_title(col, fontsize=12)
    axes[i].legend(fontsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Key Index Distributions by Class", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "05_key_indices_hist.png", dpi=150)
plt.close()
print(f"Saved: figures/05_key_indices_hist.png")

# Plot 6: Per-class mean feature profile (radar-like bar chart)
fig, ax = plt.subplots(figsize=(16, 6))
class_means_norm = (class_means - class_means.min()) / (class_means.max() - class_means.min())
class_means_norm.T.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("Normalized Per-Class Feature Means", fontsize=14)
ax.set_xlabel("Feature")
ax.set_ylabel("Normalized Mean")
ax.legend(title="Crop", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(fig_dir / "06_class_feature_profile.png", dpi=150)
plt.close()
print(f"Saved: figures/06_class_feature_profile.png")

print("\n" + "=" * 60)
print("  EXPLORATION COMPLETE")
print("=" * 60)
print(f"\nSummary:")
print(f"  - {df.shape[0]} samples, {len(feature_cols)} features, {df['class'].nunique()} classes")
print(f"  - Classes: {dict(zip(class_dist['classname'], class_dist['count']))}")
print(f"  - Imbalanced: Fallow (45%) dominates, Cotton (1.4%) and Water (0.6%) are rare")
print(f"  - Features: 10 Sentinel-2 bands + 14 vegetation/spectral indices")
print(f"  - All plots saved to: figures/")
