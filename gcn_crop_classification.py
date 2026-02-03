import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\New_Calssification _Methods\data")
FIG_DIR = Path(r"D:\Udemy_Cour\Crops_Classification\New_Calssification _Methods\figures")
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
K_NEIGHBORS = 8  # number of neighbors to build the graph

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: Loading Data")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "crop_training_data_5classes_2020.csv")

# Drop unnecessary columns
drop_cols = [c for c in [".geo", "system:index", "GCVI"] if c in df.columns]
df = df.drop(columns=drop_cols)  # GCVI is identical to CIgreen

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Separate features and labels
feature_cols = [c for c in df.columns if c not in ["class", "classname"]]
class_names = sorted(df["classname"].unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

X = df[feature_cols].values
y = df["class"].values
num_classes = len(np.unique(y))
num_features = X.shape[1]

print(f"Samples: {X.shape[0]}, Features: {num_features}, Classes: {num_classes}")
print(f"Classes: {class_names}")
print(f"Dropped GCVI (duplicate of CIgreen), duplicates removed")

# ============================================================
# 2. TRAIN / VAL / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: Splitting Data")
print("=" * 60)

# Stratified split: 70% train, 15% val, 15% test
indices = np.arange(X.shape[0])
train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=SEED)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ============================================================
# 3. NORMALIZE FEATURES (fit on train only)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: Normalizing Features")
print("=" * 60)

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[train_idx] = scaler.fit_transform(X[train_idx])
X_scaled[val_idx] = scaler.transform(X[val_idx])
X_scaled[test_idx] = scaler.transform(X[test_idx])

print(f"StandardScaler fitted on training set")

# ============================================================
# 4. BUILD GRAPH (KNN-based)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: Building KNN Graph")
print("=" * 60)

# Build a K-nearest neighbor graph from feature similarity
# This creates the adjacency structure needed for GCN
knn_adj = kneighbors_graph(X_scaled, n_neighbors=K_NEIGHBORS, mode='connectivity', include_self=False)
knn_adj = knn_adj + knn_adj.T  # make symmetric (undirected)
knn_adj[knn_adj > 1] = 1  # remove duplicate edges

# Convert to edge_index format for PyG
rows, cols = knn_adj.nonzero()
edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

print(f"Graph: {X_scaled.shape[0]} nodes, {edge_index.shape[1]} edges")
print(f"Avg degree: {edge_index.shape[1] / X_scaled.shape[0]:.1f}")

# ============================================================
# 5. CREATE PyG DATA OBJECT
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Creating PyG Data Object")
print("=" * 60)

x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create masks
train_mask = torch.zeros(X_scaled.shape[0], dtype=torch.bool)
val_mask = torch.zeros(X_scaled.shape[0], dtype=torch.bool)
test_mask = torch.zeros(X_scaled.shape[0], dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data = Data(
    x=x_tensor,
    edge_index=edge_index,
    y=y_tensor,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

print(f"Data object: {data}")

# ============================================================
# 6. COMPUTE CLASS WEIGHTS (handle imbalance)
# ============================================================
train_labels = y[train_idx]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes  # normalize
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"\nClass weights (for imbalance): {class_weights.cpu().numpy().round(3)}")

# ============================================================
# 7. GCN MODEL
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: Model Architecture")
print("=" * 60)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, out_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer (no activation, no BN, no dropout)
        out = self.convs[-1](x, edge_index)
        return out

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings from the second-to-last layer."""
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


model = GCN(
    in_dim=num_features,
    hidden_dim=HIDDEN_DIM,
    out_dim=num_classes,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ============================================================
# 8. TRAINING
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: Training")
print("=" * 60)

model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights)


def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc


train_losses = []
val_accs = []
best_val_acc = 0.0
best_epoch = 0
patience = 30
patience_counter = 0

print("..Training Model..\n")

for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    val_acc = evaluate(data.val_mask)
    train_losses.append(loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), FIG_DIR.parent / "best_gcn_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS},  Loss: {loss:.4f},  Val Acc: {val_acc:.4f}")

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
        break

print(f"\n..Training Complete..")
print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

# ============================================================
# 9. TEST EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: Test Evaluation")
print("=" * 60)

# Load best model
model.load_state_dict(torch.load(FIG_DIR.parent / "best_gcn_model.pth"))
model.eval()

test_acc = evaluate(data.test_mask)
print(f"Test Accuracy: {test_acc:.4f}")

# Detailed classification report
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()

# Map class indices to names
idx_to_class = {0: "Cotton", 1: "Wheat", 2: "Fallow", 3: "Grass", 4: "Water"}
target_names = [idx_to_class[i] for i in range(num_classes)]

print(f"\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=target_names, digits=4))

# ============================================================
# 10. PLOTS
# ============================================================
print("=" * 60)
print("  STEP 9: Generating Plots")
print("=" * 60)

# Plot 1: Training loss & validation accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, color='blue')
ax1.set_title('Training Loss', fontsize=14)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(val_accs, color='green')
ax2.set_title('Validation Accuracy', fontsize=14)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "gcn_training_curves.png", dpi=150)
plt.close()
print("Saved: figures/gcn_training_curves.png")

# Plot 2: Confusion matrix
cm = confusion_matrix(test_true, test_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names,
            yticklabels=target_names, ax=ax)
ax.set_title('Confusion Matrix (Test Set)', fontsize=14)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIG_DIR / "gcn_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: figures/gcn_confusion_matrix.png")

# Plot 3: Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=target_names,
            yticklabels=target_names, ax=ax)
ax.set_title('Normalized Confusion Matrix (Test Set)', fontsize=14)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIG_DIR / "gcn_confusion_matrix_norm.png", dpi=150)
plt.close()
print("Saved: figures/gcn_confusion_matrix_norm.png")

# Plot 4: Per-class accuracy bar chart
per_class_acc = cm_norm.diagonal()
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("Set2", num_classes)
bars = ax.bar(target_names, per_class_acc, color=colors)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", fontsize=11)
ax.set_title("Per-Class Accuracy", fontsize=14)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "gcn_per_class_accuracy.png", dpi=150)
plt.close()
print("Saved: figures/gcn_per_class_accuracy.png")

# Plot 5: t-SNE of node embeddings
print("Computing t-SNE embeddings...")
from sklearn.manifold import TSNE

with torch.no_grad():
    embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(num_classes):
    mask = (y == i)
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=5, alpha=0.5, label=target_names[i])
ax.legend(fontsize=10, markerscale=4)
ax.set_title("Node Embeddings (t-SNE)", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "gcn_tsne_embeddings.png", dpi=150)
plt.close()
print("Saved: figures/gcn_tsne_embeddings.png")

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Best model saved to: best_gcn_model.pth")
print(f"All plots saved to: figures/")
