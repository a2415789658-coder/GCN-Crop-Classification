# Requirements: pip install streamlit plotly pillow numpy
# Run with:     streamlit run app.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
FIGURES = BASE_DIR / "figures"

st.set_page_config(page_title="GCN Crop Classifier", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = '''
<style>
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 10px 24px; border-radius: 6px 6px 0 0; font-weight: 600; }
.legend-square { display: inline-block; width: 18px; height: 18px; border-radius: 3px; margin-right: 8px; vertical-align: middle; border: 1px solid #555; }
.legend-row { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.95rem; }
.sidebar-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
.sidebar-desc { font-size: 0.88rem; color: #aaa; margin-bottom: 1.2rem; line-height: 1.45; }
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CLASSES = {"Cotton": "#e6194b", "Wheat": "#f5d742", "Fallow": "#c4a35a", "Grass": "#3cb44b", "Water": "#4363d8"}
PIXEL_COUNTS = {"Fallow": 697687, "Grass": 163758, "Wheat": 141385, "Cotton": 29329, "Water": 11696}

with st.sidebar:
    st.markdown('<p class="sidebar-title">🌾 GCN Crop Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-desc">Graph Convolutional Network for multi-class crop classification from multi-spectral satellite imagery. The model leverages spectral bands, vegetation indices, and spatial graph structure to achieve near-perfect accuracy across five land-cover classes.</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Class Legend")
    for name, color in CLASSES.items():
        st.markdown(f'<div class="legend-row"><span class="legend-square" style="background:{color};"></span>{name}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Model Statistics")
    ca, cb = st.columns(2)
    ca.metric("Parameters", "20,741")
    cb.metric("Features", "23")
    cc, cd = st.columns(2)
    cc.metric("Classes", "5")
    cd.metric("Accuracy", "99.9 %")
    st.markdown("---")
    st.caption("Built with Streamlit · PyTorch Geometric · Plotly")


def load_image(name: str):
    path = FIGURES / name
    if path.exists():
        return Image.open(path)
    return None


st.title("GCN Crop Classification Dashboard")
tab_map, tab_perf, tab_data, tab_arch = st.tabs(["📍 Classification Map", "📊 Model Performance", "🔎 Data Explorer", "🏗️ Architecture"])

# TAB 1: Classification Map
with tab_map:
    col_img, col_chart = st.columns([3, 2], gap="large")
    with col_img:
        st.subheader("Predicted Crop Map")
        img_map = load_image("crop_classification_map.png")
        if img_map:
            st.image(img_map, use_container_width=True)
        else:
            st.warning("Classification map not found at figures/crop_classification_map.png")
    with col_chart:
        st.subheader("Pixel-Count Statistics")
        names = list(PIXEL_COUNTS.keys())
        counts = list(PIXEL_COUNTS.values())
        colors = [CLASSES[n] for n in names]
        total = sum(counts)
        fig_bar = go.Figure(go.Bar(x=names, y=counts, marker_color=colors, text=[f"{c:,}  ({c / total * 100:.1f}%)" for c in counts], textposition="outside", textfont=dict(size=13)))
        fig_bar.update_layout(template="plotly_dark", height=420, margin=dict(t=30, b=40, l=50, r=20), yaxis_title="Pixel Count", xaxis_title="Class", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        fig_donut = go.Figure(go.Pie(labels=names, values=counts, hole=0.50, marker=dict(colors=colors, line=dict(color="#222", width=2)), textinfo="label+percent", textfont=dict(size=13)))
        fig_donut.update_layout(template="plotly_dark", height=340, margin=dict(t=20, b=20, l=20, r=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)

# TAB 2: Model Performance
with tab_perf:
    st.subheader("Training & Evaluation")
    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        st.markdown("##### Training Curves")
        img = load_image("gcn_training_curves.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("gcn_training_curves.png not found.")
    with r1r:
        st.markdown("##### Confusion Matrix")
        img = load_image("gcn_confusion_matrix.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("gcn_confusion_matrix.png not found.")
    st.markdown("---")
    r2l, r2r = st.columns(2, gap="medium")
    with r2l:
        st.markdown("##### Per-Class Accuracy")
        img = load_image("gcn_per_class_accuracy.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("gcn_per_class_accuracy.png not found.")
    with r2r:
        st.markdown("##### t-SNE Embeddings")
        img = load_image("gcn_tsne_embeddings.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("gcn_tsne_embeddings.png not found.")
    with st.expander("Normalised Confusion Matrix", expanded=False):
        img = load_image("gcn_confusion_matrix_norm.png")
        if img:
            st.image(img, use_container_width=True)

# TAB 3: Data Explorer
with tab_data:
    st.subheader("Dataset Exploration")
    dc1, dc2 = st.columns(2, gap="medium")
    with dc1:
        st.markdown("##### Class Distribution")
        img = load_image("01_class_distribution.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("01_class_distribution.png not found.")
    with dc2:
        st.markdown("##### Correlation Heatmap")
        img = load_image("02_correlation_heatmap.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("02_correlation_heatmap.png not found.")
    st.markdown("---")
    dc3, dc4 = st.columns(2, gap="medium")
    with dc3:
        st.markdown("##### Key Index Histograms")
        img = load_image("05_key_indices_hist.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("05_key_indices_hist.png not found.")
    with dc4:
        st.markdown("##### Class Feature Profiles")
        img = load_image("06_class_feature_profile.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("06_class_feature_profile.png not found.")
    with st.expander("Additional Plots", expanded=False):
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("**Bands Box Plot**")
            img = load_image("03_bands_boxplot.png")
            if img:
                st.image(img, use_container_width=True)
        with ex2:
            st.markdown("**Indices Box Plot**")
            img = load_image("04_indices_boxplot.png")
            if img:
                st.image(img, use_container_width=True)

# TAB 4: Architecture
with tab_arch:
    st.subheader("Graph Convolutional Network Architecture")
    ac1, ac2 = st.columns([3, 2], gap="large")
    with ac1:
        img = load_image("gcn_architecture.png")
        if img:
            st.image(img, use_container_width=True, caption="GCN Architecture Diagram")
        else:
            st.info("gcn_architecture.png not found.")
    with ac2:
        st.markdown('''
#### How It Works

The **Graph Convolutional Network (GCN)** treats each pixel as a
node in a graph. Edges connect spatially adjacent pixels so the
network can learn from both *spectral* and *spatial* context.

**Pipeline overview**

1. **Input** -- 23 features per pixel (multi-spectral bands +
   derived vegetation / water / soil indices).
2. **Graph Construction** -- A k-nearest-neighbour graph is
   built from spatial coordinates, creating an adjacency matrix
   that encodes local neighbourhood structure.
3. **GCN Layers** -- Two graph-convolutional layers with ReLU
   activations propagate information along edges, allowing each
   node to aggregate features from its neighbours.
4. **Classifier Head** -- A fully-connected layer maps the
   learned embeddings to 5 output classes.

**Key numbers**

| Detail | Value |
|--------|-------|
| Trainable parameters | 20,741 |
| Input features | 23 |
| Hidden dimension | 128 |
| Output classes | 5 |
| Test accuracy | 99.9 % |
''')
        st.markdown("---")
        st.markdown("> *The near-perfect accuracy demonstrates that combining spectral indices with graph-based spatial reasoning is highly effective for agricultural land-cover mapping.*")
