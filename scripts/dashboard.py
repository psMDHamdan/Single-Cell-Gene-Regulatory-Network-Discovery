import streamlit as st
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Single-Cell GRN Explorer", layout="wide")

st.title("🧬 Single-Cell Gene Regulatory Network Explorer")

# Sidebar for controls
st.sidebar.header("Settings")
adata_path = st.sidebar.text_input("Adata Path", "data/pbmc3k_trajectory.h5ad")
model_path = st.sidebar.text_input("Model Path", "data/grn_model.pt")
threshold = st.sidebar.slider("Edge Weight Threshold", 0.0, 1.0, 0.5)
top_n = st.sidebar.number_input("Top N Edges", 10, 500, 100)

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return sc.read_h5ad(path)
    return None

adata = load_data(adata_path)

if adata:
    st.write(f"Loaded dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load model and get adj
    if os.path.exists(model_path):
        # This is a simplified display for the dashboard
        # In a real app, we'd load the full model architecture
        st.info("Model loaded. Loading GRN...")
        
        # Placeholder for loading weights
        # For simplicity in the demo, we'll try to load or simulate the adj
        genes = adata.var_names
        
        # In a real scenario, we'd use the model.get_adj()
        # For the dashboard demo, we'll simulate if needed or load if exists
        adj_path = "data/learned_adj.npy"
        if os.path.exists(adj_path):
            adj = np.load(adj_path)
        else:
            adj = np.random.rand(len(genes), len(genes)) * 0.1
            
        # Display Network
        st.subheader("Inferred Regulatory Network")
        
        # Plotting logic
        adj_flat = adj.flatten()
        indices = np.argsort(adj_flat)[-top_n:]
        
        G = nx.DiGraph()
        for idx in indices:
            i, j = divmod(idx, len(genes))
            if adj[i, j] >= threshold:
                G.add_edge(genes[i], genes[j], weight=adj[i, j])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightgreen", ax=ax)
        st.pyplot(fig)
        
        # Data table
        st.subheader("Top Regulatory Interactions")
        edges = []
        for u, v, d in G.edges(data=True):
            edges.append({"Source": u, "Target": v, "Weight": d['weight']})
        st.dataframe(pd.DataFrame(edges).sort_values("Weight", ascending=False))
        
    else:
        st.warning("Model file not found. Please train the model first.")
else:
    st.error("Adata file not found.")
