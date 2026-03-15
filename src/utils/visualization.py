import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

def visualize_grn(adj, genes, threshold=0.5, top_n=100, output_path="data/grn_plot.png"):
    """
    Visualize the top edges of the learned GRN.
    """
    # Filter for top edges
    adj_flat = adj.flatten()
    indices = np.argsort(adj_flat)[-top_n:]
    
    G = nx.DiGraph()
    for idx in indices:
        i, j = divmod(idx, len(genes))
        weight = adj[i, j]
        if weight >= threshold:
            G.add_edge(genes[i], genes[j], weight=weight)
            
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", 
            font_size=10, width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
            arrowsize=20)
    plt.title(f"Top {top_n} Regulatory Interactions")
    plt.savefig(output_path)
    plt.close()
    print(f"Network visualization saved to {output_path}")

if __name__ == "__main__":
    import argparse
    from models.model import GRNModel
    
    parser = argparse.ArgumentParser(description="Visualize inferred GRN.")
    parser.add_argument("--adata", type=str, default="data/pbmc3k_trajectory.h5ad")
    parser.add_argument("--model", type=str, default="data/grn_model.pt")
    parser.add_argument("--emb", type=str, default="data/gene_embeddings.npy")
    args = parser.parse_args()
    
    adata = sc.read_h5ad(args.adata)
    genes = adata.var_names
    gene_features = torch.tensor(np.load(args.emb), dtype=torch.float32)
    
    model = GRNModel(len(genes), gene_features.shape[1])
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    with torch.no_grad():
        adj = model.get_adj().detach().cpu().numpy()
        
    visualize_grn(adj, genes)
