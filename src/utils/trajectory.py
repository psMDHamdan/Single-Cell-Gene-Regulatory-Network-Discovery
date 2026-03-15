import scanpy as sc
import numpy as np
import os
import argparse

def infer_trajectory(adata, root_gene=None, root_cell_idx=0):
    """
    Infers trajectory and pseudo-time using PAGA and DPT.
    """
    print("Starting trajectory inference...")
    
    # Compute neighbors and UMAP if not already present
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack')
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
        
    # PAGA
    sc.tl.leiden(adata)
    sc.tl.paga(adata, groups='leiden')
    sc.pl.paga(adata, plot=False)  # This computes the connectivities
    
    # Identify root cell
    if root_gene and root_gene in adata.var_names:
        root_cell_idx = np.argmax(adata[:, root_gene].X)
        print(f"Using cell with highest {root_gene} expression as root (Index: {root_cell_idx})")
    
    adata.uns['iroot'] = root_cell_idx
    
    # DPT (Diffusion Pseudotime)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    
    print("Trajectory inference complete. 'dpt_pseudotime' added to adata.obs.")
    return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer trajectory for scRNA-seq data.")
    parser.add_argument("--input", type=str, default="data/pbmc3k_processed.h5ad", help="Path to processed h5ad file.")
    parser.add_argument("--output", type=str, default="data/pbmc3k_trajectory.h5ad", help="Path to save adata with trajectory.")
    parser.add_argument("--root_gene", type=str, default=None, help="Optional gene to identify root cell (e.g. CD34).")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
    else:
        adata = sc.read_h5ad(args.input)
        adata_traj = infer_trajectory(adata, root_gene=args.root_gene)
        adata_traj.write(args.output)
        print(f"Data with trajectory saved to {args.output}")
