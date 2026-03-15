import scanpy as sc
import pandas as pd
import numpy as np
import os
import argparse

def preprocess_adata(adata, min_genes=200, min_cells=3, n_top_genes=2000, n_cells=None):
    """
    Standard preprocessing pipeline for scRNA-seq data.
    """
    print("Starting preprocessing...")
    
    if n_cells is not None and n_cells < adata.n_obs:
        print(f"Subsampling to {n_cells} cells...")
        sc.pp.subsample(adata, n_obs=n_cells)
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter based on QC metrics (optional but recommended)
    # adata = adata[adata.obs.pct_counts_mt < 5, :]
    # adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    
    # Normalization and Log-transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Selected Highly Variable Genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
    # Scaling
    sc.pp.scale(adata, max_value=10)
    
    print(f"Preprocessing complete. Number of genes: {adata.n_vars}, Number of cells: {adata.n_obs}")
    return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq data.")
    parser.add_argument("--input", type=str, default="data/pbmc3k_raw.h5ad", help="Path to raw h5ad file.")
    parser.add_argument("--output", type=str, default="data/pbmc3k_processed.h5ad", help="Path to save processed adata.")
    parser.add_argument("--n_genes", type=int, default=2000, help="Number of HVGs to keep.")
    parser.add_argument("--n_cells", type=int, default=None, help="Number of cells to subsample.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found. Running download script first...")
        from download import download_pbmc3k
        adata = download_pbmc3k(args.input)
    else:
        adata = sc.read_h5ad(args.input)
        
    adata_processed = preprocess_adata(adata, n_top_genes=args.n_genes, n_cells=args.n_cells)
    adata_processed.write(args.output)
    print(f"Processed data saved to {args.output}")
