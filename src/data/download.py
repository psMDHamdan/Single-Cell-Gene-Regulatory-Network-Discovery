import scanpy as sc
import os
import argparse

def download_pbmc3k(output_path="data/pbmc3k_raw.h5ad"):
    """Download a standard PBMC 3k dataset from Scanpy for testing."""
    print(f"Downloading PBMC 3k dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata = sc.datasets.pbmc3k()
    adata.write(output_path)
    print("Download complete.")
    return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample scRNA-seq data.")
    parser.add_argument("--output", type=str, default="data/pbmc3k_raw.h5ad", help="Path to save the dataset.")
    args = parser.parse_args()
    
    download_pbmc3k(args.output)
