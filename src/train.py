import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scanpy as sc
import os
from models.model import GRNModel, loss_function
import argparse

def train_grn(adata_path, prior_adj_path, gene_emb_path, epochs=100, batch_size=32, lr=0.001):
    # Load data
    adata = sc.read_h5ad(adata_path)
    X = torch.tensor(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=torch.float32)
    prior_adj = np.load(prior_adj_path)
    gene_features = torch.tensor(np.load(gene_emb_path), dtype=torch.float32)
    
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_genes = X.shape[1]
    input_dim = gene_features.shape[1]
    
    model = GRNModel(n_genes, input_dim, prior_adj=prior_adj)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0
        for batch in loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            
            x_recon, adj = model(x_batch, gene_features)
            loss, r_loss, s_loss = loss_function(x_recon, x_batch, adj, prior_adj=prior_adj)
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_epoch_loss/len(loader):.6f}")
            
    print("Training complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRN inference model.")
    parser.add_argument("--adata", type=str, default="data/pbmc3k_trajectory.h5ad")
    parser.add_argument("--prior", type=str, default="data/prior_adj.npy")
    parser.add_argument("--emb", type=str, default="data/gene_embeddings.npy")
    parser.add_argument("--output_model", type=str, default="data/grn_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    if not all(os.path.exists(p) for p in [args.adata, args.prior, args.emb]):
        print("Missing input files. Please run data scripts first.")
    else:
        model = train_grn(args.adata, args.prior, args.emb, epochs=args.epochs)
        torch.save(model.state_dict(), args.output_model)
        print(f"Model saved to {args.output_model}")
