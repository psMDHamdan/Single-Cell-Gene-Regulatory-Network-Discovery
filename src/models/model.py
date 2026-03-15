import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GRNModel(nn.Module):
    def __init__(self, n_genes, input_dim, hidden_dim=64, prior_adj=None):
        """
        GNN-based VAE for GRN inference.
        n_genes: Number of genes in the dataset.
        input_dim: Dimension of gene features (e.g. from gene embeddings).
        hidden_dim: Latent dimension for gene embeddings.
        prior_adj: (n_genes, n_genes) prior adjacency matrix from TRRUST/JASPAR.
        """
        super(GRNModel, self).__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        
        # Learnable adjacency matrix (the GRN)
        if prior_adj is not None:
            self.adj = nn.Parameter(torch.tensor(prior_adj, dtype=torch.float32))
        else:
            self.adj = nn.Parameter(torch.randn(n_genes, n_genes) * 0.01)
            
        # Encoder to process gene features
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder to reconstruct expression
        # We model expression of gene i as a function of expressions of its regulators (determined by self.adj)
        self.decoder_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
    def get_adj(self):
        """Returns the learned adjacency matrix with sparsity constraint applied later."""
        return torch.sigmoid(self.adj)

    def forward(self, x, gene_features):
        """
        x: (batch_size, n_genes) expression matrix
        gene_features: (n_genes, input_dim) static gene embeddings
        """
        # Embed genes
        gene_emb = self.gene_encoder(gene_features)  # (n_genes, hidden_dim)
        
        # Get learned GRN
        adj = self.get_adj()  # (n_genes, n_genes)
        
        # Reconstruct expression
        # Simple linear approximation: x_hat = x @ (adj * mask)
        # More complex version: x_hat = activation(x @ adj_scaled)
        
        # We can also use GNN logic here
        # But for GRN, we often want to predict x_i from x_j where j -> i
        # x_recon = torch.matmul(x, adj) # (batch, n_genes)
        
        # Adding some non-linearity
        x_recon = torch.sigmoid(torch.matmul(x, adj))
        
        return x_recon, adj

def loss_function(x_recon, x, adj, sparsity_lambda=1e-4, prior_adj=None, prior_lambda=1e-2):
    """
    Reconstruction loss + Sparsity regularization + Prior consistency.
    """
    recon_loss = F.mse_loss(x_recon, x)
    
    # L1 Sparsity on adjacency
    sparsity_loss = torch.norm(adj, 1)
    
    total_loss = recon_loss + sparsity_lambda * sparsity_loss
    
    if prior_adj is not None:
        # Penalize deviations from prior knowledge
        prior_adj_tensor = torch.tensor(prior_adj, dtype=torch.float32).to(adj.device)
        prior_loss = F.binary_cross_entropy(adj, prior_adj_tensor)
        total_loss += prior_lambda * prior_loss
        
    return total_loss, recon_loss, sparsity_loss
