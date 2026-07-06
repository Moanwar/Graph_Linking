import torch
import torch.nn as nn
from typing import List, Optional
import math
import torch.nn.functional as F
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class MultiHeadEdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        # Query, Key, Value projections for nodes
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        
        # Edge feature projections
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(node_dim, node_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(node_dim)
        
    def forward(self, x, edge_index, edge_attr, device: str = 'cuda') -> torch.Tensor:
        N = x.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Project edge features
        edge_weights = self.edge_proj(edge_attr).view(-1, self.num_heads)
        # Compute attention scores
        src, dst = edge_index[0], edge_index[1]
        attn_scores = (q[src] * k[dst]).sum(-1) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + edge_weights

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout(attn_weights)
        # Aggregate messages
        messages = (v[dst] * attn_weights.unsqueeze(-1)).view(-1, self.num_heads * self.head_dim)
        out = torch.zeros_like(x)
        out = out.squeeze(0)
        out = out.scatter_add_(0, src.unsqueeze(-1).expand_as(messages), messages)
        
        # Project back and residual
        out = self.out_proj(out)
        out = self.ln(x + out)
        return out
    
class EdgeConvBlock(nn.Module):
    """Enhanced EdgeConv layer with multiple improvements.
    
    Args:
        in_feat: Input feature size.
        out_feats: List of output feature sizes for each layer.
        activation: Whether to use activation functions.
        dropout: Dropout probability.
        weighted_aggr: Whether to use weighted aggregation.
        leaky_relu_slope: Negative slope for LeakyReLU.
    """
    def __init__(self, in_feat, out_feat, edge_dim, dropout=0.2, heads=4):
        super().__init__()
        
        self.attention = MultiHeadEdgeAttention(in_feat, edge_dim, heads, dropout)
        # Edge processing MLP with gating
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_feat + edge_dim, out_feat * 2),
            nn.LayerNorm(out_feat * 2),
            nn.GLU(dim=-1),  # Gated Linear Unit
            nn.Dropout(dropout))
            
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_feat + out_feat, out_feat * 2),
            nn.LayerNorm(out_feat * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout))
        # Skip connection
        self.skip = nn.Linear(in_feat, out_feat) if in_feat != out_feat else nn.Identity()
        self.ln = nn.LayerNorm(out_feat)
        
    def forward(self, x, edge_index, edge_attr, device: str = 'cpu') -> torch.Tensor:

        if not torch.onnx.is_in_onnx_export():
            if edge_index.shape[0] == 1:
                edge_index = edge_index.squeeze(0)
        edge_index = edge_index.long()        

        # Apply attention first
        x = self.attention(x, edge_index, edge_attr)
        src, dst = edge_index[0], edge_index[1]

        x = x.squeeze(0)
        #edge_attr = edge_attr.squeeze(0)        

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(0)  # [1, 15]
        elif edge_attr.dim() == 3 and edge_attr.size(0) == 1:
            edge_attr = edge_attr.squeeze(0)

        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_out = self.edge_mlp(edge_features)

        # Aggregate edge info to nodes
        agg = torch.zeros_like(x[:, :edge_out.size(1)])
        agg = agg.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_out), edge_out)
        
        # Update nodes
        node_update = self.node_mlp(torch.cat([x, agg], dim=-1))
        out = self.ln(self.skip(x) + node_update)
        
        return out  
