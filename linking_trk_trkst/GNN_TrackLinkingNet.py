import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from EdgeConvBlock import EdgeConvBlock, MultiHeadEdgeAttention
from ClusterDataset.py import ClusterDataset
#from test_v5 import save_model
import math
import torch

def save_model(model, epoch, optimizer, loss, val_loss, output_folder, filename):
    path = os.path.join(output_folder, filename)

    print(f">>> Saving model to {path}")
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': loss,
                'validation_loss': val_loss
                }, path)

def weight_init(m):
    """Enhanced weight initialization for GNNs"""
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):  # Expanded for common layers
        # More precise gain calculation for LeakyReLU
        gain = nn.init.calculate_gain('leaky_relu', param=0.2)  # Match your LeakyReLU slope
        
        # He initialization with better fan_mode selection
        nn.init.kaiming_normal_(m.weight, 
                               mode='fan_in',  # Better for LeakyReLU
                               nonlinearity='leaky_relu',
                               a=0.2)  # Match your slope
        
        # Improved bias initialization
        if m.bias is not None:
            if hasattr(m, 'final_layer') and m.final_layer:  # Special case for output layer
                nn.init.constant_(m.bias, 0.0)  # Or task-specific value
            else:
                bound = 1 / math.sqrt(m.weight.size(1)) if m.weight.size(1) > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
    
    # Add initialization for other layer types
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
    # Optional: Add attention-specific initialization
    elif isinstance(m, (nn.MultiheadAttention, MultiHeadEdgeAttention)):
        for n, p in m.named_parameters():
            if 'weight' in n and p.dim() > 1:
                nn.init.xavier_uniform_(p)  # Often better for attention
            elif 'bias' in n:
                nn.init.constant_(p, 0.0)

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, input, target):
        loss = F.binary_cross_entropy(input, target, reduction='none')
        weight = torch.where(target > 0.5, self.pos_weight, 1.0)
        return (loss * weight).mean()
    
def prepare_network_input_data(X, edge_index, edge_features=None):
    X = torch.nan_to_num(X, nan=0.0)
    X = X[:, ClusterDataset.model_feature_keys]
    if edge_features is not None:
        edge_features = torch.nan_to_num(edge_features, nan=0.0)
        if edge_features.dim() == 1:
            edge_features = edge_features.view(1, -1)  # Ensure 2D
        return torch.unsqueeze(X, dim=0).float(), torch.unsqueeze(edge_index, dim=0).float(), torch.unsqueeze(edge_features, dim=0).float()
    return torch.unsqueeze(X, dim=0).float(), torch.unsqueeze(edge_index, dim=0).float()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    #def __call__(self, model, val_loss, loss, epoch, model_folder, optimizer):
    def __call__(self, model, F1, val_loss, loss, epoch, model_folder, optimizer):
        score = val_loss
        #score = F1
        if self.best_score is None or score < self.best_score - self.delta:
            print(f"Loss value improved from {self.best_score if self.best_score is not None else 'N/A'} to {val_loss:.6f} at epoch {epoch}. Resetting counter.")
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = model.state_dict()
            save_model(model, epoch, optimizer, loss, val_loss,
                       output_folder=model_folder,
                       filename=f"best_model_epoch_{epoch+1}_loss_{val_loss:.4f}.pt")            
            self.counter = 0
            
        else:
            self.counter += 1
            print(f"No improvement in val loss: {val_loss:.6f}, The best score {self.best_score}, Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        else:
            print("Warning: No best model state to load.")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, base_alpha=0.5, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.base_alpha = base_alpha  # Default balance when no batch stats
        self.label_smoothing = label_smoothing

    def forward(self, preds, targets):
        # Masking
        mask = (targets >= 0)
        preds = preds[mask]
        targs = targets[mask]
        
        if len(targs) == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
        # Dynamic alpha calculation (smoothed)
        num_pos = max(1, (targs == 1).sum())  # Avoid division by zero
        num_neg = max(1, (targs == 0).sum())
        dynamic_alpha = num_neg / (num_pos + num_neg)
        effective_alpha = 0.5 * self.base_alpha + 0.5 * dynamic_alpha  # Smoothed
        
        # Label smoothing
        targs = targs * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Focal loss calculation
        ce_loss = F.binary_cross_entropy(preds, targs, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = targs * effective_alpha + (1 - targs) * (1 - effective_alpha)
        loss = (alpha_t * (1 - p_t)**self.gamma * ce_loss).mean()
        
        return loss

class GNN_TrackLinkingNet(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, output_dim=1, num_layers=3,
                 edge_feature_dim=12, heads=4, dropout=0.2):
        super().__init__()
        self.pos_encoder = None
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2))

        # Edge feature processing
        self.edge_input_net = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout))

        # Graph convolutional layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeConvBlock(
                    hidden_dim, hidden_dim, 
                    edge_dim=hidden_dim,  # From edge_input_net output
                    dropout=dropout,
                    heads=heads))        

        # Enhanced edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, output_dim),
            nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # Add positional encoding if applicable
        if self.pos_encoder is not None:
            pos = x[:, :3]  # Assuming first 3 dims are spatial
            x = x + self.pos_encoder(pos)
            
        # Process inputs
        x = self.input_net(x)
        edge_features = self.edge_input_net(edge_attr)
        # Apply graph convolutions
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_features, device=device)

        # Prepare edge predictions
        if not torch.onnx.is_in_onnx_export():
            if edge_index.shape[0] == 1:
                edge_index = edge_index.squeeze(0)
        edge_index = edge_index.long()

        src, dst = edge_index[0], edge_index[1]

        if edge_features.dim() == 3:
            edge_features = edge_features.squeeze(0)

        if edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(0)
            
        edge_emb = torch.cat([
            x[src], 
            x[dst], 
            edge_features,
            edge_attr], dim=-1)
        return self.edge_classifier(edge_emb)
