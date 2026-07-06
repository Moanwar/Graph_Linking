import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch

#from test import *
#from GNN_TrackLinkingNet_v5 import prepare_network_input_data, FocalLoss
from GNN_TrackLinkingNet_v5 import FocalLoss, prepare_network_input_data

def train(model, opt, loader, epoch, edge_features=True, emb_out=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), loss_obj=FocalLoss()):

    epoch_loss = 0
    model.train()
    for sample in tqdm(loader, desc=f"Training epoch {epoch}"):
        if sample is None:
            continue
        # reset optimizer and enable training mode
        opt.zero_grad()

        # move data to the device
        sample = sample.to(device)
        # get the prediction tensor
        if sample.x.size(0) == 0 or sample.edge_index.size(1) == 0:
            #print("Skipping empty graph sample")
            continue
        if edge_features:
            if sample.edge_index.shape[1] != sample.edge_attr.shape[0]:
                continue
            data = prepare_network_input_data(sample.x, sample.edge_index, edge_features=sample.edge_attr)
        else:
            data = prepare_network_input_data(sample.x, sample.edge_index)

        if emb_out:
            z, _ = model(*data, device=device)
        else:
            z = model(*data, device=device)
        z = z.view(-1)
        predictions = z
        # compute the loss
        #print("predictions:", predictions.min().item(), predictions.max().item(), predictions.isnan().any().item())

        loss = loss_obj(z, sample.y.float())

        # back-propagate and update the weight
        loss.backward()
        opt.step()
        epoch_loss += loss
        #print(" epoch_loss ", epoch_loss)
    return float(epoch_loss)/len(loader)
