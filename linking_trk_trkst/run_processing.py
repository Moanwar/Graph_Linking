import sklearn
print(sklearn.__version__)
import torch
print(torch.version.cuda)

import os
import time
import datetime as dt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ClusterDataset_trk_trkst import ClusterDataset
from GNN_TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init, prepare_network_input_data
from training import *
from data_statistics import *
from test import *
import math
# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
#first disk
#hist_folder_train = "/cms/data/store/user/moanwar/Link_single_interfaceDisk/211/histo/"
#data_folder_training = "/cms/data/store/user/moanwar/prossesed_pion_interfaceDesk/" 

#hist_folder_train = "/cms/data/store/user/moanwar/normalLink_single_firstDesk/211/histo/"
#data_folder_training = "/cms/data/store/user/moanwar/prossesed_pion_firstDesk/"

#mix
#hist_folder_train = "/cms/data/store/user/moanwar/Link_mix_FirstDisk/211/histo/"
#data_folder_training = "/cms/data/store/user/moanwar/prossesed_pion_mix_firstDesk/"

hist_folder_train = "/cms/data/store/user/moanwar/Link_mix_interfaceDisk/211/histo/"
data_folder_training = "/cms/data/store/user/moanwar/prossesed_pion_mix_interfaceDesk/"

#testing 
#hist_folder_train = "/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/input_test/"
#data_folder_training = "/home/moanwar/linking/CMSSW_15_1_0_pre1/src/graph_nn/train_test/"
dataset_training = ClusterDataset(data_folder_training, hist_folder_train)
