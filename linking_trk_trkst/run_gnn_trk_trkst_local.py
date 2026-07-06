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

from ClusterDataset_trk_trkst_gpu import ClusterDataset
from GNN_TrackLinkingNet import GNN_TrackLinkingNet, FocalLoss, EarlyStopping, weight_init, prepare_network_input_data
from training import *
from data_statistics import *
from test import *
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch_geometric.data import Batch

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)


# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
#first disk
#firstDisk
#hist_folder = "/cms/data/store/user/moanwar/Link_single_firstDisk"
#data_folder = "/cms/data/store/user/moanwar/prossesed_had_firstDesk/"
#interface disk
#mix
hist_folder = "/cms/data/store/user/moanwar/Link_single_mix_interfaceDisk"
data_folder = "/cms/data/store/user/moanwar/prossesed_had_mix_interfaceDesk"
#testing 
#hist_folder_test = "/home/moanwar/linking/CMSSW_15_0_0_pre1/src/graph_nn/input_test"
#data_folder_test = "/home/moanwar/linking/CMSSW_15_0_0_pre1/src/graph_nn/train_test/"

#model_folder = "/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/models_gnnv1_firstLayer/"
model_folder = "/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/models_gnnv1_mix_interfaceLayer/"

full_dataset = ClusterDataset(data_folder, hist_folder)
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
dataset_training, dataset_test = random_split(full_dataset, [train_size, val_size])
print_dataset_statistics(full_dataset)
train_dl = DataLoader(dataset_training, batch_size=32, shuffle=True,  collate_fn=collate_skip_none)
test_dl= DataLoader(dataset_test, batch_size=32, shuffle=True,     collate_fn=collate_skip_none)

#dataset_training = ClusterDataset(data_folder_training, hist_folder_train)
#dataset_test = ClusterDataset(data_folder_test, hist_folder_test, test=True)
#train_dl = DataLoader(dataset_training, shuffle=True)
#test_dl = DataLoader(dataset_test, shuffle=True)
#data_sample = next(iter(DataLoader(dataset_training, batch_size=1)))
#print('data_sample = ',data_sample)

# Model setup
epochs = 400
start_epoch = 0

resume_path = None #"models_test/model_epoch_80_date_2025-06-18_loss_8.1590.pt"  # or None
                 
model = GNN_TrackLinkingNet(
    input_dim=full_dataset.model_feature_keys.shape[0],
    edge_feature_dim=full_dataset.get(0).edge_attr.shape[1]
)

model = model.to(device)
#tweaking alpha more aggressively (0.3 to 0.6) can help.

#optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=0.5)   
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
#scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2) 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3e-4,
    steps_per_epoch=len(train_dl),
    epochs=epochs)

loss_obj = FocalLoss(alpha=0.6, gamma=2)
early_stopping = EarlyStopping(patience=40, delta=1e-4)

model.apply(weight_init)

#os.makedirs(model_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

train_loss_hist = []
val_loss_hist = []
edge_features = True
date = f"{dt.datetime.now():%Y-%m-%d}"

fig_loss, ax_loss = plt.subplots(1, 1)
fig_loss.set_figwidth(6)
fig_loss.set_figheight(3)

fig_analysis, ax_analysis = plt.subplots(6, 2)
fig_analysis.set_figwidth(15)
fig_analysis.set_figheight(20)
fig_analysis.tight_layout(pad=2.0)

def eff_threshold_scores(scores, ground_truth, threshold_step=0.05):
    y = (ground_truth > 0).astype(int)
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    TNR, TPR = [], []

    for threshold in thresholds:
        prediction = (scores > threshold).astype(int)
        TN, FP, FN, TP = confusion_matrix(y, prediction, labels=[0,1]).ravel()
        TNR.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
        TPR.append(TP / (TP + FN) if (TP + FN) > 0 else 0)

    best_threshold = get_best_threshold(TNR, TPR, thresholds)
    
    pred_discrete = (scores > best_threshold).astype(int)
    y_discrete = y

    TN2, FP2, FN2, TP2 = confusion_matrix(y_discrete, pred_discrete, labels=[0,1]).ravel()
    Accuracy = balanced_accuracy_score(y_discrete, pred_discrete)
    Recall   = TP2 / (TP2 + FN2) * 100 if (TP2 + FN2) > 0 else 0
    FF1      = f1_score(y_discrete, pred_discrete, zero_division=0)

    return Accuracy, Recall, FF1


if resume_path is not None and os.path.exists(resume_path):
    print(f">>> Loading checkpoint from {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f">>> Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, epochs):
    print(f'Epoch: {epoch+1}')

    loss = train(model, optimizer, train_dl, epoch+1, device=device, edge_features=edge_features, loss_obj=loss_obj)
    train_loss_hist.append(loss)

    val_loss, pred, y = test(model, test_dl, epoch+1, loss_obj=loss_obj, edge_features=edge_features, device=device)
    val_loss_hist.append(val_loss)
    eff_threshold_scores(pred, y)
    print(" training loss : ", loss, " val_loss : ", val_loss)

    ax_loss.clear()
    plot_loss(train_loss_hist, val_loss_hist, ax=ax_loss)
    fig_loss.savefig(f"{model_folder}/loss_epoch_{epoch+1}.png", dpi=300)
    time.sleep(1)
    Accuracy, Recall, FF1 =  eff_threshold_scores(pred, y)
    
    #early_stopping(model, val_loss)
    early_stopping(model, FF1 ,val_loss, loss, epoch, model_folder, optimizer)
    #early_stopping(model, FF1, val_loss, loss, epoch, model_folder, optimizer)

    if early_stopping.early_stop:
        print(f"Early stopping after {epoch+1} epochs")
        early_stopping.load_best_model(model)

        for axes in ax_analysis:
            for ax in axes:
                ax.clear()

        plot_validation_results(pred, y, save=True, ax=ax_analysis, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}")
        time.sleep(1)
        break

    if ((epoch+1)%10==0 or epoch+1==epochs):
        for axes in ax_analysis:
            for ax in axes:
                ax.clear()

        plot_validation_results(pred, y, save=True, ax=ax_analysis, output_folder=model_folder, file_suffix=f"epoch_{epoch+1}_date_{date}")
        fig_analysis.savefig(f"{model_folder}/analysis_epoch_{epoch+1}.png", dpi=300)
        time.sleep(1)

        save_model(model, epoch, optimizer, loss, val_loss, output_folder=model_folder, filename=f"model_epoch_{epoch+1}_date_{date}_loss_{val_loss:.4f}.pt")

    elif ((epoch+1)%1==0):
        for axes in ax_analysis:
            for ax in axes:
                ax.clear()

        plot_validation_results(pred, y, save=False, ax=ax_analysis)
        fig_analysis.savefig(f"{model_folder}/analysis_epoch_{epoch+1}_preview_loss_{val_loss:.4f}.png", dpi=300)
        save_model(model, epoch, optimizer, loss, val_loss, output_folder=model_folder, filename=f"model_epoch_preview_{epoch+1}_date_{date}_loss_{val_loss:.4f}.pt")
        time.sleep(1)

    scheduler.step()
    #scheduler.step(val_loss)
    #print(f"Epoch {epoch+1}: LR = {scheduler.get_last_lr()[0]:.8f}")
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch+1} - Current LR: {param_group['lr']:.8f}")
