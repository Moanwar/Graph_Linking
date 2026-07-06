import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
import os
import datetime as dt

# Import your custom classes
from ClusterDataset_trk_trkst_gpu import ClusterDataset
from GNN_TrackLinkingNet_v7 import GNN_TrackLinkingNet
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def apply_high_threshold_cut(model_output, threshold=0.95):
    """Apply high-threshold cut to model predictions"""
    binary_predictions = (model_output > threshold).astype(int)
    return binary_predictions



def plot_final_recommendation(predictions, labels, recommended_threshold=0.95):
    """Plot showing the recommended cut"""
    plt.figure(figsize=(10, 6))
    
    true_scores = predictions[labels > 0.5]
    false_scores = predictions[labels <= 0.5]
    
    bins = np.linspace(0.9, 1.0, 51)
    
    plt.hist(true_scores, bins=bins, alpha=0.7, color='green', label='True Links', 
             density=False, edgecolor='black', linewidth=0.5)
    plt.hist(false_scores, bins=bins, alpha=0.7, color='red', label='False Links', 
             density=False, edgecolor='black', linewidth=0.5)
    
    # Highlight recommended threshold
    plt.axvline(x=recommended_threshold, color='blue', linestyle='-', 
                linewidth=3, label=f'Recommended Cut: {recommended_threshold}')
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title(f'Recommended High-Threshold Cut: {recommended_threshold}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.9, 1.0)
    
    # Add annotation
    pred_above_cut = np.sum(predictions > recommended_threshold)
    true_above_cut = np.sum(true_scores > recommended_threshold)
    false_above_cut = np.sum(false_scores > recommended_threshold)
    purity = true_above_cut / pred_above_cut if pred_above_cut > 0 else 0
    
    plt.annotate(f'Purity: {purity:.4f}\nTrue: {true_above_cut}\nFalse: {false_above_cut}', 
                xy=(0.95, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                horizontalalignment='right', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_recommendation.png", dpi=300, bbox_inches='tight')
    plt.show()



def plot_high_threshold_region(predictions, labels, save_path=None):
    """Zoom in on the high-threshold region"""
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    true_scores = predictions[labels > 0.5]
    false_scores = predictions[labels <= 0.5]
    
    plt.figure(figsize=(12, 6))
    
    # Plot only the high region
    high_bins = np.linspace(0.9, 1.0, 41)
    
    plt.hist(true_scores, bins=high_bins, alpha=0.7, color='green', label='True Links', 
             density=False, edgecolor='black', linewidth=0.5)
    plt.hist(false_scores, bins=high_bins, alpha=0.7, color='red', label='False Links', 
             density=False, edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    for threshold in [0.95, 0.97, 0.98, 0.985, 0.99]:
        plt.axvline(x=threshold, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Cut {threshold}' if threshold == 0.98 else "")
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('High-Threshold Region: True vs False Links (0.9-1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.9, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"High-threshold plot saved to: {save_path}")
    
    plt.show()


def analyze_high_threshold_cuts(predictions, labels, min_threshold=0.95, step=0.005):
    """Analyze performance at high thresholds to minimize false positives"""
    
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    thresholds = np.arange(min_threshold, 1.0, step)
    
    print("=== HIGH THRESHOLD ANALYSIS (≥0.95) ===")
    print("Threshold | Precision | Recall   | F1-score | TP  | FP  | FN  | Purity")
    print("-" * 75)
    
    best_f1 = 0
    best_threshold = min_threshold
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        
        if np.sum(pred_binary) == 0:  # No predictions above threshold
            continue
            
        precision = precision_score(labels, pred_binary, zero_division=0)
        recall = recall_score(labels, pred_binary, zero_division=0)
        f1 = f1_score(labels, pred_binary, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0,1]).ravel()
        
        # Purity = TP / (TP + FP)
        purity = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"{threshold:.3f}    | {precision:.4f}   | {recall:.4f}  | {f1:.4f}   | {tp:4d}| {fp:3d} | {fn:4d}| {purity:.4f}")
        
        if f1 > best_f1 and threshold >= 0.95:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def find_optimal_high_cut(predictions, labels):
    """Find the optimal high threshold with different optimization criteria"""
    
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    thresholds = np.arange(0.95, 1.0, 0.001)
    
    results = []
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        
        if np.sum(pred_binary) == 0:
            continue
            
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0,1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Purity (same as precision for binary)
        purity = precision
        
        # Efficiency (same as recall for binary)
        efficiency = recall
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'purity': purity,
            'efficiency': efficiency,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Find optimal based on different criteria
    if results:
        # 1. Best F1-score above 0.95
        best_f1 = max(results, key=lambda x: x['f1'])
        
        # 2. Highest purity (minimum false positives)
        best_purity = max(results, key=lambda x: x['purity'])
        
        # 3. Balanced: purity > 0.99 and reasonable efficiency
        balanced_candidates = [r for r in results if r['purity'] >= 0.99 and r['efficiency'] > 0.5]
        best_balanced = max(balanced_candidates, key=lambda x: x['f1']) if balanced_candidates else best_f1
        
        # 4. Very strict: purity > 0.995
        strict_candidates = [r for r in results if r['purity'] >= 0.995]
        best_strict = max(strict_candidates, key=lambda x: x['efficiency']) if strict_candidates else best_purity
        
        print("\n=== OPTIMAL THRESHOLD RECOMMENDATIONS ===")
        print(f"1. Best F1-score: {best_f1['threshold']:.3f}")
        print(f"   → F1: {best_f1['f1']:.4f}, Purity: {best_f1['purity']:.4f}, Efficiency: {best_f1['efficiency']:.4f}")
        print(f"   → TP: {best_f1['tp']}, FP: {best_f1['fp']}, FN: {best_f1['fn']}")
        
        print(f"\n2. Highest Purity: {best_purity['threshold']:.3f}")
        print(f"   → Purity: {best_purity['purity']:.4f}, Efficiency: {best_purity['efficiency']:.4f}, F1: {best_purity['f1']:.4f}")
        print(f"   → TP: {best_purity['tp']}, FP: {best_purity['fp']}, FN: {best_purity['fn']}")
        
        print(f"\n3. Balanced (Purity ≥ 0.99): {best_balanced['threshold']:.3f}")
        print(f"   → Purity: {best_balanced['purity']:.4f}, Efficiency: {best_balanced['efficiency']:.4f}, F1: {best_balanced['f1']:.4f}")
        print(f"   → TP: {best_balanced['tp']}, FP: {best_balanced['fp']}, FN: {best_balanced['fn']}")
        
        print(f"\n4. Very Strict (Purity ≥ 0.995): {best_strict['threshold']:.3f}")
        print(f"   → Purity: {best_strict['purity']:.4f}, Efficiency: {best_strict['efficiency']:.4f}, F1: {best_strict['f1']:.4f}")
        print(f"   → TP: {best_strict['tp']}, FP: {best_strict['fp']}, FN: {best_strict['fn']}")
        
        return best_f1, best_purity, best_balanced, best_strict
    
    return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)

def load_model(model_path, input_dim, edge_feature_dim, device):
    """Load the trained model from checkpoint"""
    model = GNN_TrackLinkingNet(
        input_dim=input_dim,
        edge_feature_dim=edge_feature_dim
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        train_loss = checkpoint.get('train_loss', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
    else:
        # If the checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
        epoch = 'Unknown'
        train_loss = 'Unknown'
        val_loss = 'Unknown'
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Epoch: {epoch}")
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")
    
    return model

def test_model(model, test_loader, device, edge_features=True):
    """Test the model and return predictions and ground truth"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
                
            batch = batch.to(device)
            x = batch.x.float()
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr.float() if edge_features else None
            batch_vector = batch.batch
            
            # CORRECTED: Your model only expects (x, edge_index, edge_attr, device)
            # Remove batch_vector from the forward call
            out = model(x, edge_index, edge_attr, device=device)
            
            # Your model already applies sigmoid in the final layer
            predictions = out.cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    return np.array(all_predictions), np.array(all_labels)

def comprehensive_debug(model, test_loader, device):
    """Run comprehensive debugging to identify the issue"""
    print("=== COMPREHENSIVE DEBUGGING ===")
    
    model.eval()
    all_raw_outputs = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                continue
                
            batch = batch.to(device)
            
            # Get model outputs
            raw_output = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), device=device)
            
            # Your model already has sigmoid in the final layer, so no need to apply again
            probabilities = raw_output
            
            all_raw_outputs.extend(raw_output.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            # Only check first few batches
            if i >= 2:
                break
    
    all_raw_outputs = np.array(all_raw_outputs)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    print(f"Model outputs range: [{all_probabilities.min():.6f}, {all_probabilities.max():.6f}]")
    print(f"Labels - True: {np.sum(all_labels > 0.5)}, False: {np.sum(all_labels <= 0.5)}")
    
    # Check distribution
    true_mask = all_labels > 0.5
    false_mask = ~true_mask
    
    if np.any(true_mask):
        print(f"True predictions - Min: {all_probabilities[true_mask].min():.3f}, "
              f"Max: {all_probabilities[true_mask].max():.3f}, "
              f"Mean: {all_probabilities[true_mask].mean():.3f}")
    
    if np.any(false_mask):
        print(f"False predictions - Min: {all_probabilities[false_mask].min():.3f}, "
              f"Max: {all_probabilities[false_mask].max():.3f}, "
              f"Mean: {all_probabilities[false_mask].mean():.3f}")
    
    return all_probabilities, all_labels

def plot_score_distribution(predictions, labels, save_path=None, threshold_step=0.05):
    """Plot score distribution for true and false labels with different colors"""
    
    # Convert to numpy arrays if they're tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Separate scores for true (positive) and false (negative) labels
    true_scores = predictions[labels > 0.5]
    false_scores = predictions[labels <= 0.5]
    
    print(f"Total samples: {len(predictions)}")
    print(f"True positives: {len(true_scores)}")
    print(f"False negatives: {len(false_scores)}")
    
    if len(true_scores) > 0:
        print(f"True score range: [{true_scores.min():.3f}, {true_scores.max():.3f}], Mean: {true_scores.mean():.3f}")
    if len(false_scores) > 0:
        print(f"False score range: [{false_scores.min():.3f}, {false_scores.max():.3f}], Mean: {false_scores.mean():.3f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    bins = np.linspace(0, 1, 51)
    
    if len(true_scores) > 0:
        plt.hist(true_scores, bins=bins, alpha=0.7, color='green', label='True Links', 
                 density=True, edgecolor='black', linewidth=0.5)
    
    if len(false_scores) > 0:
        plt.hist(false_scores, bins=bins, alpha=0.7, color='red', label='False Links', 
                 density=True, edgecolor='black', linewidth=0.5)
    
    # Calculate and plot optimal threshold
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    f1_scores = []
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        f1 = f1_score(labels, pred_binary, zero_division=0)
        f1_scores.append(f1)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    # Add vertical line for best threshold
    plt.axvline(x=best_threshold, color='blue', linestyle='--', 
                linewidth=2, label=f'Optimal Threshold: {best_threshold:.3f}\n(F1-score: {best_f1:.3f})')
    
    # Calculate metrics at optimal threshold
    pred_binary = (predictions > best_threshold).astype(int)
    accuracy = balanced_accuracy_score(labels, pred_binary)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Add text box with metrics
    metrics_text = (f'Metrics at threshold {best_threshold:.3f}:\n'
                   f'Balanced Accuracy: {accuracy:.3f}\n'
                   f'True Positive Rate: {tpr:.3f}\n'
                   f'True Negative Rate: {tnr:.3f}\n'
                   f'F1-score: {best_f1:.3f}\n'
                   f'TP: {tp}, FP: {fp}\n'
                   f'TN: {tn}, FN: {fn}')
    
    plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontfamily='monospace')
    
    # Plot styling
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution: True vs False Links', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return best_threshold, best_f1

def main(model_folder,output_dir,hist_folder,data_folder):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths - update these according to your setup
    #hist_folder = "/cms/data/store/user/moanwar/Link_single_interfaceDisk"
    #data_folder = "/cms/data/store/user/moanwar/prossesed_had_interfaceDesk"
    #model_folder = "/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/inference_interfaceLayer/"
    
    # Load the latest model or specify a specific model
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_folder}")
    
    # Sort by modification time and get the latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)), reverse=True)
    model_path = os.path.join(model_folder, model_files[0])
    print(f"Using model: {model_path}")
    
    # Load dataset to get dimensions
    full_dataset = ClusterDataset(data_folder, hist_folder)
    print(f"Dataset loaded with {len(full_dataset)} graphs")
    
    # Get model dimensions from dataset
    input_dim = full_dataset.model_feature_keys.shape[0]
    sample_data = full_dataset.get(0)
    edge_feature_dim = sample_data.edge_attr.shape[1] if hasattr(sample_data, 'edge_attr') else 0
    
    print(f"Input dimension: {input_dim}")
    print(f"Edge feature dimension: {edge_feature_dim}")
    
    # Create test dataset and loader
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=collate_skip_none)
    
    # Load model
    model = load_model(model_path, input_dim, edge_feature_dim, device)
    
    # Run debugging first
    print("\nRunning debugging...")
    probabilities, labels = comprehensive_debug(model, test_loader, device)
    
    # Test model
    print("\nTesting model...")
    predictions, labels = test_model(model, test_loader, device, edge_features=True)
    
    # Plot score distribution
    #output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = f"{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"
    save_path = os.path.join(output_dir, f"score_distribution_{date_str}.png")
    
    best_threshold, best_f1 = plot_score_distribution(predictions, labels, save_path=save_path)
    
    print(f"\n=== RESULTS ===")
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Plot saved to: {save_path}")

    print("\n" + "="*80)
    print("ANALYZING HIGH-THRESHOLD CUTS (\u22650.95)")
    print("="*80)

    # Run the analysis                                                                                        
    best_f1, best_purity, best_balanced, best_strict = find_optimal_high_cut(predictions, labels)

    # Also run the detailed threshold analysis                                                                
    print("\n" + "="*80)
    print("DETAILED THRESHOLD ANALYSIS (0.95 - 1.00)")
    print("="*80)
    best_threshold_high, best_f1_high = analyze_high_threshold_cuts(predictions, labels, min_threshold=0.95, step=0.005)
    plot_high_threshold_region(predictions, labels, f"{output_dir}/high_threshold_region.png")

    final_threshold = 0.950	
    high_confidence_predictions = apply_high_threshold_cut(predictions, final_threshold)

    print(f"Applied threshold: {final_threshold}")
    print(f"High-confidence predictions: {np.sum(high_confidence_predictions)}")
    print(f"Expected purity: >99.9%")
    print(f"Expected false positives: ~10 out of {np.sum(high_confidence_predictions)}")
    plot_final_recommendation(predictions, labels, recommended_threshold=0.95)

if __name__ == "__main__":
    model_folder ="/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/inference_interfaceLayer/"
    output_dir   ="./inference_interfaceLayer"
    hist_folder = "/cms/data/store/user/moanwar/Link_single_interfaceDisk"
    data_folder = "/cms/data/store/user/moanwar/prossesed_had_interfaceDesk"

    #model_folder ="/home/moanwar/linking/CMSSW_15_1_0_pre5/src/graph_nn/inference_firstLayer/"
    #output_dir   ="./inference_firstLayer"
    #hist_folder = "/cms/data/store/user/moanwar/Link_single_firstDisk"
    #data_folder = "/cms/data/store/user/moanwar/prossesed_had_firstDesk/"

    main(model_folder,output_dir,hist_folder,data_folder)
