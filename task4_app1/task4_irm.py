
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from itertools import cycle

# ==============================================================================
# Configuration & Hyperparameters
# ==============================================================================
class Config:
    # Data Paths
    DATA_DIR = r"c:\Users\gupta\Desktop\cvtask\mydatav2\cmnist_v2"
    TRAIN_FILE = "train_data_rg95z.npz"
    
    # Multiple Test Datasets for Comprehensive Evaluation
    TEST_DATASETS = {
        'rg95z': 'test_data_rg95z.npz',   # Biased (Easy)
        'gr95z': 'test_data_gr95z.npz',   # Inverted (Hard)
        'gr95e': 'test_data_gr95e.npz',
        'gr95m': 'test_data_gr95m.npz',
        'gr95h': 'test_data_gr95h.npz',
        'gr95vh': 'test_data_gr95vh.npz',
        'bw95z': 'test_data_bw95z.npz',   # Grayscale
        'bw100z': 'test_data_bw100z.npz'
    }
    
    # Training Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 2
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # IRM Specifics
    IRM_PENALTY_WEIGHT = 1.0
    IRM_ANNEAL_EPOCHS = 1  # Epochs before penalty kicks in
    
    # System
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

# ==============================================================================
# Model Architecture
# ==============================================================================
class CNN3Layer(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3Layer, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==============================================================================
# Utils: IRM Penalty & Data Loading
# ==============================================================================
def irm_penalty(logits, y):
    """
    Computes the IRM v1 penalty: gradient norm of the loss w.r.t a fixed scalar 1.0.
    This effectively asks: "If I multiplied the classifier output by a scalar 'w',
    would the optimal 'w' result in 0 gradient at w=1.0 across all environments?"
    """
    scale = torch.tensor(1.).to(logits.device).requires_grad_()
    loss = nn.CrossEntropyLoss()(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def get_dominant_color(img_tensor):
    """Returns dominant channel index (0=R, 1=G, 2=B) for a single image tensor."""
    means = torch.mean(img_tensor, dim=(1, 2))
    return torch.argmax(means).item()

def load_data(config):
    """Loads data and splits training set into two IRM environments."""
    print(f"\n[Data] Loading from {config.DATA_DIR}...")
    
    train_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    # Load Train (Biased)
    train_data = np.load(train_path)
    X_train = torch.tensor(train_data['images'].astype('float32') / 255.0).permute(0, 3, 1, 2)
    y_train = torch.tensor(train_data['labels']).long()

    # 1. Identify Spurious Correlation (Color Bias)
    print("[Data] Analyzing bias...")
    digit_bias_color = {}
    for d in range(10):
        # Sample subset to determine majority color
        indices = (y_train == d).nonzero(as_tuple=True)[0][:100]
        colors = [get_dominant_color(X_train[i]) for i in indices]
        majority = max(set(colors), key=colors.count)
        digit_bias_color[d] = majority
    
    # 2. Split into Environments (Aligned vs Conflict)
    # Calculate dominant color for entire batch efficiently
    img_means = torch.mean(X_train, dim=(2, 3)) # (N, 3)
    img_colors = torch.argmax(img_means, dim=1) # (N,)
    
    # Expected color based on label
    expected_colors = torch.tensor([digit_bias_color[y.item()] for y in y_train], device=X_train.device)
    
    # Mask: True if image follows bias (Environment 1), False if conflict (Environment 2)
    aligned_mask = (img_colors == expected_colors.cpu())
    
    env1_idx = torch.nonzero(aligned_mask, as_tuple=True)[0]
    env2_idx = torch.nonzero(~aligned_mask, as_tuple=True)[0]
    
    print(f"  Env 1 (Aligned/Biased): {len(env1_idx)} samples")
    print(f"  Env 2 (Conflict/OOD):   {len(env2_idx)} samples")

    # Create Datasets
    ds_env1 = TensorDataset(X_train[env1_idx], y_train[env1_idx])
    ds_env2 = TensorDataset(X_train[env2_idx], y_train[env2_idx])
    
    return ds_env1, ds_env2

def get_test_loaders(config):
    loaders = {}
    print("\n[Data] Loading Test Sets...")
    for name, filename in config.TEST_DATASETS.items():
        path = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(path):
            data = np.load(path)
            X = torch.tensor(data['images'].astype('float32') / 255.0).permute(0, 3, 1, 2)
            y = torch.tensor(data['labels']).long()
            ds = TensorDataset(X, y)
            loaders[name] = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)
        else:
            print(f"  [Warn] {filename} not found.")
    return loaders

# ==============================================================================
# Visualization Logic (User Provided + Adapted)
# ==============================================================================
def evaluate_and_plot(model, test_loaders, train_acc_history, val_acc_history, train_loss_history, val_loss_history, device):
    print("\n[Evaluate] Starting comprehensive evaluation...")
    criterion = nn.CrossEntropyLoss()
    
    test_results = {}
    all_predictions = {}
    all_targets = {}

    for name, loader in test_loaders.items():
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        avg_test_loss = test_loss / len(loader)
        test_accuracy = test_correct / test_total
        
        test_results[name] = {
            'loss': avg_test_loss,
            'accuracy': test_accuracy
        }
        all_predictions[name] = np.array(predictions)
        all_targets[name] = np.array(targets)
        
        print(f"{name:<10} | Loss: {avg_test_loss:.4f} | Acc: {test_accuracy*100:.2f}%")

    # Visualization 1: Accuracy comparison across all datasets
    plt.figure(figsize=(12, 6))
    dataset_names = list(test_results.keys())
    accuracies = [test_results[name]['accuracy'] * 100 for name in dataset_names]

    plt.bar(dataset_names, accuracies, color='steelblue', edgecolor='black')
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy on All Test Datasets', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    for i, (name, acc) in enumerate(zip(dataset_names, accuracies)):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved accuracy_comparison.png")
    # plt.show() # Skipped for non-interactive env

    # Visualization 2: Training and Validation Accuracy vs Epoch
    if len(train_acc_history) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        epochs_range = range(1, len(train_acc_history) + 1)
        plt.plot(epochs_range, [acc * 100 for acc in train_acc_history], 'b-o', label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, [acc * 100 for acc in val_acc_history], 'r-s', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_loss_history, 'b-o', label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_loss_history, 'r-s', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Saved training_curves.png")
        # plt.show()

    # Visualization 3: Confusion Matrix for each dataset
    num_datasets = len(test_results)
    cols = min(3, num_datasets)
    rows = (num_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_datasets > 1 else [axes]

    for idx, name in enumerate(dataset_names):
        cm = confusion_matrix(all_targets[name], all_predictions[name])
        
        ax = axes[idx] if num_datasets > 1 else axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar_kws={'label': 'Count'}, square=True)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {name}\nAccuracy: {test_results[name]["accuracy"]*100:.2f}%', 
                     fontsize=12, fontweight='bold')

    # Hide extra subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("Saved confusion_matrices.png")
    # plt.show()

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for name in dataset_names:
        print(f"{name:15s}: {test_results[name]['accuracy']*100:6.2f}%")
    print("="*50)

# ==============================================================================
# Training Loop
# ==============================================================================
def train(config):
    torch.manual_seed(config.SEED)
    
    # Load Data
    ds_env1, ds_env2 = load_data(config)
    test_loaders = get_test_loaders(config)
    
    # Use gr95z (Inverse) as validation set for curves if available, else gr95z_hard or similar
    # Defaulting to gr95z for OOD validation
    val_loader = test_loaders.get('gr95z')
    
    # Loaders
    loader1 = DataLoader(ds_env1, batch_size=config.BATCH_SIZE, shuffle=True)
    loader2 = DataLoader(ds_env2, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Env 2 is smaller (5%), so we cycle it to match Env 1 iterations
    iter2 = cycle(loader2)
    
    # Model & Optim
    model = CNN3Layer().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    print(f"\n[Train] Starting IRM Training on {config.DEVICE}...")
    print(f"  Penalty: {config.IRM_PENALTY_WEIGHT} (Annealed for {config.IRM_ANNEAL_EPOCHS} epochs)")
    
    # Metrics history
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x1, y1 in loader1:
            x2, y2 = next(iter2)
            
            x1, y1 = x1.to(config.DEVICE), y1.to(config.DEVICE)
            x2, y2 = x2.to(config.DEVICE), y2.to(config.DEVICE)
            
            # Forward
            logits1 = model(x1)
            logits2 = model(x2)
            
            # 1. Standard ERM Risk (Cross Entropy)
            nll1 = nn.CrossEntropyLoss()(logits1, y1)
            nll2 = nn.CrossEntropyLoss()(logits2, y2)
            risk = (nll1 + nll2) / 2
            
            # 2. Invariance Penalty
            penalty = torch.tensor(0.).to(config.DEVICE)
            if epoch >= config.IRM_ANNEAL_EPOCHS:
                p1 = irm_penalty(logits1, y1)
                p2 = irm_penalty(logits2, y2)
                penalty = (p1 + p2) / 2
            
            # Total Loss
            loss = risk + (config.IRM_PENALTY_WEIGHT if epoch >= config.IRM_ANNEAL_EPOCHS else 1.0) * penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats for Training Curve
            total_loss += loss.item() # This includes penalty! For curves, we might want just risk, but loss is fine.
            _, preds = torch.max(logits1, 1) # Acc on env1
            correct += (preds == y1).sum().item()
            total += y1.size(0)
            
        # End of Epoch Stats
        epoch_train_loss = total_loss / len(loader1)
        epoch_train_acc = correct / total
        
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        
        # Validation Stats
        val_loss = 0.0
        val_acc = 0.0
        if val_loader:
            model.eval()
            v_loss = 0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(config.DEVICE), vy.to(config.DEVICE)
                    vout = model(vx)
                    v_loss += nn.CrossEntropyLoss()(vout, vy).item()
                    _, vpred = torch.max(vout, 1)
                    v_correct += (vpred == vy).sum().item()
                    v_total += vy.size(0)
            val_loss = v_loss / len(val_loader)
            val_acc = v_correct / v_total
            model.train()
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | Val Acc (OOD): {val_acc*100:.1f}%")

    # Final Save
    save_path = "task4_irm_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Done] Model saved to {save_path}")
    
    # User Visualizations
    evaluate_and_plot(model, test_loaders, train_acc_history, val_acc_history, train_loss_history, val_loss_history, config.DEVICE)

if __name__ == "__main__":
    train(Config)
