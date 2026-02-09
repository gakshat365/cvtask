


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import gc
import gc
from collections import defaultdict
from collections import defaultdict


TRAIN_DATA_PATH = '/kaggle/input/cmnistneo1/train_data_rg95z.npz'  # Data with original color mapping
TEST_DATA_PATH = '/kaggle/input/cmnistneo1/test_data_gr95z.npz'  # Data with reversed/different color mapping
MODEL_PATH = '/kaggle/input/task1app3models/pytorch/default/2/task1approach3sc1_modelv1.pth'


# Hyperparameters
TOPK_K = 32  # Number of active features in TopK SAE
SAE_HIDDEN_DIM = 512
SAE_EPOCHS = 20
SAE_BATCH_SIZE = 64  # Reduced from 256 to save GPU memory
SAE_LEARNING_RATE = 0.001

# Memory-saving: limit samples for analysis steps
MEMORY_LIMIT_SAMPLES = 500  # Use subset for heavy analysis operations

# Model architecture (from your biased model)
NUM_CLASSES = 10
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
CONV3_CHANNELS = 64
FC1_UNITS = 128
DROPOUT_RATE = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



class CNN3Layer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN3Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, CONV1_CHANNELS, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(CONV3_CHANNELS * 3 * 3, FC1_UNITS)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(FC1_UNITS, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_fc1_activations(self, x):
        """Get FC1 activations (the hidden state we'll analyze)"""
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x



class TopKSparseAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, k):
        super(TopKSparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        # Learned encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Normalize decoder weights (helps with feature interpretability)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x):
        """Encode with TopK sparsity constraint"""
        pre_act = self.encoder(x)
        
        # TopK: only keep the top K activations, set rest to 0
        topk_values, topk_indices = torch.topk(pre_act, self.k, dim=-1)
        
        # Create sparse activation
        sparse_act = torch.zeros_like(pre_act)
        sparse_act.scatter_(-1, topk_indices, F.relu(topk_values))
        
        return sparse_act, topk_indices
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, indices = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, indices



class ContrastiveFeatureLearner(nn.Module):

    def __init__(self, input_dim, feature_dim=64):
        super(ContrastiveFeatureLearner, self).__init__()
        self.feature_dim = feature_dim
        
        # Shape-invariant feature extractor
        self.shape_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        
        # Color-specific feature extractor
        self.color_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
    
    def forward(self, x):
        shape_features = self.shape_encoder(x)
        color_features = self.color_encoder(x)
        return shape_features, color_features


def contrastive_loss(shape_features, color_features, labels, margin=1.0):
    """
    Custom contrastive loss:
    - Shape features of same class should be similar
    - Color features should capture the remaining variance
    """
    batch_size = shape_features.size(0)
    
    # Normalize features
    shape_norm = F.normalize(shape_features, dim=1)
    color_norm = F.normalize(color_features, dim=1)
    
    # Shape similarity matrix
    shape_sim = torch.mm(shape_norm, shape_norm.t())
    
    # Create label mask (1 if same class, 0 otherwise)
    labels = labels.view(-1, 1)
    label_mask = (labels == labels.t()).float()
    
    # Contrastive loss for shape features
    # Pull same-class together, push different-class apart
    pos_loss = (1 - shape_sim) * label_mask  # Same class should be similar
    neg_loss = F.relu(shape_sim - margin) * (1 - label_mask)  # Different class apart
    
    shape_loss = (pos_loss.sum() + neg_loss.sum()) / (batch_size * batch_size)
    
    # Reconstruction: shape + color should reconstruct original
    # (This is optional but helps training)
    
    return shape_loss



class ConceptSteeringVectors:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.color_vectors = {}  # Per-class color steering vectors
        self.shape_vectors = {}  # Per-class shape steering vectors
        self.mean_activations = {}
    
    def compute_steering_vectors(self, env1_images, env1_labels, env2_images, env2_labels):
        """
        Compute steering vectors from two different color environments.
        """
        self.model.eval()
        
        # Get activations for both environments
        with torch.no_grad():
            env1_acts = self._get_activations_batched(env1_images)
            env2_acts = self._get_activations_batched(env2_images)
        
        # FIXED: Compute per-class means HERE (inside no_grad, after acts computed)
        self.color_vectors = {}
        self.mean_activations = {}
        
        for class_idx in range(10):
            env1_mask = env1_labels == class_idx
            env2_mask = env2_labels == class_idx
            
            if env1_mask.sum() > 0 and env2_mask.sum() > 0:
                env1_mean = env1_acts[env1_mask].mean(axis=0)
                env2_mean = env2_acts[env2_mask].mean(axis=0)
                
                # Color vector: captures color changes for same shape
                self.color_vectors[class_idx] = env1_mean - env2_mean
                
                self.mean_activations[f'env1_class{class_idx}'] = env1_mean
                self.mean_activations[f'env2_class{class_idx}'] = env2_mean
        
        # Global color vector: average across classes
        if self.color_vectors:
            all_color_vecs = np.stack(list(self.color_vectors.values()))
            self.global_color_vector = all_color_vecs.mean(axis=0)
            self.global_color_vector = self.global_color_vector / np.linalg.norm(self.global_color_vector)
        else:
            # Fallback if insufficient data
            self.global_color_vector = np.zeros(FC1_UNITS)
        
        print(f"Computed color steering vectors for {len(self.color_vectors)} classes")
        print(f"Global color vector norm: {np.linalg.norm(self.global_color_vector):.4f}")
        
        return self.global_color_vector

    
    def _get_activations_batched(self, images, batch_size=64):
        """Get activations in batches to avoid OOM"""
        all_acts = []
        for i in range(0, len(images), batch_size):
            batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(self.device)
            acts = self.model.get_fc1_activations(batch).cpu().numpy()
            all_acts.append(acts)
            del batch
            torch.cuda.empty_cache()
        return np.concatenate(all_acts, axis=0)
        
        # Compute per-class mean activations for each environment
        for class_idx in range(10):
            env1_mask = env1_labels == class_idx
            env2_mask = env2_labels == class_idx
            
            if env1_mask.sum() > 0 and env2_mask.sum() > 0:
                env1_mean = env1_acts[env1_mask].mean(axis=0)
                env2_mean = env2_acts[env2_mask].mean(axis=0)
                
                # COLOR VECTOR: difference between same class in different color environments
                # This captures what changes when color changes but shape stays same
                self.color_vectors[class_idx] = env1_mean - env2_mean
                
                # Store means for later use
                self.mean_activations[f'env1_class{class_idx}'] = env1_mean
                self.mean_activations[f'env2_class{class_idx}'] = env2_mean
        
        # GLOBAL COLOR VECTOR: average across all classes
        all_color_vecs = np.stack(list(self.color_vectors.values()))
        self.global_color_vector = all_color_vecs.mean(axis=0)
        
        # Normalize
        self.global_color_vector = self.global_color_vector / np.linalg.norm(self.global_color_vector)
        
        print(f"Computed color steering vectors for {len(self.color_vectors)} classes")
        print(f"Global color vector norm: {np.linalg.norm(self.global_color_vector):.4f}")
        
        return self.global_color_vector
    
    def steer_activations(self, activations, direction='remove_color', strength=1.0):

        color_vec = torch.FloatTensor(self.global_color_vector).to(self.device)
        
        # Project activations onto color direction
        color_projection = torch.sum(activations * color_vec, dim=-1, keepdim=True)
        
        if direction == 'remove_color':
            # Remove color component
            steered = activations - strength * color_projection * color_vec
        else:
            # Add color component
            steered = activations + strength * color_projection * color_vec
        
        return steered
    
    def intervention_experiment(self, images, labels, strengths=[0.0, 0.5, 1.0, 2.0]):
        """Test how removing color direction affects predictions"""
        self.model.eval()
        
        results = []
        
        with torch.no_grad():
            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
            original_acts = self.model.get_fc1_activations(images_tensor)
            
            for strength in strengths:
                # Steer activations
                steered_acts = self.steer_activations(original_acts, 'remove_color', strength)
                
                # Continue forward pass
                output = self.model.fc2(self.model.dropout(steered_acts))
                predictions = torch.argmax(output, dim=1).cpu().numpy()
                
                accuracy = (predictions == labels).mean() * 100
                
                results.append({
                    'strength': strength,
                    'accuracy': accuracy,
                    'predictions': predictions
                })
                
                print(f"Steering strength {strength:.1f}: Accuracy = {accuracy:.2f}%")
        
        return results



class CausalNeuronAnalyzer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.neuron_effects = {}
    
    def ablate_neuron(self, activations, labels, neuron_idx, ablation_type='zero'):
        """
        Test model accuracy when a specific FC1 neuron is ablated.
        Uses pre-computed activations for speed.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Use pre-computed activations
            acts = activations
            
            # Ablate the specific neuron
            ablated_acts = acts.clone()
            if ablation_type == 'zero':
                ablated_acts[:, neuron_idx] = 0
            else:  # mean
                ablated_acts[:, neuron_idx] = acts[:, neuron_idx].mean()
            
            # Continue forward pass (only FC2 needed)
            output = self.model.fc2(self.model.dropout(ablated_acts))
            predictions = torch.argmax(output, dim=1).cpu().numpy()
            
            accuracy = (predictions == labels).mean() * 100
        
        return accuracy
    
    def find_causal_neurons(self, env1_images, env1_labels, env2_images, env2_labels, 
                            num_neurons=128):

        print("\nAnalyzing causal effect of each neuron...")
        
        # Baseline accuracies
        self.model.eval()
        
        with torch.no_grad():
            # Pre-compute activations ONCE
            print("  Pre-computing activations for speed...")
            env1_acts = self._get_activations_batched(env1_images)
            env2_acts = self._get_activations_batched(env2_images)
            
            # Get baseline accuracy from activations
            def get_acc_from_acts(acts, labels):
                out = self.model.fc2(self.model.dropout(acts))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                return (preds == labels).mean() * 100

            env1_baseline = get_acc_from_acts(env1_acts, env1_labels)
            env2_baseline = get_acc_from_acts(env2_acts, env2_labels)
        
            print(f"Baseline accuracies: Env1={env1_baseline:.2f}%, Env2={env2_baseline:.2f}%")
            
            # Test each neuron
            self.neuron_effects = {}
            color_neurons = []
            shape_neurons = []
            
            for neuron_idx in range(num_neurons):
                # Pass activations instead of images
                env1_ablated = self.ablate_neuron(env1_acts, env1_labels, neuron_idx)
                env2_ablated = self.ablate_neuron(env2_acts, env2_labels, neuron_idx)
                
                env1_effect = env1_baseline - env1_ablated
                env2_effect = env2_baseline - env2_ablated
                
                self.neuron_effects[neuron_idx] = {
                    'env1_effect': env1_effect,
                    'env2_effect': env2_effect,
                    'asymmetry': env1_effect - env2_effect
                }
                
                if env1_effect > 2 and env2_effect < 0:
                    color_neurons.append(neuron_idx)
                elif abs(env1_effect - env2_effect) < 1 and (env1_effect > 0.5 or env2_effect > 0.5):
                    shape_neurons.append(neuron_idx)
                
                if (neuron_idx + 1) % 32 == 0:
                    print(f"  Analyzed {neuron_idx + 1}/{num_neurons} neurons...")
            
            # Sort and print results
            sorted_neurons = sorted(self.neuron_effects.items(), 
                                key=lambda x: x[1]['asymmetry'], reverse=True)
            
            print(f"\nTop 10 COLOR-SPECIFIC neurons:")
            for neuron_idx, effects in sorted_neurons[:10]:
                print(f"  Neuron {neuron_idx}: Env1 effect={effects['env1_effect']:.2f}%, "
                      f"Asymmetry={effects['asymmetry']:.2f}")
            
            return sorted_neurons[:10], sorted_neurons[-10:]

    def _get_activations_batched(self, images, batch_size=64):
        all_acts = []
        for i in range(0, len(images), batch_size):
            batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(self.device)
            acts = self.model.get_fc1_activations(batch)
            all_acts.append(acts)
            del batch
        return torch.cat(all_acts, dim=0)

    def _get_accuracy_batched(self, images, labels, batch_size=64):
        """Get accuracy in batches to avoid OOM"""
        all_preds = []
        for i in range(0, len(images), batch_size):
            batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(self.device)
            preds = torch.argmax(self.model(batch), dim=1).cpu().numpy()
            all_preds.append(preds)
            del batch
            torch.cuda.empty_cache()
        return (np.concatenate(all_preds) == labels).mean() * 100
        




def cluster_features_by_environment(model, sae, env1_data, env2_data, n_clusters=4):

    model.eval()
    sae.eval()
    
    env1_images, env1_labels = env1_data
    env2_images, env2_labels = env2_data
    
    with torch.no_grad():
        # Get activations in batches to save memory
        def get_acts_batched(images, batch_size=64):
            all_acts = []
            for i in range(0, len(images), batch_size):
                batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
                acts = model.get_fc1_activations(batch)
                all_acts.append(acts.cpu())
                del batch
                torch.cuda.empty_cache()
            return torch.cat(all_acts, dim=0).to(device)
        
        env1_acts = get_acts_batched(env1_images)
        env2_acts = get_acts_batched(env2_images)
        
        # Get SAE encodings
        env1_z, _ = sae.encode(env1_acts)
        env2_z, _ = sae.encode(env2_acts)
        
        env1_z = env1_z.cpu().numpy()
        env2_z = env2_z.cpu().numpy()
    
    # Compute per-class mean activations for each SAE feature
    n_features = env1_z.shape[1]
    env1_class_means = np.zeros((10, n_features))
    env2_class_means = np.zeros((10, n_features))
    
    for c in range(10):
        env1_mask = env1_labels == c
        env2_mask = env2_labels == c
        
        if env1_mask.sum() > 0:
            env1_class_means[c] = env1_z[env1_mask].mean(axis=0)
        if env2_mask.sum() > 0:
            env2_class_means[c] = env2_z[env2_mask].mean(axis=0)
    
    # Feature characteristics: how each feature differs across environments
    feature_characteristics = np.zeros((n_features, 3))
    
    for feat_idx in range(n_features):
        # Mean activation difference across classes
        mean_diff = np.abs(env1_class_means[:, feat_idx] - env2_class_means[:, feat_idx]).mean()
        
        # Variance within environment (selectivity)
        env1_var = env1_class_means[:, feat_idx].var()
        env2_var = env2_class_means[:, feat_idx].var()
        
        feature_characteristics[feat_idx] = [mean_diff, env1_var, env2_var]
    
    # Cluster features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_characteristics)
    
    # Analyze clusters
    print("\n=== Feature Clustering by Environment Behavior ===")
    cluster_info = []
    
    for cluster_idx in range(n_clusters):
        cluster_mask = clusters == cluster_idx
        cluster_features = np.where(cluster_mask)[0]
        
        mean_env_diff = feature_characteristics[cluster_mask, 0].mean()
        mean_selectivity = (feature_characteristics[cluster_mask, 1] + 
                           feature_characteristics[cluster_mask, 2]).mean() / 2
        
        cluster_info.append({
            'cluster': cluster_idx,
            'n_features': len(cluster_features),
            'mean_env_diff': mean_env_diff,
            'mean_selectivity': mean_selectivity,
            'features': cluster_features[:10]  # First 10
        })
        
        cluster_type = "COLOR" if mean_env_diff > np.median(feature_characteristics[:, 0]) else "SHAPE"
        print(f"\nCluster {cluster_idx} ({cluster_type}):")
        print(f"  {len(cluster_features)} features")
        print(f"  Mean env difference: {mean_env_diff:.4f}")
        print(f"  Mean selectivity: {mean_selectivity:.4f}")
        print(f"  Example features: {cluster_features[:10]}")
    
    return clusters, cluster_info, feature_characteristics




# ==========================================
# PHASE 1: Feature Identification (Traitors & Heroes)
# ==========================================

class FeatureClassifier:
    def __init__(self, model, sae, device):
        self.model = model
        self.sae = sae
        self.device = device
        self.traitors = []
        self.heroes = []
        self.sensitivity_scores = None

    def classify_features(self, images, batch_size=64):
        """
        Classify SAE features based on sensitivity to color.
        Generates B&W counterparts in-memory to ensure perfect pairing.
        """
        print("\nRunning Feature Sensitivity Analysis...")
        self.model.eval()
        self.sae.eval()

        # 1. Prepare Data Pairs (Color vs BW)
        # Convert RGB images to Grayscale (simulate BW inputs)
        # images shape: (N, 28, 28, 3) -> Mean over channel -> (N, 28, 28, 1) -> Repeat -> (N, 28, 28, 3)
        images_bw = images.mean(axis=3, keepdims=True).repeat(3, axis=3)
        
        # 2. Extract SAE Latents for both
        def get_latents(img_data):
            z_list = []
            with torch.no_grad():
                for i in range(0, len(img_data), batch_size):
                    batch = torch.FloatTensor(img_data[i:i+batch_size]).permute(0, 3, 1, 2).to(self.device)
                    acts = self.model.get_fc1_activations(batch)
                    _, z, _ = self.sae(acts)
                    z_list.append(z.cpu().numpy())
            return np.concatenate(z_list, axis=0)

        print("  Extracting latents for Colored images...")
        z_color = get_latents(images)
        print("  Extracting latents for B&W counterparts...")
        z_bw = get_latents(images_bw)

        # 3. Calculate Sensitivity Index
        # Metric: Mean Activation (Color) / Mean Activation (BW)
        # Add epsilon to denominators to match User logic 1.2
        epsilon = 1e-4
        mean_act_color = z_color.mean(axis=0)
        mean_act_bw = z_bw.mean(axis=0)
        
        self.sensitivity_scores = (mean_act_color + epsilon) / (mean_act_bw + epsilon)
        
        # 4. Classification
        self.traitors = []
        self.heroes = []
        
        for idx, score in enumerate(self.sensitivity_scores):
            # Traitors: High index (> 2.0 implies Color signals drop significantly in BW)
            # We use a threshold of 2.0 to be safe (User suggested > 1, but > 2 is stronger evidence)
            if score > 2.0 and mean_act_color[idx] > 0.01: # Check it's not a dead neuron
                self.traitors.append((idx, score))
            
            # Heroes: Index near 1 (0.8 to 1.2)
            elif 0.8 <= score <= 1.2 and mean_act_color[idx] > 0.01:
                self.heroes.append((idx, score))

        # Sort by score
        self.traitors.sort(key=lambda x: x[1], reverse=True)
        self.heroes.sort(key=lambda x: abs(1-x[1]))

        print(f"  Found {len(self.traitors)} Traitors (Color-Obsessed)")
        print(f"  Found {len(self.heroes)} Heroes (Shape-Invariant)")
        
        if self.traitors:
            print(f"  Top Traitor: Feature {self.traitors[0][0]} (Score {self.traitors[0][1]:.2f})")
        if self.heroes:
            print(f"  Top Hero: Feature {self.heroes[0][0]} (Score {self.heroes[0][1]:.2f})")
            
        return self.traitors, self.heroes

    def plot_sensitivity_analysis(self, save_path='feature_sensitivity.png'):
        if self.sensitivity_scores is None:
            return
        
        scores = self.sensitivity_scores
        n_feats = len(scores)
        
        # Reshape for grid (Target 16x32 for 512 features, else square)
        if n_feats == 512:
            grid = scores.reshape(16, 32)
        else:
            side = int(np.ceil(np.sqrt(n_feats)))
            grid = np.zeros(side*side)
            grid[:n_feats] = scores
            grid = grid.reshape(side, side)
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Pixel Grid (Red=Traitor, Blue=Hero)
        # Using coolwarm: Low=Blue, High=Red
        im = axes[0].imshow(grid, cmap='coolwarm', vmin=0, vmax=3.0, aspect='auto')
        axes[0].set_title(f"Feature Sensitivity Map ({n_feats} Features)\nRed = Traitors (>2.0) | Blue = Heroes (~1.0)")
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], label='Sensitivity Index')
        
        # 2. Histogram with Thresholds
        # Count
        n_traitors = (scores > 2.0).sum()
        # Heroes are approx 1.0 (e.g. 0.8 to 1.2)
        n_heroes = ((scores > 0.8) & (scores < 1.2)).sum()
        
        axes[1].hist(scores, bins=50, color='gray', alpha=0.7, log=True)
        
        # Thresholds
        axes[1].axvline(2.0, color='red', linestyle='--', linewidth=2)
        axes[1].axvline(1.0, color='blue', linestyle='--', linewidth=2)
        
        # Add Text Annotations
        # Get Y-axis limit to place text (Note: hist is done so ylim is set)
        ymin, ymax = axes[1].get_ylim()
        # Position text at ~middle of log scale
        text_y = ymax * 0.5 
        
        axes[1].text(2.1, text_y, f'Traitors\n(>2.0)\nn={n_traitors}', color='red', fontweight='bold', ha='left')
        axes[1].text(0.9, text_y, f'Heroes\n(~1.0)\nn={n_heroes}', color='blue', fontweight='bold', ha='right')

        axes[1].set_title("Sensitivity Distribution")
        axes[1].set_xlabel("Sensitivity Index (Color/BW)")
        axes[1].set_ylabel("Count (Log)")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  Saved sensitivity plot to {save_path}")
        plt.show()


# ==========================================
# PHASE 2: Grad-CAM Implementation
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle_fwd = None
        self.handle_bwd = None
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handle_fwd = self.target_layer.register_forward_hook(save_activation)
        # Use register_full_backward_hook for newer pytorch, or register_backward_hook for older
        try:
            self.handle_bwd = self.target_layer.register_full_backward_hook(save_gradient)
        except AttributeError:
            self.handle_bwd = self.target_layer.register_backward_hook(save_gradient)

    def remove_hooks(self):
        if self.handle_fwd: self.handle_fwd.remove()
        if self.handle_bwd: self.handle_bwd.remove()

    def generate_heatmap(self, input_tensor, target_class_idx):
        """
        Generate Grad-CAM heatmap for a specific target class.
        input_tensor: (1, 3, 28, 28)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Target specific class
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()
            
        score = output[:, target_class_idx]
        
        # Backward pass
        score.backward()
        
        # Generate CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU
        cam = F.relu(cam)
        
        # Resize using torch interpolation (replaces cv2)
        # Assuming cam is (H, W), we need (1, 1, H, W) for interpolate
        if len(cam.shape) == 2:
            cam = cam.unsqueeze(0).unsqueeze(0)
            
        cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, target_class_idx



# ==========================================
# PHASE 3: Structural Analysis (Circuits)
# ==========================================

class StructuralAnalyzer:
    def __init__(self, sae, device):
        self.sae = sae
        self.device = device
        self.correlation_matrix = None

    def analyze_correlations(self, images, model, batch_size=64):
        """
        Compute correlation matrix of SAE features across dataset.
        """
        print("\nRunning Structural Analysis (Polysemanticity check)...")
        model.eval()
        self.sae.eval()
        
        # Get all latent activations
        all_z = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(self.device)
                acts = model.get_fc1_activations(batch)
                _, z, _ = self.sae(acts)
                all_z.append(z.cpu().numpy())
        
        Z = np.concatenate(all_z, axis=0) # Shape: (N, HiddenDim)
        
        # Filter dead neurons to avoid NaN correlations
        active_indices = np.where(Z.var(axis=0) > 1e-6)[0]
        Z_active = Z[:, active_indices]
        
        print(f"  Computing correlation matrix for {len(active_indices)} active features...")
        # Add small epsilon noise to break perfect symmetries if any
        Z_active = Z_active + np.random.normal(0, 1e-9, Z_active.shape)
        
        self.correlation_matrix = np.corrcoef(Z_active, rowvar=False)
        self.active_indices = active_indices
        return self.correlation_matrix

    def find_circuits(self, traitors, heroes):
        """
        Find high correlations between Traitors (Color) and Heroes (Shape).
        """
        if self.correlation_matrix is None:
            print("  Run analyze_correlations first.")
            return

        print("  Scanning for 'Circuits' (Traitor-Hero Pairs)...")
        
        # Create map from original index to active matrix index
        idx_map = {orig: new for new, orig in enumerate(self.active_indices)}
        
        circuits = []
        threshold = 0.3 # Correlation threshold
        
        for t_idx, t_score in traitors:
            if t_idx not in idx_map: continue
            
            for h_idx, h_score in heroes:
                if h_idx not in idx_map: continue
                
                corr = self.correlation_matrix[idx_map[t_idx], idx_map[h_idx]]
                
                if corr > threshold:
                    circuits.append({
                        'Traitor': t_idx, 'Traitor_Score': t_score,
                        'Hero': h_idx, 'Hero_Score': h_score,
                        'Correlation': corr
                    })
        
        # Sort by correlation strength
        circuits.sort(key=lambda x: x['Correlation'], reverse=True)
        
        print(f"  Found {len(circuits)} potential circuits (Correlation > {threshold})")
        for i, c in enumerate(circuits[:5]):
            print(f"    Circuit {i}: Traitor #{c['Traitor']} (Color) <--> Hero #{c['Hero']} (Shape) | r={c['Correlation']:.4f}")
            
        return circuits

    def plot_circuit_analysis(self, traitors, heroes, save_path='circuit_analysis.png'):
        """
        Visualize the interaction between Top Traitors and Top Heroes.
        Plots a correlation heatmap for the subset of features.
        """
        if self.correlation_matrix is None:
            return
        
        # Filter for Top 20 of each for visibility
        t_indices = [t[0] for t in traitors[:20]] 
        h_indices = [h[0] for h in heroes[:20]]
        
        if not t_indices or not h_indices:
            return

        # Map original feature indices to the active_indices used in correlation matrix
        idx_map = {orig: new for new, orig in enumerate(self.active_indices)}
        
        matrix_subset = np.zeros((len(t_indices), len(h_indices)))
        
        valid_t_labels = []
        valid_h_labels = []
        
        # Fill subset matrix
        for r, t_idx in enumerate(t_indices):
            if t_idx in idx_map:
                valid_t_labels.append(f"T{t_idx}")
                # We need to make sure we align the rows/cols correctly for plotting
                # But here we just want to fill the grid strictly by the list order
                
                for c, h_idx in enumerate(h_indices):
                    if h_idx in idx_map:
                        matrix_subset[r, c] = self.correlation_matrix[idx_map[t_idx], idx_map[h_idx]]
        
        # Plotting
        # Note: We might have empty rows/cols if features were 'dead' and filtered out of correlation matrix
        # Simple fix: just plot the computed matrix
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix_subset, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Pearson Correlation')
        
        # Axis labels
        ax.set_xticks(np.arange(len(h_indices)))
        ax.set_yticks(np.arange(len(t_indices)))
        ax.set_xticklabels([f"H{h}" for h in h_indices], rotation=45, ha='right')
        ax.set_yticklabels([f"T{t}" for t in t_indices])
        
        ax.set_xlabel("Heroes (Shape)")
        ax.set_ylabel("Traitors (Color)")
        ax.set_title("Circuit Map: Color-Shape Dependencies\nRed = Positive Correlation (Co-occurence)")
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  Saved Circuit Analysis to {save_path}")



# ==========================================
# PHASE 4: Causal Intervention (The Cure)
# ==========================================

class CausalIntervener:
    def __init__(self, model, sae, device):
        self.model = model
        self.sae = sae
        self.device = device

    def find_failure_cases(self, images, labels, batch_size=64):
        """
        Identify images where the model fails (Predictions != Labels).
        """
        self.model.eval()
        failures = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                batch_lbls = labels[i:i+batch_size]
                
                tensor = torch.FloatTensor(batch_imgs).permute(0, 3, 1, 2).to(self.device)
                output = self.model(tensor)
                preds = output.argmax(dim=1).cpu().numpy()
                
                # Check for errors
                for idx, (p, l) in enumerate(zip(preds, batch_lbls)):
                    if p != l:
                        failures.append({
                            'index': i + idx,
                            'image': batch_imgs[idx],
                            'label': l,
                            'pred': p
                        })
        
        print(f"  Found {len(failures)} failure cases in {len(images)} samples.")
        return failures

    def perform_surgery(self, failure_cases, traitors, heroes, boost_factor=2.0, verbose=True):
        """
        Attempt to cure failure cases by suppressing Traitors and boosting Heroes.
        """
        if verbose:
            print(f"\nPerforming Surgery with Boost Factor {boost_factor}x...")
        success_count = 0
        cured_cases_list = []
        
        # Helper lists
        traitor_indices = [t[0] for t in traitors]
        hero_indices = [h[0] for h in heroes]
        
        # Convert list of indices to tensors for faster indexing
        t_tensor = torch.LongTensor(traitor_indices).to(self.device)
        h_tensor = torch.LongTensor(hero_indices).to(self.device)

        for case in failure_cases:
            img_tensor = torch.FloatTensor(case['image']).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            target_label = case['label']
            original_pred = case['pred']
            
            # 1. Custom Forward Pass with Intervention
            with torch.no_grad():
                # A. Get FC1 activations
                fc1_act = self.model.get_fc1_activations(img_tensor)
                
                # B. Encode into SAE Latents
                # TopK encode manually to allow editing before decode
                # Note: TopKSparseAutoencoder.encode does full forward
                z, _ = self.sae.encode(fc1_act)
                
                # C. THE SURGERY
                z_edited = z.clone()
                
                # Suppress Traitors (Color) -> 0
                if len(traitor_indices) > 0:
                    z_edited[:, t_tensor] = 0.0
                
                # Boost Heroes (Shape) -> x Factor
                # We could refine this to only boost heroes relevant to the specific digit if we mapped them 
                # but boosting all heroes enhances "shapeness" generally.
                if len(hero_indices) > 0:
                    z_edited[:, h_tensor] *= boost_factor
                
                # D. Decode back to FC1
                fc1_recon = self.sae.decode(z_edited)
                
                # E. Finish Network Pass (Dropout + FC2)
                # Note: CNN3Layer forward is: pool/conv -> fc1 -> relu -> dropout -> fc2
                # get_fc1_activations returns "F.relu(self.fc1(x))"
                # So we simulate the rest:
                out = self.model.fc2(self.model.dropout(fc1_recon))
                new_pred = out.argmax(dim=1).item()
            
            # Check success
            if new_pred == target_label:
                success_count += 1
                cured_cases_list.append(case)
                if success_count <= 3: # Log first few successes
                    print(f"  [CURED] Corrected Label {target_label} (Was {original_pred} -> Now {new_pred})")
        
        print(f"  Surgery Results: Cured {success_count}/{len(failure_cases)} ({success_count/len(failure_cases)*100:.1f}%)")
        if verbose:
            print(f"  Surgery Results: Cured {success_count}/{len(failure_cases)} ({success_count/len(failure_cases)*100:.1f}%)")
        return success_count, cured_cases_list

    def sweep_intervention_strength(self, failures, traitors, heroes, factors=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]):
        print("\nSweeping Intervention Strengths (Boost Factors)...")
        results = []
        for factor in factors:
            n_cured, _ = self.perform_surgery(failures, traitors, heroes, boost_factor=factor, verbose=False)
            results.append(n_cured)
            print(f"  Factor {factor}x: Cured {n_cured}/{len(failures)}")
            
        plt.figure(figsize=(8, 5))
        plt.plot(factors, results, 'o-', linewidth=2)
        plt.xlabel("Hero Boost Factor (Multiplier)")
        plt.ylabel("Number of Cured Cases")
        plt.title("Intervention Efficacy vs Strength")
        plt.grid(True, alpha=0.3)
        plt.savefig('intervention_sweep.png')
        print("  Saved sweep plot to intervention_sweep.png")

    def visualize_cures(self, cured_cases, model, sae, traitors, heroes, device, save_path='surgery_validation.png'):
        """
        Visualize the effect of surgery using Grad-CAM on the Intervened Model.
        """
        if not cured_cases:
            return

        print(f"\nVisualizing surgery effects on {min(5, len(cured_cases))} cured cases...")
        
        # 1. Define Intervened Model Wrapper
        # This allows Grad-CAM to forward pass through the surgery logic
        class IntervenedModel(nn.Module):
            def __init__(self, original_model, sae_model, t_idxs, h_idxs, boost):
                super().__init__()
                self.model = original_model
                self.sae = sae_model
                self.t_idxs = torch.LongTensor(t_idxs).to(device)
                self.h_idxs = torch.LongTensor(h_idxs).to(device)
                self.boost = boost
                
            def forward(self, x):
                # Manual Forward of CNN3Layer until FC1
                x = self.model.pool1(F.relu(self.model.conv1(x)))
                x = self.model.pool2(F.relu(self.model.conv2(x)))
                x = self.model.pool3(F.relu(self.model.conv3(x))) # Conv3 Hook fires here
                x = x.reshape(x.size(0), -1)
                fc1_act = F.relu(self.model.fc1(x))
                
                # SAE Surgery
                z, _ = self.sae.encode(fc1_act)
                if len(self.t_idxs) > 0: z[:, self.t_idxs] = 0.0
                if len(self.h_idxs) > 0: z[:, self.h_idxs] *= self.boost
                fc1_recon = self.sae.decode(z)
                
                # Finish
                return self.model.fc2(self.model.dropout(fc1_recon))

        # Setup Models
        t_indices = [t[0] for t in traitors]
        h_indices = [h[0] for h in heroes]
        
        wrapped_model = IntervenedModel(model, sae, t_indices, h_indices, 2.5).to(device)
        wrapped_model.eval()
        
        # Setup Grad-CAMs
        # One for original model (to see error), One for new model (to see cure)
        # Note: Both share 'model.conv3' so we must manage hooks carefully.
        # GradCAM class adds hooks to the layer. Since both models use the SAME layer object,
        # we can just use one GradCAM instance and swap the .model attribute for context.
        
        grad_cam = GradCAM(model, model.conv3) # Attached to conv3
        
        n_show = min(len(cured_cases), 5)
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4*n_show))
        if n_show == 1: axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            case = cured_cases[i]
            img = case['image']
            img_tensor = torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            
            # A. PRE-SURGERY (Original Model)
            grad_cam.model = model # Point to original
            # TARGET: The WRONG prediction (we want to see why it was fooled)
            hm_bad, _ = grad_cam.generate_heatmap(img_tensor, case['pred'])
            
            # B. POST-SURGERY (Intervened Model)
            grad_cam.model = wrapped_model # Point to wrapper (Hooks still on conv3)
            # TARGET: The CORRECT label (which is now the prediction)
            hm_good, _ = grad_cam.generate_heatmap(img_tensor, case['label'])
            
            # Plot
            ax0, ax1, ax2 = axes[i]
            
            ax0.imshow(img)
            ax0.set_title(f"Input (True: {case['label']})")
            ax0.axis('off')
            
            ax1.imshow(img)
            ax1.imshow(hm_bad, cmap='jet', alpha=0.5)
            ax1.set_title(f"Original (Pred: {case['pred']})\nConfusion")
            ax1.axis('off')
            
            ax2.imshow(img)
            ax2.imshow(hm_good, cmap='jet', alpha=0.5)
            ax2.set_title(f"Cured (Pred: {case['label']})\nFocus")
            ax2.axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  Saved Surgery Validation to {save_path}")
        grad_cam.remove_hooks()



def validate_traitors_with_gradcam(model, sae, traitors, images, labels, device, n_examples=3):

    """
    Phase 2 Validation: Run Grad-CAM on images where Traitor features are active.
    """
    if not traitors:
        print("No traitors to validate.")
        return

    print(f"\nPhase 2: Verifying Traitor Features with Grad-CAM...")
    
    # Initialize GradCAM on the last conv layer
    grad_cam = GradCAM(model, model.conv3)
    
    # Select top few traitors
    top_traitors = [t[0] for t in traitors[:3]]
    
    # Pre-compute activations to find max activating images for these traitors
    # We'll do this on a subset to save time
    subset_size = min(len(images), 1000)
    subset_images = images[:subset_size]
    subset_labels = labels[:subset_size]
    
    subset_tensor = torch.FloatTensor(subset_images).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        fc1_acts = model.get_fc1_activations(subset_tensor)
        _, sae_acts, _ = sae(fc1_acts)
        sae_acts = sae_acts.cpu().numpy()
    
    fig, axes = plt.subplots(len(top_traitors), n_examples * 2, figsize=(3 * n_examples * 2, 3 * len(top_traitors)))
    if len(top_traitors) == 1: axes = axes.reshape(1, -1)
    
    for row_idx, trait_idx in enumerate(top_traitors):
        # Find images where this traitor is most active
        feature_acts = sae_acts[:, trait_idx]
        top_img_indices = np.argsort(feature_acts)[-n_examples:][::-1]
        
        for col_idx, img_idx in enumerate(top_img_indices):
            img = subset_images[img_idx]
            label = subset_labels[img_idx]
            activation_val = feature_acts[img_idx]
            
            img_tensor = torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            
            # Run Grad-CAM
            # We ask: What predicted class is the model seeing?
            heatmap, pred_class = grad_cam.generate_heatmap(img_tensor, None) # Predict predicted class
            
            # Visualization
            ax_orig = axes[row_idx, col_idx * 2]
            ax_cam = axes[row_idx, col_idx * 2 + 1]
            
            ax_orig.imshow(img)
            ax_orig.set_title(f"Traitor {trait_idx}\nAct: {activation_val:.2f}\nPred: {pred_class} (True {label})")
            ax_orig.axis('off')
            
            ax_cam.imshow(img)
            ax_cam.imshow(heatmap, cmap='jet', alpha=0.5)
            ax_cam.set_title(f"Grad-CAM\nDoes it look at color?")
            ax_cam.axis('off')

    plt.tight_layout()
    plt.savefig('traitor_gradcam_validation.png')
    print("  Saved Grad-CAM validation to traitor_gradcam_validation.png")
    plt.show() # Show in notebook/script if interactive
    
    grad_cam.remove_hooks()


def evaluate_reconstruction(model, sae, data, device):
    images, _ = data
    model.eval()
    sae.eval()
    
    subset_size = min(len(images), 1000)
    subset = torch.FloatTensor(images[:subset_size]).permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        fc1_acts = model.get_fc1_activations(subset)
        recon_acts, _, _ = sae(fc1_acts)
        
        mse = F.mse_loss(recon_acts, fc1_acts).item()
        l2_norm = torch.norm(fc1_acts, p=2).mean().item()
        
    print(f"\n[SAE Quality Check] Reconstruction MSE: {mse:.6f} (vs Avg Activation Norm: {l2_norm:.4f})")
    if mse > 0.1:
        print("  WARNING: SAE Reconstruction is poor. Interventions may be unreliable.")
    else:
        print("  ✓ SAE Reconstruction looks reasonable.")


def train_concept_probes(model, images, labels, device):

    model.eval()
    
    # Get activations
    with torch.no_grad():
        images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(device)
        activations = model.get_fc1_activations(images_tensor).cpu().numpy()
    
    # Detect dominant color from images
    # Simple: extract mean R, G, B channel values
    mean_colors = images.mean(axis=(1, 2))  # Shape: (N, 3)
    dominant_color_idx = mean_colors.argmax(axis=1)  # 0=R, 1=G, 2=B
    
    # Train probes
    # 1. Shape probe (predict digit)
    shape_probe = LogisticRegression(max_iter=1000, random_state=42)
    shape_probe.fit(activations, labels)
    shape_acc = shape_probe.score(activations, labels)
    
    # 2. Color probe (predict dominant color)
    color_probe = LogisticRegression(max_iter=1000, random_state=42)
    color_probe.fit(activations, dominant_color_idx)
    color_acc = color_probe.score(activations, dominant_color_idx)
    
    print("\n=== Linear Probe Analysis ===")
    print(f"Shape (digit) probe accuracy: {shape_acc*100:.2f}%")
    print(f"Color probe accuracy: {color_acc*100:.2f}%")
    
    print(f"\nInterpretation:")
    if color_acc > shape_acc:
        print("  ⚠️  Color is MORE linearly separable than shape!")
        print("  → The model represents color more explicitly than shape")
    else:
        print("  ✓ Shape is more linearly separable than color")
        print("  → The model represents shape more explicitly")
    
    return shape_probe, color_probe, shape_acc, color_acc



def main():
    print("="*70)
    print("Task 6: ADVANCED DECOMPOSITION - SOTA Interpretability Techniques")
    print("="*70)
    
    # Load data
    print("\n[1/7] Loading data...")
    
    def load_data(path):
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return None, None
        data = np.load(path)
        return data['images'].astype('float32') / 255.0, data['labels']
    
    env1_images, env1_labels = load_data(TRAIN_DATA_PATH)
    env2_images, env2_labels = load_data(TEST_DATA_PATH)
    
    if env1_images is None:
        print("Data not found. Please update paths.")
        return
    
    print(f"Environment 1: {env1_images.shape}")
    print(f"Environment 2: {env2_images.shape}")
    
    # Load model
    print("\n[2/7] Loading biased model...")
    model = CNN3Layer().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model not found!")
        return
    model.eval()
    
    # TECHNIQUE 1: TopK Sparse Autoencoder
    print("\n[3/7] Training TopK Sparse Autoencoder...")
    print(f"Architecture: 128 -> {SAE_HIDDEN_DIM} (TopK={TOPK_K}) -> 128")
    

    def get_activations_batched(images, batch_size=64):
        all_acts = []
        for i in range(0, len(images), batch_size):
            batch = torch.FloatTensor(images[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
            acts = model.get_fc1_activations(batch).cpu()
            all_acts.append(acts)
            del batch
            torch.cuda.empty_cache()
        return torch.cat(all_acts, dim=0)
    
    with torch.no_grad():
        activations = get_activations_batched(env1_images)
    

    gc.collect()
    torch.cuda.empty_cache()
    

    topk_sae = TopKSparseAutoencoder(FC1_UNITS, SAE_HIDDEN_DIM, TOPK_K).to(device)
    optimizer = optim.Adam(topk_sae.parameters(), lr=SAE_LEARNING_RATE)
    
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=SAE_BATCH_SIZE, shuffle=True)
    
    for epoch in range(SAE_EPOCHS):
        epoch_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            x_recon, z, _ = topk_sae(x)
            loss = F.mse_loss(x_recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            with torch.no_grad():
                topk_sae.decoder.weight.data = F.normalize(topk_sae.decoder.weight.data, dim=0)
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{SAE_EPOCHS}: Loss = {epoch_loss/len(loader):.6f}")
    

    print("\n[4/7] Computing Concept Steering Vectors...")
    csv = ConceptSteeringVectors(model, device)
    color_vector = csv.compute_steering_vectors(env1_images, env1_labels, 
                                            env2_images, env2_labels)

    
    print("\nTesting color removal intervention:")
    csv.intervention_experiment(env2_images[:MEMORY_LIMIT_SAMPLES], env2_labels[:MEMORY_LIMIT_SAMPLES])
    

    print("\n[5/7] Performing Causal Neuron Ablation Analysis...")
    gc.collect()
    torch.cuda.empty_cache()
    cna = CausalNeuronAnalyzer(model, device)
    color_neurons, shape_neurons = cna.find_causal_neurons(
        env1_images[:MEMORY_LIMIT_SAMPLES], env1_labels[:MEMORY_LIMIT_SAMPLES],
        env2_images[:MEMORY_LIMIT_SAMPLES], env2_labels[:MEMORY_LIMIT_SAMPLES]
    )
    

    print("\n[6/7] Clustering SAE Features by Cross-Environment Behavior...")
    clusters, cluster_info, feat_chars = cluster_features_by_environment(
        model, topk_sae, 
        (env1_images, env1_labels), 
        (env2_images, env2_labels)
    )

    # ==========================================
    # PHASE 1 & 2 Execution
    # ==========================================
    print("\n" + "="*50)
    print("PHASE 1: Feature Identification & Classification")
    print("="*50)
    
    # Initialize Classifier
    feature_clf = FeatureClassifier(model, topk_sae, device)
    
    # Run Classification on Env1 (Colored) vs Generated B&W
    # Uses MEMORY_LIMIT_SAMPLES to keep it fast
    traitors, heroes = feature_clf.classify_features(env1_images[:1000])
    feature_clf.plot_sensitivity_analysis()
    
    print("\n" + "="*50)
    print("PHASE 2: Feature Validation with Grad-CAM")
    print("="*50)
    
    # Validate the discovered Traitors
    validate_traitors_with_gradcam(
        model, topk_sae, traitors, 
        env1_images, env1_labels, 
        device
    )
    

    print("\n[7/7] Training Linear Concept Probes...")
    shape_probe, color_probe, shape_acc, color_acc = train_concept_probes(
        model, env1_images, env1_labels, device
    )
    

    

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    effects = [cna.neuron_effects[i] for i in range(FC1_UNITS)]
    env1_effects = [e['env1_effect'] for e in effects]
    env2_effects = [e['env2_effect'] for e in effects]
    
    axes[0].scatter(env1_effects, env2_effects, alpha=0.6, c='blue', edgecolors='k')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].plot([-10, 10], [-10, 10], 'r--', alpha=0.5, label='Equal effect')
    axes[0].set_xlabel('Effect on Env1 (Original Colors) %')
    axes[0].set_ylabel('Effect on Env2 (Reversed Colors) %')
    axes[0].set_title('Causal Neuron Analysis\nNeurons above line = COLOR-specific')
    axes[0].legend()
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    

    axes[1].scatter(feat_chars[:, 0], feat_chars[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    axes[1].set_xlabel('Cross-Environment Activation Difference')
    axes[1].set_ylabel('Within-Environment Variance')
    axes[1].set_title('SAE Feature Clustering\nHigher X = More Color-Sensitive')
    
    plt.tight_layout()
    plt.savefig('advanced_sae_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    

    fig, ax = plt.subplots(figsize=(8, 5))
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = csv.intervention_experiment(env2_images[:MEMORY_LIMIT_SAMPLES], env2_labels[:MEMORY_LIMIT_SAMPLES], strengths)
    accs = [r['accuracy'] for r in results]
    
    ax.plot(strengths, accs, 'bo-', linewidth=2, markersize=8)
    ax.axhline(accs[0], color='gray', linestyle='--', label='Baseline (no steering)')
    ax.set_xlabel('Color Removal Strength')
    ax.set_ylabel('Accuracy on Reversed-Color Data (%)')
    ax.set_title('Effect of Removing "Color Direction" from Activations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_steering_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    

    
    print("\n" + "="*50)
    print("PHASE 3: Structural Analysis")
    print("="*50)
    
    analyzer = StructuralAnalyzer(topk_sae, device)
    analyzer = StructuralAnalyzer(topk_sae, device)
    analyzer.analyze_correlations(env1_images[:1000], model) # Use subset for speed
    circuits = analyzer.find_circuits(traitors, heroes)
    analyzer.plot_circuit_analysis(traitors, heroes)
    
    
    # Evaluate SAE Reconstruction Quality first
    evaluate_reconstruction(model, topk_sae, (env1_images, env1_labels), device)
    
    print("\n" + "="*50)
    print("PHASE 4: Causal Intervention")
    print("="*50)
    
    intervener = CausalIntervener(model, topk_sae, device)
    
    # 1. Identify failures in the Reversed (Hard) Environment
    print("Identifying failures in Env2 (Reversed Data)...")
    failures = intervener.find_failure_cases(env2_images[:1000], env2_labels[:1000])
    
    # 2. Perform Surgery
    if failures:
        # Sweep first
        intervener.sweep_intervention_strength(failures, traitors, heroes)
        
        # Then perform standard surgery for visualization
        n_cured, cured_list = intervener.perform_surgery(failures, traitors, heroes, boost_factor=2.5)
        # 3. Visualize Cures
        if n_cured > 0:
            intervener.visualize_cures(cured_list, model, topk_sae, traitors, heroes, device)
    else:
        print("No failures found to audit (Model is too good?).")

    # Save Models
    torch.save(topk_sae.state_dict(), 'topk_sae_model.pth')
    np.save('color_steering_vector.npy', color_vector)
    
    print("\nTask 6 Complete - All Phases Executed.")




if __name__ == "__main__":
    main()
