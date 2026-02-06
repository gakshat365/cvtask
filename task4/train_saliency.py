import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_directml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Device setup
device = torch_directml.device()
print(f"Using DirectML device: {device}")
print(f"Device name: {torch_directml.device_name(0)}")

# 1. Load Data and Generate Masks
TRAIN_PATH = 'mydata/dataset/train_data_rg.npz'
TEST_PATH = 'mydata/dataset/test_data_rg.npz'

def load_npz_with_masks(path):
    data = np.load(path)
    X = data['images'].astype('float32') / 255.0
    y = data['labels']
    # Generate a simple mask by taking the max across channels (finds the digit)
    masks = (X.max(axis=-1) > 0.2).astype('float32')
    return X, y, masks

X_train, y_train, M_train = load_npz_with_masks(TRAIN_PATH)
X_test, y_test, M_test = load_npz_with_masks(TEST_PATH)

X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
y_train_t = torch.LongTensor(y_train)
M_train_t = torch.FloatTensor(M_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
y_test_t = torch.LongTensor(y_test)

# 2. Model with Grad-CAM Support
class CNNGradCAM(nn.Module):
    def __init__(self):
        super(CNNGradCAM, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_gradcam(self, class_idx):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        batch_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        batch_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam = (cam - batch_min) / (batch_max - batch_min + 1e-8)
        return cam

# 3. Saliency Training Logic
def train_saliency(model, loader, optimizer, alpha, device):
    model.train()
    total_ce = 0
    total_sal = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        optimizer.zero_grad()
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward(retain_graph=True)
        cam = model.get_gradcam(y)
        saliency_loss = F.mse_loss(cam, m)
        (alpha * saliency_loss).backward()
        optimizer.step()
        total_ce += ce_loss.item()
        total_sal += saliency_loss.item()
    return total_ce / len(loader), total_sal / len(loader)

# 4. Main Training Loop
def main():
    model = CNNGradCAM().to(device)
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t, M_train_t), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 15
    alpha = 5.0

    print("\nStarting Saliency-Guided Training...")
    for epoch in range(epochs):
        avg_ce, avg_sal = train_saliency(model, train_loader, optimizer, alpha, device)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            acc = (logits.argmax(1) == y_test_t.to(device)).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} | CE: {avg_ce:.4f} | Sal: {avg_sal:.4f} | Test Acc: {acc:.2%}")

    torch.save(model.state_dict(), 'task4/task4_method2.pth')
    print("Model saved as task4/task4_method2.pth")

if __name__ == "__main__":
    main()
