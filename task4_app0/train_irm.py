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

# 1. Load Biased Data
TRAIN_PATH = 'mydata/dataset/train_data_rg.npz'
TEST_PATH = 'mydata/dataset/test_data_rg.npz'

def load_npz(path):
    data = np.load(path)
    X = data['images'].astype('float32') / 255.0
    y = data['labels']
    return X, y

X_train, y_train = load_npz(TRAIN_PATH)
X_test, y_test = load_npz(TEST_PATH)

X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
y_test_tensor = torch.LongTensor(y_test)

# 2. Model Definition
class CNN3Layer(nn.Module):
    def __init__(self):
        super(CNN3Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 3. IRM Logic
def compute_irm_penalty(logits, y, device):
    scale = torch.tensor(1.).to(device).requires_grad_(True)
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def train_irm(model, loader, optimizer, penalty_weight, device):
    model.train()
    total_loss = 0
    total_penalty = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        penalty = compute_irm_penalty(logits, y, device)
        loss = ce_loss + penalty_weight * penalty
        loss.backward()
        optimizer.step()
        total_loss += ce_loss.item()
        total_penalty += penalty.item()
    return total_loss / len(loader), total_penalty / len(loader)

# 4. Main Training Loop
def main():
    model = CNN3Layer().to(device)
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    penalty_anneal_iters = 5
    penalty_weight = 100.0

    print("\nStarting IRM Training...")
    for epoch in range(epochs):
        weight = penalty_weight if epoch >= penalty_anneal_iters else 1.0
        avg_loss, avg_penalty = train_irm(model, train_loader, optimizer, weight, device)
        
        # Eval
        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor.to(device))
            acc = (logits.argmax(1) == y_test_tensor.to(device)).float().mean().item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Penalty: {avg_penalty:.6f} | Test Acc: {acc:.2%}")

    torch.save(model.state_dict(), 'task4/task4_method1.pth')
    print("Model saved as task4/task4_method1.pth")

if __name__ == "__main__":
    main()
