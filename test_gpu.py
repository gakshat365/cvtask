import torch
import torch_directml
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# DirectML device
device = torch_directml.device()
print(f"DirectML device: {device}")
print(f"Device name: {torch_directml.device_name(0)}")

# Tensor Operation Test
print("\nRunning Tensor Operation Test on GPU...")
try:
    # Create tensors on DirectML device
    x = torch.ones(3, 3).to(device)
    y = torch.ones(3, 3).to(device)
    
    # Perform addition
    z = x + y
    
    print("Successfully performed tensor addition on DirectML device:")
    print(z)
except Exception as e:
    print(f"DirectML Error: {e}")
