"""Sanity check: loss forward pass (CPU, DINO only — action ckpt 별도)"""
import torch
from loss import DINOLoss

device = torch.device("cpu")
B, H, W = 2, 320, 512

print("Loading DINO...")
dino = DINOLoss(device)

pred   = torch.randn(B, 3, H, W).clamp(-1, 1)
target = torch.randn(B, 3, H, W).clamp(-1, 1)

loss = dino(pred, target)
print("DINO loss:", loss.item())
print("OK")
