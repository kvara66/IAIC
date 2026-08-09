"""Sanity check: model forward pass (no loss, CPU)"""
import torch
from model import ImageEditingModel

B, H, W = 2, 320, 512
model = ImageEditingModel(cond_dim=256)
model.eval()

frame_i  = torch.randn(B, 3, H, W).clamp(-1, 1)
state_i  = torch.randn(B, 6)
state_i1 = torch.randn(B, 6)

with torch.no_grad():
    pred_frame, pred_latent = model(frame_i, state_i, state_i1)

print("pred_frame  :", pred_frame.shape, pred_frame.min().item(), pred_frame.max().item())
print("pred_latent :", pred_latent.shape)
print("OK")
