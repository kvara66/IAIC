"""
Latent Residual UNet with Action FiLM Conditioning (Non-AR, bigger UNet + timestep t).

Pipeline:
  frame_0 (B,3,H,W) → SDXL VAE encode → latent_0 (B,4,60,80)
  [state_0, state_i, t/15] (B,13) → ActionMLP → cond (B,512)   [absolute + timestep]
  latent_0 + cond → UNet (ch=[256,512,1024]) → delta (B,4,60,80)
  pred_latent = latent_0 + delta
  pred_latent → SDXL VAE decode → pred_frame_i (B,3,H,W)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    def __init__(self, pretrained: str = "madebyollin/sdxl-vae-fp16-fix"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained)
        self.vae.requires_grad_(False)
        self.scale = 0.13025

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.mean * self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scale).sample


class ActionMLP(nn.Module):
    """[state_0 (6D), state_i (6D), t_norm (1D)] → cond vector (cond_dim)"""
    def __init__(self, in_dim: int = 13, cond_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
            nn.SiLU(),
        )

    def forward(self, state_0: torch.Tensor, state_i: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        # t_norm: (B,) → (B,1)
        x = torch.cat([state_0, state_i, t_norm.unsqueeze(-1)], dim=-1)
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)
        return x * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(out_ch, cond_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class LatentResidualUNet(nn.Module):
    def __init__(self, latent_ch: int = 4, cond_dim: int = 256):
        super().__init__()
        ch = [256, 512, 1024]  # 2x bigger than baseline [128,256,512]

        self.enc_in = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        self.enc0 = ResBlock(ch[0], ch[0], cond_dim)
        self.down0 = nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1)
        self.enc1 = ResBlock(ch[1], ch[1], cond_dim)
        self.down1 = nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1)

        self.mid0 = ResBlock(ch[2], ch[2], cond_dim)
        self.attn = AttentionBlock(ch[2])
        self.mid1 = ResBlock(ch[2], ch[2], cond_dim)

        self.up1 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec1 = ResBlock(ch[1] * 2, ch[1], cond_dim)
        self.up0 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec0 = ResBlock(ch[0] * 2, ch[0], cond_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(min(8, ch[0]), ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], latent_ch, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.enc_in(latent)
        s0 = self.enc0(x, cond)
        x = self.down0(s0)
        s1 = self.enc1(x, cond)
        x = self.down1(s1)

        x = self.mid0(x, cond)
        x = self.attn(x)
        x = self.mid1(x, cond)

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1), cond)
        x = self.up0(x)
        x = self.dec0(torch.cat([x, s0], dim=1), cond)

        return self.out(x)


class ImageEditingModel(nn.Module):
    def __init__(self, cond_dim: int = 256):
        super().__init__()
        self.vae = VAEWrapper()
        self.action_mlp = ActionMLP(in_dim=13, cond_dim=cond_dim)
        self.unet = LatentResidualUNet(latent_ch=4, cond_dim=cond_dim)

    def forward(
        self,
        frame_0: torch.Tensor,
        state_0: torch.Tensor,
        state_i: torch.Tensor,
        t_norm: torch.Tensor,  # (B,) normalized timestep: t/15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_0 = self.vae.encode(frame_0)
        cond = self.action_mlp(state_0, state_i, t_norm)
        delta = self.unet(latent_0, cond)
        pred_latent = latent_0 + delta
        pred_frame = self.vae.decode(pred_latent)
        return pred_frame, pred_latent
