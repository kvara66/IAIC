"""
Independent action extractor.

공식:  Plain 3D ResBlock (3x3→3x3) + Bidirectional GRU + GroupNorm + SiLU
독자:  Bottleneck 3D ResBlock (1x1→3x3→1x1) + Bidirectional LSTM + BatchNorm3d + SiLU

채널: base_channels=8, expansion=4 → out channels [32,64,128,128] (공식과 동일 규모)

Input : (B, C, T, H, W) float32 in [-1, 1]
Output: (B, T, 6) float32 normalized action
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck3DBlock(nn.Module):
    """
    Bottleneck 3D ResBlock: 1x1 → 3x3 → 1x1
    공식 plain 3x3→3x3 와 완전히 다른 설계.
    """
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride=(1,1,1), dropout: float = 0.0):
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=1)
        self.bn1   = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1)
        self.bn2   = nn.BatchNorm3d(mid_ch)
        self.conv3 = nn.Conv3d(mid_ch, out_ch, kernel_size=1)
        self.bn3   = nn.BatchNorm3d(out_ch)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_ch),
        ) if (in_ch != out_ch or stride != (1,1,1)) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x)))
        h = F.silu(self.bn2(self.conv2(h)))
        h = self.drop(h)
        h = self.bn3(self.conv3(h))
        return F.silu(h + self.skip(x))


class ActionExtractorIndep(nn.Module):
    """
    독자 아키텍처 Action Extractor.

    공식과의 차이:
      블록 설계  : Plain 3x3→3x3 → Bottleneck 1x1→3x3→1x1
      정규화     : GroupNorm → BatchNorm3d
      시계열     : Bidirectional GRU → Bidirectional LSTM
    """

    def __init__(
        self,
        action_dims: int = 6,
        base_channels: int = 8,          # expansion=4 → out: [32,64,128,128]
        channel_mults=(1, 2, 4, 4),
        blocks_per_stage: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        mid_chs = [base_channels * m for m in channel_mults]   # [8,16,32,32]
        stem_ch = mid_chs[0] * Bottleneck3DBlock.expansion     # 32

        self.stem = nn.Sequential(
            nn.Conv3d(3, stem_ch, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            nn.BatchNorm3d(stem_ch),
            nn.SiLU(),
        )

        stages, cur = [], stem_ch
        for i, mid in enumerate(mid_chs):
            out = mid * Bottleneck3DBlock.expansion
            for j in range(blocks_per_stage):
                stride = (1,2,2) if i > 0 and j == 0 else (1,1,1)
                stages.append(Bottleneck3DBlock(cur, mid, stride=stride, dropout=dropout))
                cur = out
        self.encoder = nn.Sequential(*stages)
        self.final_ch = cur  # 128

        # BiLSTM (공식은 BiGRU)
        self.temporal = nn.LSTM(
            input_size=self.final_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, action_dims),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, C, T, H, W) → (B, T, action_dims)"""
        x = self.stem(video)               # (B, stem_ch, T, H/2, W/2)
        x = self.encoder(x)                # (B, final_ch, T, H', W')
        x = x.mean(dim=(-1,-2))            # (B, final_ch, T)
        x = x.permute(0, 2, 1)            # (B, T, final_ch)
        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.temporal(x)        # (B, T, lstm_hidden*2)
        return self.head(x)                # (B, T, action_dims)


def build_extractor(device="cuda") -> ActionExtractorIndep:
    return ActionExtractorIndep().to(device)


def compute_action_loss(
    extractor: ActionExtractorIndep,
    pred_video: torch.Tensor,
    gt_states: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(extractor(pred_video), gt_states)


if __name__ == "__main__":
    model = build_extractor("cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total/1e6:.1f}M")
    print(f"최종 채널: {model.final_ch}")  # 128 (공식과 동일)
    dummy = torch.randn(2, 3, 16, 320, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"출력 shape: {out.shape}")      # (2, 16, 6)
