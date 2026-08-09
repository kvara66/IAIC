"""
Loss functions for image editing regression.

Total loss = 0.3 * dino_loss + 0.3 * r3d_loss + 0.4 * action_loss
- dino_loss : 1 - cosine_similarity(DINO features)
- r3d_loss  : 1 - cosine_similarity(R3D-18 features)  [Stage 2, optional]
- action_loss: plain average of 6 per-dim MAE in normalized space, equal weight.
  채점기와 동일하게 6개 관절 평균 하나로 집계함(합산 아님). 차등 가중치 없음 —
  total_loss가 대회 점수의 정직한 대리값이어야 하므로.

Action extractor: independently trained ActionExtractorIndep
(action_extractor_indep.py), weights from runs/extractor_v2/best.pt
(50-epoch run, best val_mae=0.201 @ epoch 43).
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import MC3_18_Weights, mc3_18

from action_extractor_indep import ActionExtractorIndep

SUBMISSION_KIT = Path(__file__).parent.parent / "submission_kit"
ACTION_CKPT    = Path(__file__).parent / "runs" / "extractor_v2" / "best.pt"

ACTIVE_DIMS = [0, 1, 2, 3, 4, 5]  # 전체 6개 dim
# 관절 순서: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

SUBMISSION_H = 320
SUBMISSION_W = 512


def _resize_pad(frames: torch.Tensor, target_h: int, target_w: int, pad_value: float = 0.0) -> torch.Tensor:
    """비율 유지 리사이즈 + 패딩. 채점기(submission_kit/feature_csv_utils.py)의
    preprocess_video/_resize_pad_frame_batch와 동일 로직 — 실제 채점 시 모든 영상이
    이 방식으로 먼저 320x512(SUBMISSION_H/W)로 정규화된 뒤 각 모델별로 처리되므로,
    학습 loss 계산도 이 전처리를 거쳐야 채점 방식과 일치함.

    frames: (..., 3, H, W) — 마지막 3차원(C,H,W) 기준, 앞의 배치/시간 차원은 임의.
    입력은 [0, 1] 범위여야 함(그 이상은 호출부에서 변환).
    """
    orig_shape = frames.shape
    x = frames.reshape(-1, *orig_shape[-3:])  # (N, 3, H, W)
    _, _, height, width = x.shape
    scale = min(target_h / height, target_w / width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    x = F.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
    return x.reshape(*orig_shape[:-2], target_h, target_w)


# ── DINO ───────────────────────────────────────────────────────────────────────

class DINOLoss(nn.Module):
    """
    1 - cosine_similarity of CLIP ViT-B/16 (OpenAI) image features.

    채점기는 vit_small_patch14_dinov2(self-supervised)를 씀 — 동일 모델 사용 금지 규정으로
    CLIP(contrastive, 다른 학습 패러다임/데이터)로 대체. patch32보다 patch16이 토큰 해상도가
    높아(7x7->14x14) DINO의 patch14 세밀도에 더 가까움. 클래스명은 하위 호환 위해 유지.
    """
    MODEL_NAME = "vit_base_patch16_clip_224.openai"
    IMG_SIZE = 224  # CLIP ViT-B/16 native resolution

    def __init__(self, device: torch.device):
        super().__init__()
        self.model = timm.create_model(self.MODEL_NAME, pretrained=True, num_classes=0)
        self.model.to(device).eval().requires_grad_(False)
        # OpenAI CLIP normalization (ImageNet 통계 아님)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [-1, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        # 채점기와 동일: 먼저 320x512로 비율유지+패딩, 그 다음 DINO 입력 크기로 또 한번 비율유지+패딩
        x = _resize_pad(x, SUBMISSION_H, SUBMISSION_W)
        x = _resize_pad(x, self.IMG_SIZE, self.IMG_SIZE)
        x = (x - self.mean) / self.std
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", next(iter(out.values())))
        return out

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_pred = self._extract(pred)
        with torch.no_grad():
            f_target = self._extract(target)
        return 1 - F.cosine_similarity(f_pred, f_target, dim=-1).mean()


# ── R3D-18 ─────────────────────────────────────────────────────────────────────

class R3DLoss(nn.Module):
    """
    1 - cosine_similarity of MC3-18 video features (Stage 2 only).

    채점기는 r3d_18(순수 3D conv)를 씀 — 동일 모델 사용 금지 규정으로 MC3-18(초반만 3D,
    나머지 2D conv)로 대체. Kinetics-400 전처리(mean/std/112x112)는 r3d_18과 동일해서
    변경 없음. 클래스명은 하위 호환 위해 유지.
    """
    KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    KINETICS_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    def __init__(self, device: torch.device):
        super().__init__()
        model = mc3_18(weights=MC3_18_Weights.DEFAULT)
        model.fc = nn.Identity()
        self.model = model.to(device).eval().requires_grad_(False)
        self.register_buffer("mean", self.KINETICS_MEAN.to(device))
        self.register_buffer("std", self.KINETICS_STD.to(device))

    def _extract(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, 3, H, W) in [-1, 1]
        x = (video.clamp(-1, 1) + 1) / 2
        # 채점기와 동일: 먼저 320x512로 비율유지+패딩(공통 전처리)
        x = _resize_pad(x, SUBMISSION_H, SUBMISSION_W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        # R3D 자체는 채점기도 패딩 없이 그냥 112x112로 직접 리사이즈함(여기는 원래대로)
        x = F.interpolate(x, size=(x.shape[2], 112, 112), mode="trilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)

    def forward(self, pred_video: torch.Tensor, target_video: torch.Tensor) -> torch.Tensor:
        f_pred = self._extract(pred_video)
        with torch.no_grad():
            f_target = self._extract(target_video)
        return 1 - F.cosine_similarity(f_pred, f_target, dim=-1).mean()


# ── Action MAE ─────────────────────────────────────────────────────────────────

class ActionMAELoss(nn.Module):
    """
    6개 관절 각각 개별 L1 loss로 계산(동일 가중치)한 뒤 평균 — 채점기
    (feature_csv_utils.py의 torch.mean(..., dim=(1,2)))와 정확히 동일한 집계 방식.
    6개로 나눠서 계산하나 한 번에 flat mean을 내나 값/gradient 둘 다 완전히
    동일함(수학적으로 동치) — 6개로 나누는 유일한 이유는 로깅(per_dim)용.
    uncertainty weighting 등 관절별 차등 가중치는 안 씀 — total_loss가 대회
    점수의 정직한 대리값이어야 해서, 학습되는 가중치로 왜곡되면 안 됨.

    Returns (total_loss, dict of per-dim losses — 로깅 전용, 학습엔 영향 없음).
    """

    def __init__(self, ckpt_path: str | None = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = str(ACTION_CKPT)

        self.extractor = ActionExtractorIndep()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.extractor.load_state_dict(state_dict)
        self.extractor.to(device).eval().requires_grad_(False)
        # forward()가 LSTM 호출을 자체적으로 cudnn.flags(enabled=False)로 감싸서
        # train/eval 모드가 cuDNN RNN backward에 영향 없음 -> eval() 고정 가능
        # (BatchNorm3d/Dropout을 위해 eval 고정이 맞음, 공식 extractor의 GroupNorm+GRU와 다름)

    def forward(
        self,
        pred_video: torch.Tensor,
        gt_actions_normalized: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        pred_video: (B, T, 3, H, W) in [-1, 1]
        gt_actions_normalized: (B, T, 6) normalized observation.state

        Returns (total_loss, per_dim_losses)
        """
        # 채점기와 동일: 먼저 320x512로 비율유지+패딩(공통 전처리). Action은 채점기도
        # 그 이후 추가 리사이즈 없이 그대로 씀 — extractor가 [-1,1] 입력을 기대하므로 왕복 변환
        x01 = (pred_video.clamp(-1, 1) + 1) / 2
        x01 = _resize_pad(x01, SUBMISSION_H, SUBMISSION_W)
        x = x01 * 2.0 - 1.0

        # extractor expects (B, C, T, H, W) in [-1, 1]
        x = x.permute(0, 2, 1, 3, 4)
        pred_actions = self.extractor(x)  # (B, T, 6)

        per_dim = {}
        total = torch.tensor(0.0, device=pred_video.device)
        for i in ACTIVE_DIMS:
            d = F.l1_loss(pred_actions[:, :, i], gt_actions_normalized[:, :, i])
            per_dim[f"action_mae_dim{i}"] = d
            total = total + d

        # 채점기(feature_csv_utils.py)는 torch.mean(..., dim=(1,2))로 T축+6개 관절축을
        # 한꺼번에 평균냄 — 6개를 더하기만 하면 6배 부풀려지므로 평균으로 맞춤
        total = total / len(ACTIVE_DIMS)

        return total, per_dim


# ── Combined Loss ──────────────────────────────────────────────────────────────

class TotalLoss(nn.Module):
    """
    Stage 1: 0.3 * DINO + 0.4 * Action  (R3D 제외)
    Stage 2: 0.3 * DINO + 0.3 * R3D + 0.4 * Action
    """
    W_DINO   = 0.3
    W_R3D    = 0.3
    W_ACTION = 0.4

    def __init__(self, device: torch.device, action_ckpt: str | None = None, use_r3d: bool = False):
        super().__init__()
        self.dino = DINOLoss(device)
        self.action = ActionMAELoss(action_ckpt, device)
        self.use_r3d = use_r3d
        if use_r3d:
            self.r3d = R3DLoss(device)

    def forward(
        self,
        pred_frame: torch.Tensor,
        gt_frame: torch.Tensor,
        gt_actions_norm: torch.Tensor,
        pred_video: torch.Tensor | None = None,
        gt_video: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        pred_frame : (B, 3, H, W)   — 단일 프레임 예측
        gt_frame   : (B, 3, H, W)   — GT 다음 프레임
        gt_actions_norm: (B, T, 6)  — normalized observation.state
        pred_video : (B, T, 3, H, W) — Stage 2용 (optional)
        gt_video   : (B, T, 3, H, W) — Stage 2용 (optional)
        """
        logs = {}

        dino = self.dino(pred_frame, gt_frame)
        logs["dino_loss"] = dino

        # action loss: 단일 프레임을 (B,1,3,H,W)로 확장해서 넘김
        pred_vid_single = pred_frame.unsqueeze(1)
        action_total, action_per_dim = self.action(pred_vid_single, gt_actions_norm[:, :1])
        logs.update(action_per_dim)
        logs["action_loss"] = action_total

        loss = self.W_DINO * dino + self.W_ACTION * action_total

        if self.use_r3d and pred_video is not None and gt_video is not None:
            r3d = self.r3d(pred_video, gt_video)
            logs["r3d_loss"] = r3d
            loss = loss + self.W_R3D * r3d

        logs["total_loss"] = loss
        return loss, logs
