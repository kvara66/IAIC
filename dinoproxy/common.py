"""채점기 전처리를 한 글자도 안 바꾸고 옮긴 것 + 후보 모델용 일반화.

원본: open/submission_kit/feature_csv_utils.py

왜 그대로여야 하나:
  목적은 "DINO 가 두 이미지의 닮음을 어떻게 판단하는가" 를 알아내는 것이다.
  전처리가 다르면 애초에 다른 그림을 비교하는 셈이라 상관계수가 무의미해진다.

원본 함수와의 대응:
  preprocess_video               -> stage1_video_norm      (영상 규격 320x512)
  to_eval_uint8                  -> stage1_to_uint8        (uint8 왕복 = 양자화)
  _resize_pad_frame_batch        -> stage2_resize_pad      (모델 입력 크기)
  resolve_dino_image_size        -> model_input_size
  _normalize_image_model_output  -> pool_output

리사이즈가 두 번 일어난다 (둘 다 비율 유지 + 중앙 정렬 + 검은 패딩):
  480x640 원본 --1단계--> 320x427 + 좌우 검은 여백 42/43 = 320x512
  320x512      --2단계--> 324x518 + 상하 검은 여백 97/97 = 518x518   (DINO)
  최종 내용 면적 = 0.834 x 0.625 = 52%.  나머지 48% 가 검은색이다.

⚠ 2단계는 [0,1] 공간에서 0.0 으로 패딩한 "뒤에" 정규화한다.
  순서를 바꾸면 검은 여백이 (0-mean)/std 가 아니라 0 이 되어 값이 달라진다.
  원본 238~249행 + 310행이 그 순서다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# 원본 20~21행. DINO 는 이 값을 쓴다(timm dinov2 의 pretrained_cfg 와 동일).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

EVAL_H, EVAL_W = 320, 512          # 대회 영상 규격


# ---------------------------------------------------------------- 1단계
def stage1_video_norm(frames_hwc_uint8: np.ndarray,
                      target_height: int = EVAL_H,
                      target_width: int = EVAL_W,
                      pad: bool = True) -> torch.Tensor:
    """원본 preprocess_video (36~70행) 그대로. 입력 (T,H,W,3) uint8 -> (C,T,H,W) [-1,1]."""
    video_tensor = torch.from_numpy(frames_hwc_uint8).float().permute(0, 3, 1, 2) / 255.0
    if pad:
        _, _, height, width = video_tensor.shape
        scale = min(target_height / height, target_width / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor, size=(resized_height, resized_width),
            mode="bilinear", align_corners=False,
        )
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left
        video_tensor = F.pad(video_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    else:
        _, _, height, width = video_tensor.shape
        scale = max(target_height / height, target_width / width)
        resized_height = max(target_height, round(height * scale))
        resized_width = max(target_width, round(width * scale))
        video_tensor = F.interpolate(
            video_tensor, size=(resized_height, resized_width),
            mode="bilinear", align_corners=False,
        )
        top = (resized_height - target_height) // 2
        left = (resized_width - target_width) // 2
        video_tensor = video_tensor[:, :, top:top + target_height, left:left + target_width]

    video_tensor = (video_tensor - 0.5) * 2.0
    return video_tensor.permute(1, 0, 2, 3).contiguous()


def stage1_to_uint8(frames_hwc_uint8: np.ndarray,
                    target_height: int = EVAL_H,
                    target_width: int = EVAL_W,
                    pad: bool = True) -> torch.Tensor:
    """원본 to_eval_uint8 (105~108행) 그대로. 반환 (T,H,W,C) uint8.

    ⚠ 이 uint8 왕복이 양자화를 넣는다. 빼면 채점기와 값이 미세하게 달라진다.
    """
    video = stage1_video_norm(frames_hwc_uint8, target_height, target_width, pad).clamp(-1.0, 1.0)
    video = ((video + 1.0) / 2.0 * 255.0).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous()


# ---------------------------------------------------------------- 2단계
def stage2_resize_pad(frames_nchw_uint8: torch.Tensor, size: int,
                      pad_value: float = 0.0) -> torch.Tensor:
    """원본 _resize_pad_frame_batch (238~249행) 그대로. (N,C,H,W) uint8 -> (N,C,size,size) [0,1]."""
    frames = frames_nchw_uint8.float() / 255.0
    _, _, height, width = frames.shape
    scale = min(size / height, size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    frames = F.interpolate(frames, size=(resized_height, resized_width),
                           mode="bilinear", align_corners=False)
    pad_top = (size - resized_height) // 2
    pad_bottom = size - resized_height - pad_top
    pad_left = (size - resized_width) // 2
    pad_right = size - resized_width - pad_left
    return F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)


def content_fraction(h: int, w: int, size: int) -> float:
    """2단계 뒤 실제 내용이 차지하는 면적 비율. 검은 여백이 얼마나 되는지 보려고."""
    scale = min(size / h, size / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    return (rh * rw) / (size * size)


# ---------------------------------------------------------------- 모델 쪽
def model_input_size(model: torch.nn.Module, requested_size: int = 0) -> int:
    """원본 resolve_dino_image_size (252~266행) 그대로.

    requested_size <= 0 이면 model.patch_embed.img_size[0] 을 쓴다.
    채점기는 0 을 넘기므로 DINO 는 518 이 된다 (224 아님).
    """
    expected_size = None
    patch_embed = getattr(model, "patch_embed", None)
    img_size = getattr(patch_embed, "img_size", None)
    if isinstance(img_size, tuple) and img_size:
        expected_size = int(img_size[0])
    elif isinstance(img_size, int):
        expected_size = int(img_size)

    if requested_size <= 0:
        return expected_size or 224
    if expected_size is not None and requested_size != expected_size:
        return expected_size
    return requested_size


def pool_output(output) -> torch.Tensor:
    """원본 _normalize_image_model_output (269~286행) 그대로."""
    if isinstance(output, dict):
        if "x_norm_clstoken" in output:
            output = output["x_norm_clstoken"]
        elif "features" in output:
            output = output["features"]
        else:
            tensor_values = [v for v in output.values() if isinstance(v, torch.Tensor)]
            if not tensor_values:
                raise TypeError("모델이 텐서 없는 dict 를 돌려줬다")
            output = tensor_values[0]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        output = output[:, 0]
    elif output.ndim > 3:
        output = output.flatten(2).mean(dim=-1)
    return output


def normalize_stats(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """모델 자기 정규화 상수를 pretrained_cfg 에서 읽는다.

    왜 ImageNet 고정이 아닌가:
      채점기는 DINO 에 ImageNet 상수를 쓰는데, 그건 dinov2 의 pretrained_cfg 값과
      같아서 맞는 것이다. CLIP 은 (0.4815,0.4578,0.4082), SigLIP 은 (0.5,0.5,0.5)
      이라 ImageNet 을 강제하면 "정규화가 틀렸을 때 얼마나 망가지나" 를 재게 된다.
      우리가 알고 싶은 건 그게 아니라 각 모델의 유사도 판단이다.
    """
    cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return (torch.tensor(mean).view(1, 3, 1, 1),
            torch.tensor(std).view(1, 3, 1, 1))


@torch.no_grad()
def extract_features(frames_nchw_uint8: torch.Tensor, model: torch.nn.Module,
                     device: torch.device, size: int,
                     mean: torch.Tensor, std: torch.Tensor,
                     batch: int = 32) -> torch.Tensor:
    """원본 extract_dino_features (300~316행) 와 같은 순서.

    2단계 리사이즈/패딩 -> 정규화 -> forward -> pool.  배치 32도 원본과 동일.
    """
    mean = mean.to(device)
    std = std.to(device)
    outs = []
    for start in range(0, frames_nchw_uint8.shape[0], batch):
        chunk = frames_nchw_uint8[start:start + batch]
        x = stage2_resize_pad(chunk, size, pad_value=0.0).to(device)
        x = (x - mean) / std
        outs.append(pool_output(model(x)).cpu().float())
    return torch.cat(outs, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """행별 코사인 유사도. 거리는 1 - 유사도 (낮을수록 좋음)."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1)
