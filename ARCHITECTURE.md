# v3 모델 아키텍처 & 학습 구조

## 태스크 정의

```
(frame[i], state[i], state[i+1])  →  frame[i+1]
```

- 입력: 현재 프레임 + 현재/다음 로봇 상태
- 출력: 다음 프레임 (이미지 회귀)
- 추론 시: 15스텝 autoregressive chaining → 16프레임 영상

---

## 파이프라인 흐름

```
frame[i]  (B, 3, 320, 512)
    ↓  SDXL VAE encode (frozen)
latent[i]  (B, 4, 40, 64)
    ↓
    +──→ LatentResidualUNet ──→ delta_latent (B, 4, 40, 64)
    |          ↑ FiLM cond
    |
    |   [state[i] (6D), state[i+1] (6D)]  →  ActionMLP  →  cond (B, 256)
    |
pred_latent = latent[i] + delta_latent
    ↓  SDXL VAE decode (frozen)
frame[i+1]_pred  (B, 3, 320, 512)  in [-1, 1]
```

---

## 컴포넌트별 상세

### 1. VAE (`model.py: VAEWrapper`)

- 모델: `stabilityai/sdxl-vae`
- **scale factor: 0.13025** (SD1.5는 0.18215 — 혼동 주의!)
- encode: `x → vae.encode(x).latent_dist.mean * 0.13025`
- decode: `z → vae.decode(z / 0.13025).sample`
- `requires_grad_(False)` — 완전 frozen

### 2. ActionMLP (`model.py: ActionMLP`)

- 입력: `torch.cat([state_i, state_i1], dim=-1)` → (B, 12)
- 구조: Linear(12→128) → SiLU → Linear(128→256) → SiLU
- 출력: cond (B, 256)
- 입력 state는 **z-score 정규화된 값** (dataset에서 나오는 그대로)

### 3. LatentResidualUNet (`model.py: LatentResidualUNet`)

채널: 4 → 64 → 128 → 256(mid) → 128 → 64 → 4

```
enc_in: Conv2d(4, 64)
enc0: ResBlock(64→64) + FiLM
down0: Conv2d(64→128, stride=2)   [40×64 → 20×32]
enc1: ResBlock(128→128) + FiLM
down1: Conv2d(128→256, stride=2)  [20×32 → 10×16]

mid0: ResBlock(256→256) + FiLM
attn: AttentionBlock(256)          [self-attention, no heads split]
mid1: ResBlock(256→256) + FiLM

up1: ConvTranspose2d(256→128, stride=2)  [10×16 → 20×32]
dec1: ResBlock(256→128) + FiLM          [skip-cat from enc1]
up0: ConvTranspose2d(128→64, stride=2)  [20×32 → 40×64]
dec0: ResBlock(128→64) + FiLM           [skip-cat from enc0]

out: GroupNorm → SiLU → Conv2d(64→4)    [zero-init weight & bias]
```

**핵심**: out layer는 zero-init → 학습 초기 delta=0 (입력 그대로 통과)

### 4. FiLM (`model.py: FiLM`)

```python
gamma, beta = Linear(256, channels*2)(cond).chunk(2, dim=-1)
output = x * (1 + gamma) + beta
```

---

## 데이터셋 (`dataset.py`)

### EpisodeIndex

- train 데이터 전체 scan → `(dataset_root, ep_idx, n_frames)` 리스트
- `episode_index.pkl`로 캐시 저장/로드

### LeRobotSequenceDataset

- 에피소드를 n_frames 비례 가중치로 랜덤 샘플
- 에피소드 내 랜덤 위치에서 16프레임 연속 추출
- 반환:
  - `frames`: (16, 3, 320, 512) float32, [-1, 1]
  - `states`: (16, 6) float32, z-score 정규화

### State 정규화

```
normalized = (raw_state - mean) / std
mean = [3.0, 117.6, 109.8, 61.7, -29.4, 9.8]
std  = [26.4, 50.2, 46.6, 29.7, 62.5, 16.4]
```

파일: `../data/train/so100_action_statistics.json`

### 데이터 형식

- 상태: parquet 파일, `observation.state` 컬럼 (6D float)
- 영상: mp4 파일, `observation.images.image` 키
- 경로: `meta/info.json`의 `data_path`, `video_path` 포맷 문자열 사용

---

## Loss (`loss.py`)

```
L = 0.3 × DINO + 0.3 × R3D + 0.4 × Action_MAE
```

### DINOLoss

- 모델: `vit_small_patch14_dinov2.lvd142m` (timm)
- 입력 크기: 518×518 (bilinear resize)
- 정규화: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Loss: `1 - cosine_similarity(pred_feat, gt_feat)`
- B×15 프레임을 (B*15, 3, H, W)로 reshape해서 한번에 처리

### R3DLoss

- 모델: `r3d_18` (torchvision, pretrained)
- 입력: (B, 3, T, H, W) in [0, 1]
- B×16 전체 영상에서 feature 추출
- Loss: `1 - cosine_similarity(pred_feat, gt_feat)`

### ActionMAELoss

- 모델: submission_kit ckpt 임시 사용 (팀원 가중치 수령 후 교체)
- 입력: pred_video (B, T, 3, H, W) → permute → (B, 3, T, H, W) 로 extractor 통과
- 출력: pred_actions (B, T, 6) normalized 공간
- gt: dataset에서 나온 normalized states 그대로 비교
- Loss: 6차원 각각 L1, 합산

---

## 학습 (`train.py`)

### Teacher Forcing Rollout

```python
for t in range(15):
    pred_frame, _ = model(frames[:, t], states[:, t], states[:, t+1])
pred_frames  # (B, 15, 3, H, W)
pred_video = cat([frames[:, :1], pred_frames], dim=1)  # (B, 16, 3, H, W)
```

### 실행 명령

```bash
python train.py \
  --train-root ../data/train \
  --output-dir runs/v1 \
  --batch-size 4 \
  --num-workers 4 \
  --amp
```

### 체크포인트 저장 내용

- `unet` state_dict
- `action_mlp` state_dict
- `optimizer` state_dict
- `scaler` state_dict (AMP)
- `step`, `best_loss`

---

## 추론 (미구현 — infer.py 작성 필요)

```
eval 샘플 로드 (초기 프레임 + 16개 state)
  ↓
frame[0] + states[0→1] → model → frame[1]
frame[1] + states[1→2] → model → frame[2]
...
frame[14] + states[14→15] → model → frame[15]
  ↓
16프레임 → mp4 저장 (216 샘플)
```

---

## 파일 구조

```
v3/
├── dataset.py          # EpisodeIndex, LeRobotSequenceDataset
├── model.py            # VAEWrapper, ActionMLP, LatentResidualUNet, ImageEditingModel
├── loss.py             # DINOLoss, R3DLoss, ActionMAELoss
├── train.py            # teacher forcing 학습 루프
├── test_dataset.py     # dataset sanity check
├── test_loss.py        # DINO loss sanity check
├── ARCHITECTURE.md     # 이 파일
└── runs/               # 학습 체크포인트 (gitignore)
    └── v1/
        ├── tb/         # TensorBoard 로그
        ├── best.ckpt
        ├── stepN.ckpt
        └── epochN.ckpt
```
