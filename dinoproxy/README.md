# DINO 대리 모델 찾기

**목표** — 채점기의 DINO가 두 이미지의 닮음을 판단하는 방식과 **가장 비슷하게 판단하는 다른 모델**을 찾는다. 찾은 모델은 이후 학습 로스로 쓴다.

```
기준   vit_small_patch14_dinov2.lvd142m      518×518, patch14, 384차원, Apache-2.0
후보   CLIP ViT-L/14 · SigLIP ViT-B/16 · MAE ViT-B/16 · CLIP ViT-B/16
```

---

## 먼저 알아야 할 사실 세 가지

### ① 채점기의 DINO는 킷 내부 체크포인트가 아니다

`submission_kit/feature_csv_utils.py` 373~374행:

```python
dino_model_name = "vit_small_patch14_dinov2.lvd142m"
dino_model = load_dino_model(device, dino_model_name, pretrained=True)
#   -> timm.create_model(model_name, pretrained=True, num_classes=0)
```

**timm에서 공개 가중치를 받아 씁니다. 라이선스 Apache-2.0.** 킷 안에 실제로 들어 있는 체크포인트는 `action_extractor.ckpt` 하나뿐입니다.

### ② 입력 크기는 224가 아니라 518이다

```python
dino_image_size = resolve_dino_image_size(model, requested_size=0)
#   requested_size <= 0 이면 model.patch_embed.img_size[0] 을 반환
```

HuggingFace 모델 카드: **"Image size: 518 x 518"**, 출력 `(1, 1370, 384)`.

### ③ 리사이즈가 두 번 일어나고, DINO가 보는 화면의 절반이 검은색이다

두 단계 모두 **비율 유지 + 중앙 정렬 + `value=0.0` 검은 패딩**입니다.

```
480×640 원본 ──1단계──▶ 320×427 + 좌우 여백 42/43  = 320×512
                                   가로 중 내용 83.4%

320×512     ──2단계──▶ 324×518 + 상하 여백 97/97   = 518×518
                                   세로 중 내용 62.5%

최종 내용 면적 = 0.834 × 0.625 = 52%      나머지 48%가 검은색
```

**후보 모델도 반드시 같은 기하 처리를 받아야 합니다.** 안 그러면 다른 그림을 비교하는 셈이라 상관계수가 무의미해집니다.

⚠ 2단계는 `[0,1]` 공간에서 0으로 패딩한 **뒤에** 정규화합니다. 순서를 바꾸면 검은 여백이 `(0-mean)/std`가 아니라 `0`이 되어 값이 달라집니다.

---

## 파일

| 파일 | 하는 일 |
|---|---|
| `common.py` | 채점기 전처리를 그대로 옮긴 것 + 후보 모델용 일반화 |
| `verify.py` | **재구현이 채점기와 수치적으로 같은지 증명** (오차 0) |
| `mkpairs.py` | 정답 300 + 예측 300 = 300쌍을 320×512 PNG로 저장 |
| `extract.py` | 모델별 코사인 유사도 CSV 생성 |
| `corr.py` | DINO CSV와 대조해 상관관계 순위 |

---

## 실행 순서

```bash
cd /workspace/dinoproxy
source /workspace/env.sh

# 0) 재구현이 채점기와 같은지 먼저 증명한다. 여기서 실패하면 나머지는 무의미하다.
python verify.py --kit /workspace/open/submission_kit --model

# 1) 300쌍 만들기 (전부 무작위)
python mkpairs.py --out /workspace/dinopairs --train-root /workspace/open/data/train

# 2) 모델 5개 특징 추출 + 코사인 유사도
python extract.py --pairs /workspace/dinopairs --device cuda --determinism

# 3) 상관관계 순위
python corr.py --pairs /workspace/dinopairs
```

---

## 트리플 체크 — 무엇이 자동으로 검증되나

이 파이프라인은 **틀렸을 때 조용히 넘어가지 않도록** 만들었습니다. 검증이 실패하면 스크립트가 그 자리에서 멈춥니다.

| 층 | 무엇을 확인하나 | 어디서 |
|---|---|---|
| 1 | 재구현 함수와 채점기 원본 함수의 출력 차이가 **0** | `verify.py` A~F |
| 2 | DINO 실물 특징이 채점기 경로와 코사인 유사도 1.0 | `verify.py` G |
| 3 | 같은 그림끼리는 코사인 거리 **0** (모든 모델) | `extract.py`, `corr.py` |
| 4 | 특징에 NaN/Inf 없음 | `extract.py` |
| 5 | 두 번 돌려 같은 값 (`--determinism`) | `extract.py` |
| 6 | `gt/`와 `pred/`의 pair_id가 정확히 일치 | `extract.py` |
| 7 | 모든 PNG가 320×512 | `extract.py` |
| 8 | DINO 대 DINO 상관계수가 정확히 **1.0** | `corr.py` |
| 9 | 거리 분포 범위가 충분히 넓은가 (좁으면 불안정) | `corr.py` |
| 10 | 봉우리가 둘인가 (그러면 상관계수가 부풀려짐) | `corr.py` |

**10번이 특히 중요합니다.** 무작위로 짝지으면 우연히 같은 에피소드에서 걸린 쌍이 0.99에, 나머지가 0.4에 뭉쳐 봉우리가 둘이 됩니다. 그러면 **두 덩어리를 갈라놓기만 해도 상관계수가 0.95가 나와서** 모델 순위가 실제 판단력과 무관해집니다. 경고가 뜨면 전체 수치 대신 **구간별 상관계수**를 보세요.

---

## 결과 읽는 법

`corr.py`가 세 가지를 냅니다.

**전체 상관계수** — Pearson과 Spearman. 로스로 쓸 거면 값 자체가 따라가야 하므로 Pearson이 중요하고, 순위만 맞으면 되면 Spearman입니다. **둘이 크게 다르면 관계가 비선형**이라는 뜻이고, 로스로 쓸 때 문제가 됩니다.

**쌍 종류별** — 무작위 쌍과 검산쌍을 나눠서 봅니다.

**DINO 거리 사분위 구간별** — Q1(가장 닮은 쌍)부터 Q4(가장 다른 쌍)까지. **전체 상관계수가 높아도 구간별로 순위가 뒤집히면 그 모델은 특정 구간에서만 맞는 것입니다.** 예측-정답 비교는 Q1 구간에서 일어나므로, 실제로 쓸 거면 Q1 성적을 같이 보셔야 합니다.

---

## 설계상 정한 것

**각 모델은 자기 정규화 상수를 씁니다** (`pretrained_cfg`에서 읽음). 채점기가 DINO에 ImageNet 상수를 쓰는 건 dinov2의 `pretrained_cfg` 값과 같아서 맞는 것이고, CLIP은 `(0.4815, 0.4578, 0.4082)`, SigLIP은 `(0.5, 0.5, 0.5)`입니다. ImageNet을 강제하면 "정규화가 틀렸을 때 얼마나 망가지나"를 재게 되는데, 우리가 알고 싶은 건 각 모델의 **유사도 판단**입니다.

**기하 처리는 모두 동일합니다** — 비율 유지 리사이즈 + 중앙 정렬 + 검은 패딩. 크기만 각 모델 것을 씁니다(DINO 518, 나머지 224).

**특징은 `num_classes=0`의 풀링 출력**입니다. 채점기가 DINO에 쓰는 것과 같은 인터페이스라, 모델 간 비교가 일관됩니다. CLIP의 경우 투영 전 특징일 수 있으니 `extract.py`가 찍는 **특징 차원**으로 확인하세요(ViT-L/14 투영 전 1024, 투영 후 768).

---

## 규정

- DINO를 포함한 후보 전부 **공개 사전학습 모델**입니다. 규정이 명시적으로 허용합니다 — *"공식적으로 가중치가 공개된 사전학습모델 중 ... MIT, Apache 2.0, CC BY, CC BY-NC 등"*
- `verify.py`는 `submission_kit`을 **읽기만** 합니다. 고치지 않습니다.
- **eval 데이터를 쓰지 않습니다.** 쌍은 전부 train에서 뽑습니다.
- 이 파이프라인은 **영상 선택·보정에 관여하지 않습니다.** 학습 로스를 고르기 위한 사전 조사입니다.
