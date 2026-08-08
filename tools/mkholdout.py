"""학습에 쓰이지 않은 검증 에피소드로 홀드아웃 세트를 만든다.

왜 필요한가:
  submission_kit 은 최종 mp4 가 확정된 뒤 제출 CSV 를 만들 때만 쓸 수 있다.
  그래서 설정을 비교할 때 쓸 점수가 없다. 그런데 train 데이터에는 정답 영상이
  있으므로, 학습에서 빠진 에피소드로 직접 재면 킷 없이 숫자를 얻을 수 있다.

핵심 두 가지:
  1) 학습에 안 쓰인 것만 쓴다. 데이터셋이 seed=0 으로 섞은 뒤 앞의 val_fraction 을
     검증으로 떼는데, train=False 로 인스턴스를 만들면 정확히 그 집합이 나온다.
  2) 시작 프레임을 고정한다. __getitem__ 은 매번 rng.randint 로 시작을 다시 뽑아
     같은 인덱스라도 다른 구간이 나온다. 홀드아웃은 항상 같아야 하므로 직접 정한다.

출력은 eval 과 같은 형식이라 추론 스크립트를 한 줄도 안 고치고 쓸 수 있다.
  holdout/images/sample_XXXXXX.png    첫 프레임 (원본 해상도)
  holdout/actions/sample_XXXXXX.npy   (16, 6) 정규화 전 액션
  holdout/gt/sample_XXXXXX.npy        (16, H, W, 3) uint8 정답 프레임
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset, _decode_video_clip

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/holdout")
TRAJ = 16

ds = LeRobotSO100Dataset(
    root="/workspace/open/data/train",
    dataset_paths="auto",
    train=False,                 # 검증 분할 = 학습에서 빠진 에피소드
    traj_len=TRAJ,
    target_height=320, target_width=512, pad=True,
    camera_key="auto",
    val_fraction=0.05, seed=0,   # 학습 설정과 동일해야 한다
    downsample=1, use_language=False,
)
print(f"검증 분할 에피소드 {len(ds.examples)}개 (학습에는 안 쓰인 것)")
if len(ds.examples) < N:
    print(f"  [경고] 요청 {N}개보다 적다. {len(ds.examples)}개로 진행")
    N = len(ds.examples)

for d in ("images", "actions", "gt"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

meta = []
made = 0
for i, ex in enumerate(ds.examples):
    if made >= N:
        break
    # 시작 프레임 고정 - 에피소드 중앙에서 뽑는다. 초반은 정지 구간이 많아
    # 어떤 설정을 써도 비슷하게 나와서 비교가 안 된다.
    max_start = ex["length"] - TRAJ
    if max_start < 0:
        continue
    start = max_start // 2
    idx = list(range(start, start + TRAJ))

    try:
        with ds._materialize_file(ex["data_ref"]) as p:
            table = pd.read_parquet(p, columns=["action"])
        actions = np.stack(table["action"].iloc[idx].to_numpy()).astype(np.float32)
        with ds._materialize_file(ex["video_ref"]) as p:
            video = _decode_video_clip(p, idx)          # (16, H, W, 3) uint8
    except Exception as e:
        print(f"  건너뜀 {ex.get('data_ref')}: {type(e).__name__} {e}")
        continue

    sid = f"sample_{made:06d}"
    Image.fromarray(video[0]).save(OUT / "images" / f"{sid}.png")
    np.save(OUT / "actions" / f"{sid}.npy", actions)
    np.save(OUT / "gt" / f"{sid}.npy", video)
    meta.append({"sample_id": sid, "data_ref": str(ex["data_ref"]),
                 "start": int(start), "length": int(ex["length"]),
                 "task": ex.get("task", ""), "fps": int(ex.get("fps", 0))})
    made += 1
    if made % 5 == 0:
        print(f"  {made}/{N}")

(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n완료 {made}개 -> {OUT}")
print(f"  영상 크기 {video.shape[1]}x{video.shape[2]}, 액션 {actions.shape}")
