"""중심화된 액션의 정규화 통계를 만든다.

왜 필요한가:
  action_center 를 켜면 액션에서 첫 프레임 값을 뺀다. 그러면 값의 크기가
  훨씬 작아지는데, 기존 통계(절대값 기준 std)로 나누면 정규화 후 신호가
  원래보다 약해진다. 액션 조건화를 강화하려는 목적과 정반대가 된다.

  예: 차원1 절대값 std 50.19 인데 중심화하면 std 가 한 자릿수로 떨어진다.
      절대값 std 로 나누면 정규화 값이 0.1 대에 머문다.

무엇을 하나:
  학습 분할(train=True) 에피소드에서 16프레임 창을 겹치지 않게 잘라
  각 창을 첫 프레임 기준으로 중심화한 뒤 차원별 mean/std 를 낸다.
  학습 때 데이터셋이 하는 것과 같은 방식이다.

  검증 분할(val_fraction=0.05, seed=0)은 제외한다 - 학습 통계이므로.

출력은 원본과 같은 형식이라 --action-stats-path 로 그대로 넘길 수 있다.
  {"mean": [...6개...], "std": [...6개...]}

사용:
  python mkcenterstats.py [출력경로] [에피소드 상한]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import LeRobotSO100Dataset

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/workspace/open/data/train/so100_action_statistics_centered.json")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 이면 전부
TRAJ = 16

ds = LeRobotSO100Dataset(
    root="/workspace/open/data/train",
    dataset_paths="auto",
    train=True,                      # 학습 분할만. 검증은 제외
    traj_len=TRAJ,
    target_height=320, target_width=512, pad=True,
    camera_key="auto",
    val_fraction=0.05, seed=0,       # 학습 설정과 동일해야 한다
    downsample=1, use_language=False,
)
examples = ds.examples if not LIMIT else ds.examples[:LIMIT]
print("학습 에피소드 %d개 (전체 %d개)" % (len(examples), len(ds.examples)))

# Welford 로 누적한다. 전체를 메모리에 올리지 않는다.
n = 0
mean = np.zeros(6, dtype=np.float64)
m2 = np.zeros(6, dtype=np.float64)
skipped = 0

for i, ex in enumerate(examples):
    try:
        with ds._materialize_file(ex["data_ref"]) as p:
            a = np.stack(pd.read_parquet(p, columns=["action"])["action"].to_numpy()).astype(np.float64)
    except Exception:
        skipped += 1
        continue
    if a.shape[-1] != 6 or len(a) < TRAJ:
        skipped += 1
        continue

    # 겹치지 않는 16프레임 창을 잘라 각각 첫 프레임으로 중심화
    for s in range(0, len(a) - TRAJ + 1, TRAJ):
        c = a[s:s + TRAJ] - a[s:s + 1]
        for row in c:
            n += 1
            d = row - mean
            mean += d / n
            m2 += d * (row - mean)

    if (i + 1) % 1000 == 0:
        print("  %d/%d  누적 %d 프레임" % (i + 1, len(examples), n))

if n < 2:
    raise SystemExit("[실패] 표본이 없다")

std = np.sqrt(m2 / (n - 1))
# std 가 0 이면 정규화에서 0 나누기가 난다. 안전값으로 막는다.
std = np.where(std < 1e-6, 1.0, std)

print("")
print("표본 %d 프레임, 건너뜀 %d 에피소드" % (n, skipped))
print("중심화 mean", np.round(mean, 4))
print("중심화 std ", np.round(std, 4))

orig = Path("/workspace/open/data/train/so100_action_statistics.json")
if orig.exists():
    o = json.load(open(orig))
    print("")
    print("절대값 std (참고)", np.round(np.array(o["std"], dtype=np.float64), 2))
    ratio = np.array(o["std"], dtype=np.float64) / std
    print("절대값std / 중심화std =", np.round(ratio, 1), " <- 이 배수만큼 신호가 커진다")

OUT.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=1), encoding="utf-8")
print("")
print("저장 %s" % OUT)
print("추론·학습 모두 --action-stats-path 또는 action_stats_path 를 이 파일로 지정할 것")
