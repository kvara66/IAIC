# 규정 확인 완료 (2026-08-06) — submission_kit 을 사용하지 않으며 선택·보정도 하지 않는다.
"""시프트가 의도대로 동작하는지 확인한다.

확인할 것:
  (1) action_shift=0 이면 기존과 비트 단위로 같은가  (기존 실험이 재현 가능해야 한다)
  (2) action_shift=1 이면 정말 [a0, a0, a1, ..., a14] 인가
  (3) 학습 경로와 추론 경로가 같은 식을 쓰는가  (어긋나면 조건이 통째로 무의미해진다)
"""
import sys
from pathlib import Path
import numpy as np
import torch

K = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(K / "src"))
sys.path.insert(0, str(K / "scripts"))
from eval.feature_csv_utils import build_inference_batch, load_action_stats

ROOT = Path("/workspace/open/data/eval")
ids = ["sample_000000", "sample_000001"]
mean, std = load_action_stats("/workspace/open/data/train/so100_action_statistics.json")
dev = torch.device("cpu")

def run(shift):
    return build_inference_batch(ROOT, ids, 320, 512, True, 6, mean, std, dev,
                                 False, False, None, shift)["act"]

a0 = run(0)
a1 = run(1)
raw = np.load(ROOT / "actions" / (ids[0] + ".npy")).astype(np.float32)

print("=== (1) shift=0 이 기존과 같은가 ===")
n = ((a0[0] * std + mean).numpy() - raw)
print(f"  정규화를 되돌린 값과 원본 액션의 최대 차이 = {np.abs(n).max():.3e}")
print("  >>> " + ("통과: 0 이면 원본 그대로다." if np.abs(n).max() < 1e-3 else "실패"))

print()
print("=== (2) shift=1 이 [a0, a0, a1, ...] 인가 ===")
ok = torch.allclose(a1[0][1:], a0[0][:-1], atol=1e-6) and torch.allclose(a1[0][0], a0[0][0], atol=1e-6)
print(f"  a1[1:] == a0[:-1] 이고 a1[0] == a0[0] : {ok}")
print(f"  첫 관절 원본  : {np.array2string(raw[:5,0], precision=2)}")
print(f"  첫 관절 shift1: {np.array2string((a1[0]*std+mean).numpy()[:5,0], precision=2)}")
print("  >>> " + ("통과: 한 칸 밀렸다." if ok else "실패: 밀린 모양이 다르다."))

print()
print("=== (3) 학습/추론이 같은 식을 쓰는가 ===")
import re
ds = (K / "src/ldwma/datasets/lerobot_so100.py").read_text(encoding="utf-8")
inf = (K / "scripts/eval/feature_csv_utils.py").read_text(encoding="utf-8")
def norm(s):
    m = re.search(r"np\.concatenate\(\[np\.repeat\((\w+)\[:1\],\s*(\w+),\s*axis=0\),\s*\1\[:-\2\]\], axis=0\)",
                  s.replace("\n", " "))
    return bool(m)
print(f"  학습 경로 식 일치: {norm(ds)}")
print(f"  추론 경로 식 일치: {norm(inf)}")
print("  >>> " + ("통과: 같은 식이다." if norm(ds) and norm(inf) else "실패: 식이 다르다. 반드시 맞춰야 한다."))
