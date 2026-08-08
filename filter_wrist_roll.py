"""
에피소드별 wrist_roll (dim 4) 평균 계산 후 threshold 필터링 테스트.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_ROOT = Path(__file__).parent.parent / "data"
TRAIN_ROOT = DATA_ROOT / "train"
EVAL_ACTIONS = DATA_ROOT / "eval" / "actions"

# ── 1. Eval wrist_roll 분포 ─────────────────────────────────────────────────
eval_files = sorted(EVAL_ACTIONS.glob("*.npy"))
eval_wrist = []
for f in eval_files:
    arr = np.load(f)  # (16, 6)
    eval_wrist.append(arr[:, 4].mean())  # dim 4 = wrist_roll
eval_wrist = np.array(eval_wrist)
print(f"[Eval] wrist_roll  mean={eval_wrist.mean():.1f}°  std={eval_wrist.std():.1f}°  min={eval_wrist.min():.1f}°  max={eval_wrist.max():.1f}°")

# ── 2. Train 에피소드별 wrist_roll 평균 ────────────────────────────────────
parquet_files = sorted(TRAIN_ROOT.rglob("*.parquet"))
print(f"\n[Train] parquet 파일 수: {len(parquet_files)}")

rows = []
for pq in parquet_files:
    try:
        df = pd.read_parquet(pq, columns=["observation.state"])
        states = np.stack(df["observation.state"].values).astype(np.float32)
        wrist_mean = states[:, 4].mean()
        rows.append({"path": str(pq), "wrist_roll_mean": wrist_mean})
    except Exception:
        continue

ep_df = pd.DataFrame(rows)
print(f"[Train] 에피소드 수: {len(ep_df)}")
print(f"[Train] wrist_roll  mean={ep_df['wrist_roll_mean'].mean():.1f}°  std={ep_df['wrist_roll_mean'].std():.1f}°")

# ── 3. threshold 별 생존 에피소드 수 ───────────────────────────────────────
print("\n[Threshold 테스트]")
thresholds = [-60, -30, 0, 10, 20, 30, 40, 50]
for t in thresholds:
    kept = (ep_df["wrist_roll_mean"] > t).sum()
    print(f"  wrist_roll > {t:4d}°  →  {kept:5d}개 ({kept/len(ep_df)*100:.1f}%)")

# ── 4. 시각화 ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ep_df["wrist_roll_mean"], bins=80, alpha=0.6, color="steelblue", label="Train episodes")
ax.axvline(eval_wrist.mean(), color="red", linewidth=2, label=f"Eval mean ({eval_wrist.mean():.1f}°)")
ax.axvspan(eval_wrist.min(), eval_wrist.max(), alpha=0.15, color="red", label="Eval range")
for t in [0, 20]:
    ax.axvline(t, color="orange", linewidth=1.5, linestyle="--", label=f"threshold={t}°")
ax.set_xlabel("Episode mean wrist_roll (°)")
ax.set_ylabel("Episode count")
ax.set_title("Train vs Eval: wrist_roll distribution")
ax.legend()
plt.tight_layout()
out = Path(__file__).parent / "cluster_output" / "wrist_roll_filter.png"
plt.savefig(out, dpi=150)
print(f"\n[저장] {out}")
plt.show()

# ── 5. 선택된 에피소드 목록 저장 (threshold=0 기준) ─────────────────────
kept_df = ep_df[ep_df["wrist_roll_mean"] > 0].copy()
kept_df.to_csv(Path(__file__).parent / "cluster_output" / "kept_episodes_wrist0.csv", index=False)
print(f"[저장] kept_episodes_wrist0.csv  ({len(kept_df)}개)")
