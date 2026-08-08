"""홀드아웃 생성 영상을 정답과 직접 비교한다. submission_kit 을 쓰지 않는다.

왜 이렇게 하나:
  킷은 최종 mp4 가 확정된 뒤 제출 CSV 를 만들 때만 쓸 수 있으므로, 설정을 고를 때
  쓸 점수가 없다. 그런데 train 데이터에는 정답 영상이 있다. 학습에서 빠진
  에피소드로 재면 킷 없이 숫자가 나온다.

무엇을 재나:
  recon   프레임별 MSE 평균. 정답과 화면이 얼마나 다른가
  motion  프레임 간 변화량의 MSE. "움직임이 정답만큼 일어났나"
          Action 성분에 대응하는 자체 대리 지표다
  drift   마지막 프레임 MSE / 첫 프레임 MSE. 뒤로 갈수록 벌어지는 정도

주의: 이 숫자와 대회 점수(DINO/Video/Action)가 얼마나 맞는지는 모른다.
      방향을 좁히는 용도이고, 최종 확인은 제출 한 번으로 해야 한다.

사용: evalholdout.py RUN_DIR [RUN_DIR ...]
      각 RUN_DIR 은 sample_XXXXXX.mp4 들이 있는 폴더
"""
import sys
from pathlib import Path
import numpy as np
import torch
import av

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import preprocess_video

HOLD = Path("/workspace/holdout")
H, W = 320, 512


def read_mp4(path):
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    return np.stack(frames)


def to_tensor(video_uint8):
    """정답과 생성본을 같은 공간에 놓는다. 생성본은 이미 320x512 라 패딩이 없다."""
    return preprocess_video(video_uint8, H, W, True)      # (T,3,H,W) float


def label(d):
    d = Path(d)
    return d.parent.name if d.name == "videos" else d.name


def score_run(run_dir):
    run_dir = Path(run_dir)
    # 폴더를 통째로 줘도 되고 videos 까지 줘도 되게
    if not any(run_dir.glob("*.mp4")) and (run_dir / "videos").is_dir():
        run_dir = run_dir / "videos"
    gts = sorted((HOLD / "gt").glob("*.npy"))
    rows = []
    for g in gts:
        sid = g.stem
        mp4 = run_dir / f"{sid}.mp4"
        if not mp4.exists():
            continue
        gt = to_tensor(np.load(g))
        gen = to_tensor(read_mp4(mp4))
        n = min(len(gt), len(gen))
        gt, gen = gt[:n], gen[:n]

        per_frame = ((gt - gen) ** 2).mean(dim=(1, 2, 3))          # (T,)
        recon = per_frame.mean().item()

        # 프레임 간 변화량 - 움직임이 정답만큼 일어났는가
        dgt = (gt[1:] - gt[:-1]).abs()
        dgen = (gen[1:] - gen[:-1]).abs()
        motion = ((dgt - dgen) ** 2).mean().item()

        drift = (per_frame[-1] / per_frame[0]).item() if per_frame[0] > 0 else float("nan")
        rows.append((sid, recon, motion, drift))

    if not rows:
        return None
    r = np.array([[x[1], x[2], x[3]] for x in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "recon": r[:, 0].mean(),
        "motion": r[:, 1].mean(),
        "drift": np.nanmean(r[:, 2]),
        "rows": rows,
    }


if len(sys.argv) < 2:
    raise SystemExit(__doc__)

print(f"{'설정':22s} {'n':>3s} {'recon':>10s} {'motion':>10s} {'drift':>8s}")
print("-" * 58)
results = []
for d in sys.argv[1:]:
    s = score_run(d)
    if s is None:
        print(f"{label(d):22s}  영상 없음")
        continue
    results.append((label(d), s))
    print(f"{label(d):22s} {s['n']:3d} {s['recon']:10.5f} {s['motion']:10.5f} {s['drift']:8.2f}")

if len(results) > 1:
    best = min(results, key=lambda x: x[1]["recon"])
    print(f"\nrecon 최저: {best[0]}  ({best[1]['recon']:.5f})")
    bm = min(results, key=lambda x: x[1]["motion"])
    print(f"motion 최저: {bm[0]}  ({bm[1]['motion']:.5f})")
    if best[0] != bm[0]:
        print("  * 두 지표가 다른 설정을 가리킨다. 어느 쪽이 대회 점수와 맞는지는 제출로 확인해야 한다.")
