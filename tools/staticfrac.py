"""정지 출력 정책의 최적 문턱을 홀드아웃으로 찾는다. GPU 불필요.

핵심: 이미 만든 생성 영상과 정지 영상(첫 프레임 반복)을 각각 정답과 비교해두면
      어떤 문턱을 쓰든 점수를 즉시 계산할 수 있다. 다시 생성할 필요가 없다.

정책: 주어진 액션 시퀀스의 총 변화량이 작은 샘플은 정지 영상을 낸다.
      "명령된 움직임이 거의 없으면 화면도 거의 안 움직여야 한다" 는 규칙이다.
      입력만 보고 판단하며 submission_kit 을 쓰지 않는다.

문턱은 train 홀드아웃으로 정한다. 예전 40% 는 채점기로 정한 값이라 쓸 수 없다.
"""
import sys
from pathlib import Path
import numpy as np
import av

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import preprocess_video as _pv


def preprocess_video(v, h, w, pad):
    # (C,T,H,W) -> (T,C,H,W). 시간축이 앞이어야 프레임 단위 계산이 맞다.
    return _pv(v, h, w, pad).permute(1, 0, 2, 3).contiguous()

HOLD = Path("/workspace/holdout")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/g10s50/videos")
if not any(RUN.glob("*.mp4")) and (RUN / "videos").is_dir():
    RUN = RUN / "videos"
H, W = 320, 512


def read_mp4(p):
    fr = []
    with av.open(str(p)) as c:
        for f in c.decode(c.streams.video[0]):
            fr.append(f.to_ndarray(format="rgb24"))
    return np.stack(fr)


def scores(gt, vid):
    n = min(len(gt), len(vid))
    gt, vid = gt[:n], vid[:n]
    pf = ((gt - vid) ** 2).mean(dim=(1, 2, 3))
    dgt = (gt[1:] - gt[:-1]).abs()
    dv = (vid[1:] - vid[:-1]).abs()
    return pf.mean().item(), ((dgt - dv) ** 2).mean().item()


rows = []
for g in sorted((HOLD / "gt").glob("*.npy")):
    sid = g.stem
    mp4 = RUN / (sid + ".mp4")
    if not mp4.exists():
        continue
    raw = np.load(g)
    gt = preprocess_video(raw, H, W, True)
    gen = preprocess_video(read_mp4(mp4), H, W, True)
    sta = preprocess_video(np.repeat(raw[:1], len(raw), axis=0), H, W, True)

    act = np.load(HOLD / "actions" / (sid + ".npy")).astype(np.float64)
    motion_cmd = float(np.abs(np.diff(act, axis=0)).sum())   # 주어진 액션의 총 변화량

    gr, gm = scores(gt, gen)
    sr, sm = scores(gt, sta)
    rows.append((sid, motion_cmd, gr, gm, sr, sm))

if not rows:
    raise SystemExit("영상을 못 찾았다")

rows.sort(key=lambda r: r[1])          # 덜 움직이는 것부터
n = len(rows)
print("홀드아웃 %d개, 생성본 = %s" % (n, RUN))
print("")
print("%8s %5s %10s %10s" % ("정지비율", "개수", "recon", "motion"))
print("-" * 38)

best = None
table = []
for k in range(0, n + 1):
    recon = float(np.mean([r[4] if i < k else r[2] for i, r in enumerate(rows)]))
    mot = float(np.mean([r[5] if i < k else r[3] for i, r in enumerate(rows)]))
    table.append((k, recon, mot))
    if best is None or recon < best[1]:
        best = (k, recon, mot)

step = max(1, n // 10)
shown = {0, n, best[0]} | set(range(0, n + 1, step))
for k, recon, mot in table:
    if k in shown:
        mark = "   <- 최적" if k == best[0] else ""
        print("%7.0f%% %5d %10.5f %10.5f%s" % (100 * k / n, k, recon, mot, mark))

k, br, bm = best
print("")
print("최적: 정지 %.0f%% (%d/%d)   recon %.5f   motion %.5f" % (100 * k / n, k, n, br, bm))
print("전부 생성:  recon %.5f   motion %.5f" % (table[0][1], table[0][2]))
print("전부 정지:  recon %.5f   motion %.5f" % (table[n][1], table[n][2]))
if k:
    print("")
    print("문턱값: 액션 총변화량 < %.4f 이면 정지" % rows[k - 1][1])
    print("개선폭: recon %.5f -> %.5f  (%.1f%%)" % (table[0][1], br, 100 * (table[0][1] - br) / table[0][1]))
