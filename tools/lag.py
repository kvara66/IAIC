"""action_shift 가 값을 할지 학습 전에 확인한다. GPU 불필요.

가설:
  실제 로봇은 구동 지연이 있어 화면이 명령보다 한 프레임 늦게 반응한다.
  action_embed 가 프레임마다 독립이라 모델이 스스로 못 당겨오므로,
  액션을 한 칸 밀어서 넣으면 맞춰질 것이다.

검증 방법:
  명령 변화량과 화면 변화량의 상관을 지연 -2..+2 에서 각각 재고 최고점을 찾는다.
    정답 영상의 최고점  = 실제 시스템의 지연
    생성 영상의 최고점  = 우리 모델이 학습한 지연
  둘이 다르면 그 차이만큼 밀어야 한다. 같으면 action_shift 는 값이 없다.

lag k 의 뜻: 화면 변화[t] 를 명령 변화[t-k] 와 맞춰본다.
             k=+1 이면 화면이 명령보다 한 프레임 늦다는 뜻이다.
"""
import sys
from pathlib import Path
import numpy as np
import av

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import preprocess_video as _pv


def preprocess_video(v, h, w, pad):
    return _pv(v, h, w, pad).permute(1, 0, 2, 3).contiguous()


HOLD = Path("/workspace/holdout")
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/g10s50/videos")
if not any(RUN.glob("*.mp4")) and (RUN / "videos").is_dir():
    RUN = RUN / "videos"
H, W = 320, 512
LAGS = [-2, -1, 0, 1, 2]


def read_mp4(p):
    fr = []
    with av.open(str(p)) as c:
        for f in c.decode(c.streams.video[0]):
            fr.append(f.to_ndarray(format="rgb24"))
    return np.stack(fr)


def step_motion(v):
    return (v[1:] - v[:-1]).abs().mean(dim=(1, 2, 3)).numpy()


def lag_corr(cmd, mot, k):
    """화면 변화[t] 와 명령 변화[t-k] 의 상관"""
    if k >= 0:
        a, b = cmd[: len(cmd) - k] if k else cmd, mot[k:]
    else:
        a, b = cmd[-k:], mot[: len(mot) + k]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    if m < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


gt_rows, gen_rows = [], []
for g in sorted((HOLD / "gt").glob("*.npy")):
    sid = g.stem
    mp4 = RUN / (sid + ".mp4")
    if not mp4.exists():
        continue
    raw = np.load(g)
    gt = preprocess_video(raw, H, W, True)
    gen = preprocess_video(read_mp4(mp4), H, W, True)
    n = min(len(gt), len(gen)); gt, gen = gt[:n], gen[:n]

    act = np.load(HOLD / "actions" / (sid + ".npy")).astype(np.float64)
    cmd = np.abs(np.diff(act, axis=0)).sum(axis=1)[: n - 1]
    mg, mn = step_motion(gt), step_motion(gen)
    gt_rows.append([lag_corr(cmd, mg, k) for k in LAGS])
    gen_rows.append([lag_corr(cmd, mn, k) for k in LAGS])

if not gt_rows:
    raise SystemExit("영상을 못 찾았다")

G = np.array(gt_rows, dtype=np.float64)
N = np.array(gen_rows, dtype=np.float64)
print("홀드아웃 %d개, 생성본 = %s" % (len(gt_rows), RUN))
print("")
print("%6s %12s %12s" % ("lag", "정답 영상", "생성 영상"))
print("-" * 32)
gm = np.nanmean(G, axis=0)
nm = np.nanmean(N, axis=0)
for i, k in enumerate(LAGS):
    m1 = " <-" if i == int(np.nanargmax(gm)) else "  "
    m2 = " <-" if i == int(np.nanargmax(nm)) else "  "
    print("%+6d %10.3f%s %10.3f%s" % (k, gm[i], m1, nm[i], m2))

bg, bn = LAGS[int(np.nanargmax(gm))], LAGS[int(np.nanargmax(nm))]
print("")
print("정답 최고점 lag %+d   생성 최고점 lag %+d" % (bg, bn))
print("")
if bg == bn:
    print("-> 모델이 이미 정답과 같은 지연을 학습했다. action_shift 는 값이 없다.")
else:
    print("-> 차이 %+d 프레임. action_shift=%d 로 학습하면 이 간극이 메워질 수 있다." % (bg - bn, bg - bn))
print("")
print("참고: 정답 최고 상관 %.3f, 생성 최고 상관 %.3f (달성률 %.0f%%)"
      % (gm[np.nanargmax(gm)], nm[np.nanargmax(nm)], 100 * nm[np.nanargmax(nm)] / gm[np.nanargmax(gm)]))
