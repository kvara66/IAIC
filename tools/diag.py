"""모델이 왜 정지보다 못한지 진단한다. GPU 불필요.

세 가지를 본다.
  1) 움직임 크기 - 모델이 정답보다 많이 움직이나 적게 움직이나
  2) 샘플별 승패 - 명령 움직임이 큰 샘플에서는 생성이 정지를 이기나
  3) 액션 추종 - 명령된 움직임과 화면 움직임의 상관.
     정답에서 잰 값이 달성 가능한 상한이고, 생성본에서 잰 값이 우리가 잡은 몫이다.

3번이 핵심이다. 상관이 0 근처면 모델이 액션을 무시하는 것이고,
정답 상관에 가까우면 이미 따라가고 있으니 다른 곳을 봐야 한다.
"""
import sys
from pathlib import Path
import numpy as np
import av

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
from ldwma.datasets.lerobot_so100 import preprocess_video as _pv


def preprocess_video(v, h, w, pad):
    return _pv(v, h, w, pad).permute(1, 0, 2, 3).contiguous()      # (T,C,H,W)


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


def step_motion(v):
    """프레임별 화면 변화량 (T-1,)"""
    return (v[1:] - v[:-1]).abs().mean(dim=(1, 2, 3)).numpy()


def corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


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
    n = min(len(gt), len(gen)); gt, gen, sta = gt[:n], gen[:n], sta[:n]

    act = np.load(HOLD / "actions" / (sid + ".npy")).astype(np.float64)
    cmd = np.abs(np.diff(act, axis=0)).sum(axis=1)[:n - 1]      # 프레임별 명령 변화량

    mg, mn = step_motion(gt), step_motion(gen)
    rec_gen = ((gt - gen) ** 2).mean().item()
    rec_sta = ((gt - sta) ** 2).mean().item()

    rows.append(dict(sid=sid, cmd_total=float(cmd.sum()),
                     mot_gt=float(mg.mean()), mot_gen=float(mn.mean()),
                     corr_gt=corr(cmd, mg), corr_gen=corr(cmd, mn),
                     corr_gtgen=corr(mg, mn),
                     rec_gen=rec_gen, rec_sta=rec_sta))

if not rows:
    raise SystemExit("영상을 못 찾았다")
n = len(rows)
A = lambda k: np.array([r[k] for r in rows], dtype=np.float64)

print("홀드아웃 %d개, 생성본 = %s" % (n, RUN))
print("")
print("== 1) 움직임 크기 ==")
mg, mn = A("mot_gt"), A("mot_gen")
print("  정답 화면 변화량   평균 %.5f" % mg.mean())
print("  생성 화면 변화량   평균 %.5f   (정답 대비 %.2f배)" % (mn.mean(), mn.mean() / mg.mean()))
over = int((mn > mg).sum())
print("  정답보다 많이 움직인 샘플 %d/%d" % (over, n))

print("")
print("== 2) 샘플별 승패 (생성 vs 정지) ==")
rg, rs = A("rec_gen"), A("rec_sta")
win = rg < rs
print("  생성이 이긴 샘플 %d/%d" % (int(win.sum()), n))
order = np.argsort(A("cmd_total"))
half = n // 2
lo, hi = order[:half], order[half:]
print("  명령 움직임 하위 절반: 생성 승 %d/%d" % (int(win[lo].sum()), len(lo)))
print("  명령 움직임 상위 절반: 생성 승 %d/%d" % (int(win[hi].sum()), len(hi)))

print("")
print("== 3) 액션 추종 (명령 변화량 vs 화면 변화량 상관) ==")
cg, cn, cc = A("corr_gt"), A("corr_gen"), A("corr_gtgen")
print("  정답 영상   상관 %.3f   <- 달성 가능한 상한" % np.nanmean(cg))
print("  생성 영상   상관 %.3f   <- 우리가 잡은 몫" % np.nanmean(cn))
if np.nanmean(cg) > 0.01:
    print("  달성률      %.0f%%" % (100 * np.nanmean(cn) / np.nanmean(cg)))
print("  정답-생성 움직임 상관 %.3f" % np.nanmean(cc))
print("")
print("  상관이 양수인 샘플: 정답 %d/%d, 생성 %d/%d"
      % (int(np.nansum(cg > 0)), n, int(np.nansum(cn > 0)), n))
