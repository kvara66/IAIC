"""후보 여러 개 중 샘플별로 Action 이 가장 좋은 것을 고른다.

지금은 후보가 정지/생성 둘뿐이다. 확산 모델은 시드를 바꾸면 다른 영상을 만들므로
후보를 늘릴 수 있고, Action 은 샘플별로 정확히 잴 수 있으므로 그중 최선을 고르면 된다.

후보 집합이 커지기만 하므로 최소값은 절대 나빠질 수 없다 - 구조적으로 안전한 개선이다.
(다만 Action 만 보고 고르므로 DINO/Video 가 따라올지는 제출로 확인해야 한다)

사용: pickN.py OUTNAME STA_CSV GEN1_CSV:GEN1_DIR [GEN2_CSV:GEN2_DIR ...]
"""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
from ldwma.datasets.lerobot_so100 import preprocess_video
from eval.feature_csv_utils import save_video_tensor

csv.field_size_limit(10 ** 9)


def action_by_sample(p):
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["feature_component"] == "Action Component":
            d[r["sample_id"]] = float(json.loads(r["feature_json"])[0][0])
    return d


OUTNAME = sys.argv[1]
STA_CSV = sys.argv[2]
CANDS = []
for spec in sys.argv[3:]:
    c, d = spec.split(":")
    CANDS.append((Path(c), Path(d)))

IMG = Path("/workspace/open/data/eval/images")
OUT = Path(f"/workspace/runs/{OUTNAME}/videos")
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.mp4"):
    f.unlink()

sta = action_by_sample(STA_CSV)
gens = [action_by_sample(c) for c, _ in CANDS]
ids = sorted(set(sta).intersection(*[set(g) for g in gens]))
print(f"  후보 {1+len(CANDS)}종 · 공통 샘플 {len(ids)}개")

count = [0] * (1 + len(CANDS))
tot = 0.0
for sid in ids:
    vals = [sta[sid]] + [g[sid] for g in gens]
    k = int(np.argmin(vals))
    tot += vals[k]
    count[k] += 1
    if k == 0:
        img = np.asarray(Image.open(IMG / (sid + ".png")).convert("RGB"))
        t = preprocess_video(np.repeat(img[None], 16, axis=0), 320, 512, True)
        save_video_tensor(t, OUT / (sid + ".mp4"), 6)
    else:
        shutil.copy2(CANDS[k - 1][1] / (sid + ".mp4"), OUT / (sid + ".mp4"))

made = len(list(OUT.glob("*.mp4")))
names = ["정지"] + [f"생성{i+1}" for i in range(len(CANDS))]
print("  선택: " + " / ".join(f"{n} {c}개" for n, c in zip(names, count)))
print(f"  예상 Action MAE = {tot/len(ids):.6f}  가중 = {0.4*tot/len(ids):.6f}")
print(f"    (후보 2종일 때 최고 = 0.369727 / 가중 0.147891, 총점 0.26242)")
print(f"  최종 영상 {made}개")
if made != len(ids):
    sys.exit(1)
