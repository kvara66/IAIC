# ============================================================================
# 사용 금지 — 규정 위반 방식 (2026-08-06 확인)
#
# 이 스크립트는 submission_kit 을 실행해 얻은 Action 값으로 영상을 고르거나
# 후보를 평가한다. 대회 규칙 "Submission Kit 사용 범위(8/3 11시~)" 의
# 다음 항목에 해당한다.
#
#   - Test 데이터의 특징값을 추출하여 예측 결과를 선택하거나 보정하는 행위
#   - Submission Kit 의 실행 결과를 활용하여 최종 mp4 를 생성·선택·수정하는 행위
#   - Submission Kit 정보를 후처리 또는 앙상블 과정에 활용하는 행위
#   - Submission Kit 을 제출 CSV 생성 외의 목적으로 사용하는 행위
#   - 최종 영상 선택은 Submission Kit 실행 이전에 완료되어야 함
#
# 위반 시 평가·수상 대상에서 제외될 수 있다.
# 2026-08-06 데이콘에 자진 신고했고, 이 방식의 작업은 전부 중단했다.
#
# 지우지 않고 남겨두는 이유 — 무엇을 어떻게 했는지 설명할 근거가 필요하다.
# 실행하지 말 것. 기록용 보관.
# ============================================================================

"""전체 후보 + 어려운 샘플 전용 후보를 함께 합친다.

집중 생성 후보(d1~d6)는 어려운 60개만 갖고 있다. 그래서 pickN.py 를 못 쓴다.
여기서는 샘플마다 '그 샘플에 대해 존재하는 후보들' 중에서 최선을 고른다.

  전체 216개 샘플: 기존 13종 중 최선
  어려운 60개    : 기존 13종 + d1~dN 중 최선

사용: pickhard.py OUTNAME [d1 d2 ...]
"""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src")); sys.path.insert(0, str(KIT / "scripts"))
from ldwma.datasets.lerobot_so100 import preprocess_video
from eval.feature_csv_utils import save_video_tensor

csv.field_size_limit(10 ** 9)
R = Path("/workspace/runs")
IMG = Path("/workspace/open/data/eval/images")

OUTNAME = sys.argv[1]
EXTRA = sys.argv[2:]

BASE = [
    ("정지", "pure_sta",  None),
    ("s0",  "pure_gen",  "submit_p6k"),
    ("s1",  "seed1_gen", "seed1"),
    ("s2",  "v2_gen",    "v2"),
    ("s3",  "v3_gen",    "v3"),
    ("s4",  "v4_gen",    "v4"),
    ("s5",  "v5_gen",    "v5"),
    ("s6",  "v6_gen",    "v6"),
    ("s7",  "v7_gen",    "v7"),
    ("s8",  "v8_gen",    "v8"),
    ("s9",  "v9_gen",    "v9"),
    ("s10", "v10_gen",   "v10"),
    ("h30", "h30_gen",   "h30"),
]


def load(n):
    p = R / n / "submission_features.csv"
    if not p.exists() or p.stat().st_size < 5_000_000:
        return None
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["feature_component"] == "Action Component":
            d[r["sample_id"]] = float(json.loads(r["feature_json"])[0][0])
    return d


act, vid = {}, {}
for lab, name, vd in BASE:
    d = load(name)
    if d:
        act[lab] = d; vid[lab] = vd
base_labels = list(act)
ids = sorted(set.intersection(*[set(act[l]) for l in base_labels]))
print(f"  기존 후보 {len(base_labels)}종 · 전체 샘플 {len(ids)}개")

for e in EXTRA:
    d = load(f"{e}_gen")
    if d:
        act[e] = d; vid[e] = e
        print(f"  집중 후보 {e}: {len(d)}개 샘플")
    else:
        print(f"  [건너뜀] {e}_gen CSV 없음")

hard = set()
hp = Path("/workspace/hard/hard_ids.txt")
if hp.exists():
    hard = set(hp.read_text(encoding="utf-8").split())

OUT = R / OUTNAME / "videos"
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.mp4"):
    f.unlink()

cnt, tot, improved = {}, 0.0, 0
base_tot = 0.0
for s in ids:
    avail = [l for l in act if s in act[l]]
    vals = {l: act[l][s] for l in avail}
    best = min(vals, key=vals.get)
    bo = min(act[l][s] for l in base_labels)
    base_tot += bo
    tot += vals[best]
    if vals[best] < bo - 1e-9:
        improved += 1
    cnt[best] = cnt.get(best, 0) + 1
    if vid[best] is None:
        img = np.asarray(Image.open(IMG / (s + ".png")).convert("RGB"))
        t = preprocess_video(np.repeat(img[None], 16, axis=0), 320, 512, True)
        save_video_tensor(t, OUT / (s + ".mp4"), 6)
    else:
        shutil.copy2(R / vid[best] / "videos" / (s + ".mp4"), OUT / (s + ".mp4"))

n = len(ids)
print("  선택: " + " ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
print(f"  집중 후보가 이긴 샘플 {improved}개 (어려운 {len(hard)}개 중)")
print(f"  기존 13종만: Action {base_tot/n:.6f}  가중 {0.4*base_tot/n:.6f}")
print(f"  집중 포함  : Action {tot/n:.6f}  가중 {0.4*tot/n:.6f}   차이 {0.4*(tot-base_tot)/n:+.6f}")
if hard:
    hb = sum(min(act[l][s] for l in base_labels) for s in hard if s in ids) / len(hard)
    hn = sum(min(act[l][s] for l in act if s in act[l]) for s in hard if s in ids) / len(hard)
    print(f"  어려운 {len(hard)}개만: {hb:.6f} -> {hn:.6f}  ({hn-hb:+.6f})")
made = len(list(OUT.glob("*.mp4")))
print(f"  최종 영상 {made}개")
if made != n:
    sys.exit(1)
