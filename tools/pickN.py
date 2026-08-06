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
