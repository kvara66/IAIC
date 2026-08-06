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

"""아직 잘 안 되는 샘플만 골라 별도 폴더를 만든다.

착상: 지금은 후보를 만들 때마다 216개를 전부 생성한다. 그런데 실측을 보면
      하위 25%(53개)가 전체 Action 합의 41.8% 를 차지한다.
      나머지는 이미 잘 맞아서 후보를 더 만들어도 안 바뀐다.
      즉 GPU 의 75% 를 이미 해결된 샘플에 쓰고 있다.

방법: 현재 12종 후보의 샘플별 최선 Action 을 구해, 나쁜 순으로 N 개를 고른다.
      그 샘플들의 images/actions 만 담은 폴더를 만들고 --challenge-root 로 가리키면
      추론 스크립트가 그것만 생성한다. 코드 수정이 필요 없다.

사용: hardset.py N
"""
import csv, json, shutil, sys
from pathlib import Path

csv.field_size_limit(10 ** 9)
R = Path("/workspace/runs")
EV = Path("/workspace/open/data/eval")
OUT = Path("/workspace/hard")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

NAMES = ["pure_sta", "pure_gen", "seed1_gen", "v2_gen", "v3_gen", "v4_gen",
         "v5_gen", "v6_gen", "v7_gen", "v8_gen", "v9_gen", "v10_gen", "h30_gen"]


def load(n):
    p = R / n / "submission_features.csv"
    if not p.exists() or p.stat().st_size < 27_000_000:
        return None
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["feature_component"] == "Action Component":
            d[r["sample_id"]] = float(json.loads(r["feature_json"])[0][0])
    return d


data = {n: d for n in NAMES if (d := load(n))}
print(f"후보 {len(data)}종: {', '.join(data)}")
ids = sorted(set.intersection(*[set(v) for v in data.values()]))
best = {s: min(data[n][s] for n in data) for s in ids}

order = sorted(ids, key=lambda s: -best[s])
hard = order[:N]
rest = order[N:]

import statistics as st
print(f"\n전체 {len(ids)}개 · 현재 최선 Action 평균 {st.mean(best.values()):.6f}")
print(f"어려운 {N}개  평균 {st.mean(best[s] for s in hard):.6f}  (전체 합의 "
      f"{100*sum(best[s] for s in hard)/sum(best.values()):.1f}%)")
print(f"나머지 {len(rest)}개 평균 {st.mean(best[s] for s in rest):.6f}")
print()
print("  * 어려운 쪽만 개선하면 전체 평균이 그 비중만큼 내려간다.")
print(f"    예: 어려운 {N}개가 {st.mean(best[s] for s in hard):.3f} -> 0.40 이 되면")
d = (st.mean(best[s] for s in hard) - 0.40) * N / len(ids)
print(f"       전체 Action -{d:.4f} (가중 -{0.4*d:.4f})")

for sub in ("images", "actions"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    for f in (OUT / sub).glob("*"):
        f.unlink()
ext = {"images": ".png", "actions": ".npy"}
for s in hard:
    for sub in ("images", "actions"):
        src = EV / sub / (s + ext[sub])
        dst = OUT / sub / (s + ext[sub])
        if src.exists():
            shutil.copy2(src, dst)

print()
print(f"생성 대상 폴더: {OUT}")
print(f"  이미지 {len(list((OUT/'images').glob('*.png')))}개  액션 {len(list((OUT/'actions').glob('*.npy')))}개")
(OUT / "hard_ids.txt").write_text("\n".join(hard), encoding="utf-8")
print(f"  목록 저장: {OUT}/hard_ids.txt")
print()
print(f"비용: 216개 대비 {100*N/len(ids):.0f}%  ->  같은 1시간에 후보 {len(ids)/N:.1f}세트")
