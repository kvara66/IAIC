"""어떤 시드/guidance 조합이 실제로 값을 하는지 잰다. GPU 불필요.

샘플별 최선을 고르는 방식에서는 두 가지가 중요하다:
  (1) 후보 자체의 품질 (개별 Action 이 낮을수록 좋다)
  (2) 다양성 (다른 후보와 '다른 샘플에서' 실패해야 합쳐서 이득이 난다)

개별 품질만 좋고 남들과 똑같이 틀리면 합쳐도 소용이 없다.
그래서 빼봤을 때 얼마나 나빠지는지(leave-one-out)로 기여도를 잰다.
"""
import csv, json, itertools
from pathlib import Path
import statistics as st

csv.field_size_limit(10 ** 9)
R = Path("/workspace/runs")

CANDS = [
    ("정지",        "pure_sta",  None, None),
    ("submit_p6k",  "pure_gen",  0, 1.0),
    ("seed1",       "seed1_gen", 1, 1.0),
    ("v2",          "v2_gen",    2, 1.0),
    ("v3",          "v3_gen",    3, 1.5),
    ("v4",          "v4_gen",    4, 2.5),
    ("v5",          "v5_gen",    5, 1.0),
    ("v6",          "v6_gen",    6, 1.5),
    ("v7",          "v7_gen",    7, 2.0),
    ("v8",          "v8_gen",    8, 0.7),
    ("v9",          "v9_gen",    9, 1.0),
    ("v10",         "v10_gen",  10, 1.8),
    ("v11",         "v11_gen",  11, 1.2),
    ("v12",         "v12_gen",  12, 2.2),
]


def load(name):
    p = R / name / "submission_features.csv"
    if not p.exists() or p.stat().st_size < 27_000_000:
        return None
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["feature_component"] == "Action Component":
            d[r["sample_id"]] = float(json.loads(r["feature_json"])[0][0])
    return d


data, meta = {}, {}
for label, name, seed, g in CANDS:
    d = load(name)
    if d:
        data[label] = d
        meta[label] = (seed, g)
print(f"불러온 후보 {len(data)}개: {', '.join(data)}")
ids = sorted(set.intersection(*[set(v) for v in data.values()]))
print(f"공통 샘플 {len(ids)}개")
print()

labels = list(data)
def mean_min(subset):
    return st.mean(min(data[l][s] for l in subset) for s in ids)

full = mean_min(labels)
print(f"전체 합침 Action MAE = {full:.6f}  (가중 {0.4*full:.6f})")
print()

print("=== 후보별 기여도 (빼면 얼마나 나빠지나) ===")
print(f"{'후보':<12}{'시드':>5}{'guidance':>10}{'개별MAE':>10}{'빼면 악화':>11}{'최선인 샘플':>11}")
print("-" * 62)
rows = []
for l in labels:
    rest = [x for x in labels if x != l]
    loss = mean_min(rest) - full
    own = st.mean(data[l][s] for s in ids)
    wins = sum(1 for s in ids if data[l][s] == min(data[x][s] for x in labels))
    sd, g = meta[l]
    rows.append((l, sd, g, own, loss, wins))
for l, sd, g, own, loss, wins in sorted(rows, key=lambda r: -r[4]):
    print(f"{l:<12}{'-' if sd is None else sd:>5}{'-' if g is None else g:>10}{own:>10.4f}{loss:>11.5f}{wins:>11}")

print()
print("=== guidance 값별 묶음 ===")
bins = {}
for l, sd, g, own, loss, wins in rows:
    if g is None:
        continue
    bins.setdefault(g, []).append((own, loss, wins))
print(f"{'guidance':>9}{'개수':>5}{'평균 개별MAE':>13}{'평균 기여도':>12}{'평균 승수':>10}")
for g in sorted(bins):
    v = bins[g]
    print(f"{g:>9}{len(v):>5}{st.mean(x[0] for x in v):>13.4f}"
          f"{st.mean(x[1] for x in v):>12.5f}{st.mean(x[2] for x in v):>10.1f}")

print()
print("=== 후보 쌍의 상관 (낮을수록 서로 다르게 틀린다 = 합칠 때 유리) ===")
gen = [l for l in labels if meta[l][0] is not None]
def corr(a, b):
    xa = [data[a][s] for s in ids]; xb = [data[b][s] for s in ids]
    ma, mb = st.mean(xa), st.mean(xb)
    num = sum((x-ma)*(y-mb) for x, y in zip(xa, xb))
    den = (sum((x-ma)**2 for x in xa) * sum((y-mb)**2 for y in xb)) ** 0.5
    return num/den if den else 0.0
pairs = sorted(((corr(a, b), a, b) for a, b in itertools.combinations(gen, 2)))
print("  가장 다른 쌍 5개:")
for c, a, b in pairs[:5]:
    print(f"    {a:>6}(g{meta[a][1]}) vs {b:>6}(g{meta[b][1]})   상관 {c:+.3f}")
print("  가장 비슷한 쌍 5개:")
for c, a, b in pairs[-5:]:
    print(f"    {a:>6}(g{meta[a][1]}) vs {b:>6}(g{meta[b][1]})   상관 {c:+.3f}")
print()
print(f"  전체 평균 상관 {st.mean(p[0] for p in pairs):+.3f}")
