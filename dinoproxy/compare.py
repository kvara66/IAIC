"""1위와 2위의 차이가 표본 잡음인지 부트스트랩으로 검정한다.

왜 필요한가:
  상관계수 0.6341 대 0.5723 은 차이가 0.062 뿐이다. 표본 300개에서 이 정도는
  우연히 뒤집힐 수 있다. 단순히 큰 쪽을 1위라고 선언하면 틀릴 수 있다.

방법:
  같은 300쌍을 복원추출로 다시 뽑아 상관계수를 다시 계산하는 것을 10000번 한다.
  매번 어느 모델이 이기는지 세면 "1위가 진짜 1위일 확률" 이 나온다.
  두 상관계수가 같은 DINO 값을 공유하므로 쌍 단위로 함께 리샘플한다(paired).
"""
import csv, sys
from pathlib import Path
import numpy as np

pairs = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/dinopairs")
feats = pairs / "feats"

def load(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["pair_id"]] = float(r["cos_dist"])
    return d

strat = {}
with open(pairs / "manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        strat[r["pair_id"]] = r["strategy"]

data = {p.stem: load(p) for p in sorted(feats.glob("*.csv"))}
ids = sorted([p for p in data["dino"] if strat.get(p) != "identity"])
base = np.array([data["dino"][p] for p in ids])
cands = {k: np.array([v[p] for p in ids]) for k, v in data.items() if k != "dino"}
n = len(ids)
print("표본 %d쌍" % n)

def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x*x).sum() * (y*y).sum())
    return (x*y).sum()/d if d > 0 else np.nan

obs = {k: pear(base, v) for k, v in cands.items()}
order = sorted(obs, key=lambda k: -obs[k])
print("")
print("관측 Pearson:")
for k in order:
    print("   %-12s %.4f" % (k, obs[k]))

rng = np.random.default_rng(0)
B = 10000
idx = rng.integers(0, n, size=(B, n))
boot = {k: np.empty(B) for k in cands}
for b in range(B):
    i = idx[b]
    bb = base[i]
    for k, v in cands.items():
        boot[k][b] = pear(bb, v[i])

print("")
print("부트스트랩 10000회 - 95%% 신뢰구간과 1위 차지 비율:")
wins = np.zeros(len(order))
stack = np.stack([boot[k] for k in order])          # (모델, B)
top = np.argmax(stack, axis=0)
for j, k in enumerate(order):
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    print("   %-12s r=%.4f  95%%CI [%.4f, %.4f]  1위비율 %5.1f%%"
          % (k, obs[k], lo, hi, (top == j).mean()*100))

a, b2 = order[0], order[1]
d = boot[a] - boot[b2]
lo, hi = np.percentile(d, [2.5, 97.5])
print("")
print("1위(%s) - 2위(%s) 차이" % (a, b2))
print("   관측 %.4f   95%%CI [%.4f, %.4f]   차이가 0보다 클 확률 %.1f%%"
      % (obs[a]-obs[b2], lo, hi, (d > 0).mean()*100))
print("")
if lo > 0:
    print("판정: 신뢰구간이 0 을 넘지 않는다. %s 가 유의하게 앞선다." % a)
else:
    print("판정: 신뢰구간이 0 을 포함한다. %s 우세지만 표본 300개로는 확정 못 한다." % a)
