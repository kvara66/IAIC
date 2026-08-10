"""데이터를 더 뽑지 않고 결론이 튼튼한지 확인하는 두 가지 검사.

1) 반쪽 나누기(split-half)
   300쌍을 무작위로 절반씩 갈라 각각 순위를 매긴다. 두 절반이 같은 1위를 내면
   결론이 표본 절반에도 견딘다는 뜻이다. 이걸 2000번 반복한다.

2) 가장 닮은 구간(Q1)에서의 비교
   예측-정답 비교는 유사도가 높은 구간에서 일어난다. 그 구간만 떼어
   1위가 여전히 앞서는지 부트스트랩으로 본다.
"""
import csv, sys
from pathlib import Path
import numpy as np

pairs = Path("/workspace/dinopairs"); feats = pairs / "feats"
def load(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f): d[r["pair_id"]] = float(r["cos_dist"])
    return d
strat = {}
with open(pairs/"manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f): strat[r["pair_id"]] = r["strategy"]
data = {p.stem: load(p) for p in sorted(feats.glob("*.csv"))}
ids = sorted([p for p in data["dino"] if strat.get(p) != "identity"])
base = np.array([data["dino"][p] for p in ids])
cands = {k: np.array([v[p] for p in ids]) for k,v in data.items() if k!="dino"}
names = sorted(cands)
n = len(ids)

def pear(x,y):
    x=x-x.mean(); y=y-y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum())
    return (x*y).sum()/d if d>0 else np.nan

rng = np.random.default_rng(7)
print("=== 1) 반쪽 나누기 2000회 ===")
cnt = {k:0 for k in names}; agree = 0
for _ in range(2000):
    perm = rng.permutation(n); h = n//2
    a, b = perm[:h], perm[h:]
    wa = max(names, key=lambda k: pear(base[a], cands[k][a]))
    wb = max(names, key=lambda k: pear(base[b], cands[k][b]))
    cnt[wa]+=1; cnt[wb]+=1
    if wa==wb: agree+=1
for k in sorted(cnt, key=lambda k:-cnt[k]):
    print("   %-12s 반쪽에서 1위 %5.1f%%" % (k, cnt[k]/4000*100))
print("   두 절반이 같은 1위를 낸 비율 %.1f%%" % (agree/2000*100))

print("")
print("=== 2) 가장 닮은 구간(Q1, DINO 거리 하위 25%%)에서 비교 ===")
q1 = base <= np.quantile(base, 0.25)
m = int(q1.sum()); print("   Q1 표본 %d쌍" % m)
b1 = base[q1]
obs = {k: pear(b1, cands[k][q1]) for k in names}
for k in sorted(obs, key=lambda k:-obs[k]): print("   %-12s r=%.4f" % (k, obs[k]))
o = sorted(obs, key=lambda k:-obs[k])
B=10000; idx = rng.integers(0,m,size=(B,m)); bt={k:np.empty(B) for k in names}
for i in range(B):
    j=idx[i]; bb=b1[j]
    for k in names: bt[k][i]=pear(bb, cands[k][q1][j])
st=np.stack([bt[k] for k in o]); top=np.argmax(st,axis=0)
for j,k in enumerate(o): print("   %-12s Q1 1위비율 %5.1f%%" % (k,(top==j).mean()*100))
d = bt[o[0]]-bt[o[1]]; lo,hi = np.percentile(d,[2.5,97.5])
print("   %s - %s : %.4f  95%%CI [%.4f, %.4f]  0보다 클 확률 %.1f%%"
      % (o[0],o[1],obs[o[0]]-obs[o[1]],lo,hi,(d>0).mean()*100))
