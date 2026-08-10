"""예측 vs 예측 쌍을 크게 만든다. 장면·프레임·설정을 전부 무작위로 뽑는다.

왜 늘리나:
  300쌍에서 1위-2위 차이의 95%CI 가 [-0.0077, 0.0660] 로 0 을 아슬하게 포함했다.
  표본을 4배로 하면 신뢰구간이 약 1.8배 좁아져 확정될 가능성이 높다.

무작위성:
  (장면 216, 프레임 16, 설정쌍 6) 조합 20,736개에서 중복 없이 무작위 추출.
  같은 장면이 여러 번 뽑히면 군집 부트스트랩에서 장면 단위로 묶어 처리한다.

메모리:
  장면 하나씩 처리한다. 4벌 x 16프레임 = 64장(31MB)만 올린다.
"""
import csv, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, av
import imageio.v2 as iio
sys.path.insert(0,"/workspace/dinoproxy")
import torch; torch.set_num_threads(8)
from common import stage1_to_uint8, EVAL_H, EVAL_W

SETS=["sub_ck667","sub_g25","final_g10s50","final_static"]
R=Path("/workspace/runs")
N=int(sys.argv[1]) if len(sys.argv)>1 else 1200
out=Path("/workspace/pairs_big"); (out/"gt").mkdir(parents=True,exist_ok=True); (out/"pred").mkdir(parents=True,exist_ok=True)

ids=sorted(set.intersection(*[{p.stem for p in (R/s/"videos").glob("*.mp4")} for s in SETS]))
print("네 설정 모두에 있는 장면 %d개" % len(ids))
combos=[(a,b) for i,a in enumerate(SETS) for b in SETS[i+1:]]
print("설정쌍 %d가지: %s" % (len(combos),combos))

allc=[(sid,t,c) for sid in ids for t in range(16) for c in range(len(combos))]
print("가능한 조합 %d개 중 %d개 추출" % (len(allc),N))
rng=random.Random(0); picks=rng.sample(allc,min(N,len(allc)))
byscene=defaultdict(list)
for sid,t,c in picks: byscene[sid].append((t,c))
print("장면 %d개에 분포" % len(byscene))

def vid(p):
    o=[]
    with av.open(str(p)) as c:
        for f in c.decode(c.streams.video[0]): o.append(f.to_ndarray(format="rgb24"))
    return o
def canon(f): return stage1_to_uint8(f[None],EVAL_H,EVAL_W,True)[0].numpy()

rows=[]
for n,(sid,items) in enumerate(sorted(byscene.items())):
    vids={s:vid(R/s/"videos"/(sid+".mp4")) for s in SETS}
    for t,c in items:
        a,b=combos[c]
        if t>=len(vids[a]) or t>=len(vids[b]): continue
        i=len(rows)
        iio.imwrite(out/"gt"/("%05d.png"%i), canon(vids[a][t]))
        iio.imwrite(out/"pred"/("%05d.png"%i), canon(vids[b][t]))
        rows.append({"pair_id":"%05d"%i,"strategy":"pred_vs_pred","detail":"%s_f%02d_%s|%s"%(sid,t,a,b)})
    del vids
    if (n+1)%40==0: print("  장면 %d/%d, 누적 %d쌍" % (n+1,len(byscene),len(rows)))

with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["pair_id","strategy","detail"]); w.writeheader(); w.writerows(rows)
print("")
print("총 %d쌍 -> %s" % (len(rows),out))
from collections import Counter
cc=Counter(r["detail"].split("_",2)[2] for r in rows)
for k,v in cc.most_common(): print("   %-32s %d쌍" % (k,v))
