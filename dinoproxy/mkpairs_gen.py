"""생성 영상으로 쌍을 만든다. 채점기가 실제로 보는 상황에 맞춘다.

실험 A (pairs_gt)  진짜 정답 vs 우리 예측
  홀드아웃 정답 npy (16,480,640,3) 과 같은 장면의 우리 생성 mp4 (16,320,512)
  프레임 t 끼리 짝짓는다. 채점기가 하는 것과 완전히 같은 비교다.
  정답은 480x640 이라 1단계에서 좌우 검은 패딩이 붙고, 생성물은 이미 320x512 라 통과한다.
  20샘플 x 16프레임 = 320쌍

실험 B (pairs_gen)  우리 예측 vs 우리 예측
  같은 sample_id 를 서로 다른 설정으로 만든 두 생성물.
  정답이 없어도 된다 - 두 이미지를 비교하는 두 자리일 뿐이다.
  216장면에서 무작위로 300쌍

⚠ 두 실험 모두 1단계(320x512 비율유지+중앙정렬+검은패딩)를 거쳐 PNG 로 저장한다.
  기존 실험과 완전히 같은 경로다.
"""
import random, sys, csv
from pathlib import Path
import numpy as np, av
import imageio.v2 as iio
sys.path.insert(0,"/workspace/dinoproxy")
import torch; torch.set_num_threads(8)
from common import stage1_to_uint8, EVAL_H, EVAL_W

def canon(f): return stage1_to_uint8(f[None],EVAL_H,EVAL_W,True)[0].numpy()
def vid(p):
    out=[]
    with av.open(str(p)) as c:
        for f in c.decode(c.streams.video[0]): out.append(f.to_ndarray(format="rgb24"))
    return out

# ---------------- A ----------------
GT=Path("/workspace/holdout/gt"); PR=Path("/workspace/runs/ck667/videos")
gids={p.stem for p in GT.glob("*.npy")}; pids={p.stem for p in PR.glob("*.mp4")}
both=sorted(gids&pids)
print("[A] 정답·생성 양쪽에 있는 샘플 %d개 (정답 %d, 생성 %d)"%(len(both),len(gids),len(pids)))
outA=Path("/workspace/pairs_gt"); (outA/"gt").mkdir(parents=True,exist_ok=True); (outA/"pred").mkdir(parents=True,exist_ok=True)
rows=[]
for sid in both:
    g=np.load(GT/(sid+".npy")); p=vid(PR/(sid+".mp4"))
    n=min(len(g),len(p))
    for t in range(n):
        i=len(rows)
        iio.imwrite(outA/"gt"/("%04d.png"%i), canon(g[t]))
        iio.imwrite(outA/"pred"/("%04d.png"%i), canon(p[t]))
        rows.append({"pair_id":"%04d"%i,"strategy":"gt_vs_pred","detail":"%s_f%02d"%(sid,t)})
with (outA/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["pair_id","strategy","detail"]); w.writeheader(); w.writerows(rows)
print("[A] %d쌍 -> %s"%(len(rows),outA))

# ---------------- B ----------------
V1=Path("/workspace/runs/sub_ck667/videos"); V2=Path("/workspace/runs/sub_g25/videos")
ids=sorted({p.stem for p in V1.glob("*.mp4")}&{p.stem for p in V2.glob("*.mp4")})
print("[B] 두 설정 모두에 있는 장면 %d개"%len(ids))
outB=Path("/workspace/pairs_gen"); (outB/"gt").mkdir(parents=True,exist_ok=True); (outB/"pred").mkdir(parents=True,exist_ok=True)
rng=random.Random(0); rows=[]
picks=[(sid,t) for sid in ids for t in range(16)]
rng.shuffle(picks)
cache={}
for sid,t in picks[:300]:
    if sid not in cache: cache[sid]=(vid(V1/(sid+".mp4")),vid(V2/(sid+".mp4")))
    a,b=cache[sid]
    if t>=len(a) or t>=len(b): continue
    i=len(rows)
    iio.imwrite(outB/"gt"/("%04d.png"%i), canon(a[t]))
    iio.imwrite(outB/"pred"/("%04d.png"%i), canon(b[t]))
    rows.append({"pair_id":"%04d"%i,"strategy":"pred_vs_pred","detail":"%s_f%02d"%(sid,t)})
    if len(cache)>40: cache.clear()
with (outB/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["pair_id","strategy","detail"]); w.writeheader(); w.writerows(rows)
print("[B] %d쌍 -> %s"%(len(rows),outB))

# 눈으로 확인할 그림
for nm,d in [("A",outA),("B",outB)]:
    tiles=[np.concatenate([iio.imread(d/"gt"/("%04d.png"%i)),iio.imread(d/"pred"/("%04d.png"%i))],axis=1) for i in (0,1,2)]
    iio.imwrite("/workspace/pairs_%s_check.png"%nm, np.concatenate(tiles,axis=0))
print("확인용 그림 저장")
