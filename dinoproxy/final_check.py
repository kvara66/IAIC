"""최종 검증 - 파이프라인 자체가 맞는지 네 가지로 확인한다.

1) PNG 왕복 무손실   1단계 결과를 파일로 저장했다 읽어도 값이 그대로인가
                     (채점기는 메모리에서 바로 넘긴다. 여기가 유일하게 안 잰 구간)
2) 채점기 전체 경로   영상배열 -> 채점기 방식  vs  영상배열 -> PNG -> 내 방식
                     최종 코사인 유사도까지 같은가
3) 상관계수 재계산    내가 손으로 짠 pear() 를 numpy.corrcoef 로 대조
4) 재현성            dino 특징을 처음부터 다시 뽑아 기존 CSV 와 대조
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, timm
sys.path.insert(0,"/workspace/dinoproxy")
sys.path.insert(0,"/workspace/open/submission_kit")
import feature_csv_utils as K
from common import (EVAL_H, EVAL_W, stage1_to_uint8, model_input_size,
                    normalize_stats, extract_features, cosine)
import imageio.v2 as iio
torch.set_num_threads(8)
FAIL=[]
def ck(label,d,tol=0.0):
    ok=d<=tol; print("  %-52s %.3e  %s"%(label,d,"통과" if ok else "★실패★"))
    if not ok: FAIL.append(label)

print("=== 1) PNG 왕복이 무손실인가 ===")
rng=np.random.default_rng(0)
for (h,w) in [(480,640),(720,1280),(1080,1920),(240,320)]:
    v=rng.integers(0,256,(1,h,w,3),dtype=np.uint8)
    mem=stage1_to_uint8(v,EVAL_H,EVAL_W,True)[0].numpy()
    iio.imwrite("/tmp/rt.png",mem)
    back=iio.imread("/tmp/rt.png")[...,:3]
    ck("원본 %dx%d -> 320x512 -> PNG -> 읽기"%(h,w), float(np.abs(mem.astype(int)-back.astype(int)).max()))

print("")
print("=== 2) 채점기 전체 경로와 내 경로가 같은 답을 내는가 ===")
dev=torch.device("cuda")
model=K.load_dino_model(dev,"vit_small_patch14_dinov2.lvd142m",pretrained=True)
size_k=K.resolve_dino_image_size(model,requested_size=0)
A=rng.integers(0,256,(16,480,640,3),dtype=np.uint8)   # 정답 역할
B=rng.integers(0,256,(16,480,640,3),dtype=np.uint8)   # 예측 역할

# 채점기 방식: 배열 -> to_eval_uint8 -> extract_dino_features
ka=K.to_eval_uint8(A,320,512,True); kb=K.to_eval_uint8(B,320,512,True)
fa=K.extract_dino_features(ka[None],model,dev,size_k)[0]
fb=K.extract_dino_features(kb[None],model,dev,size_k)[0]
sim_kit=torch.nn.functional.cosine_similarity(fa,fb,dim=-1)

# 내 방식: 배열 -> 1단계 -> PNG 저장 -> 읽기 -> 2단계 -> 특징
mm,ss=normalize_stats(model); size_c=model_input_size(model,0)
def mine(arr):
    out=[]
    for i in range(arr.shape[0]):
        one=stage1_to_uint8(arr[i:i+1],EVAL_H,EVAL_W,True)[0].numpy()
        iio.imwrite("/tmp/m%02d.png"%i,one); out.append(iio.imread("/tmp/m%02d.png"%i)[...,:3])
    t=torch.from_numpy(np.stack(out)).permute(0,3,1,2).contiguous()
    return extract_features(t,model,dev,size_c,mm,ss,16)
sim_mine=cosine(mine(A),mine(B))
ck("코사인 유사도 16프레임 전부", float((sim_kit.cpu()-sim_mine).abs().max()), tol=1e-6)
print("     채점기 %s" % np.round(sim_kit.cpu().numpy()[:4],6).tolist())
print("     내 것  %s" % np.round(sim_mine.numpy()[:4],6).tolist())

print("")
print("=== 3) 상관계수 계산이 맞는가 (numpy 로 재계산) ===")
P=Path("/workspace/dinopairs")
def load(p):
    return {r["pair_id"]:float(r["cos_dist"]) for r in csv.DictReader(open(p,encoding="utf-8"))}
strat={r["pair_id"]:r["strategy"] for r in csv.DictReader(open(P/"manifest.csv",encoding="utf-8"))}
D=load(P/"feats/dino.csv"); ids=sorted([k for k in D if strat.get(k)!="identity"])
base=np.array([D[k] for k in ids])
def pear(x,y):
    x=x-x.mean(); y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))
for n in ["siglip_b16","clip_b16","clip_l14","mae_b16"]:
    v=np.array([load(P/("feats/%s.csv"%n))[k] for k in ids])
    mine_r=pear(base,v); np_r=float(np.corrcoef(base,v)[0,1])
    print("  %-12s 내 계산 %.6f   numpy %.6f   차이 %.2e" % (n,mine_r,np_r,abs(mine_r-np_r)))
    ck("   %s 두 계산 일치"%n, abs(mine_r-np_r), tol=1e-12)

print("")
print("=== 4) dino 특징을 처음부터 다시 뽑아 기존 CSV 와 대조 ===")
def side(d):
    fs=sorted((P/d).glob("*.png"))
    return torch.from_numpy(np.stack([iio.imread(f)[...,:3] for f in fs])).permute(0,3,1,2).contiguous(),[f.stem for f in fs]
gt,gid=side("gt"); pr,_=side("pred")
fg=extract_features(gt,model,dev,size_c,mm,ss,16); fp=extract_features(pr,model,dev,size_c,mm,ss,16)
redo=(1.0-cosine(fg,fp)).numpy()
old=np.array([D[k] for k in gid])
ck("기존 dino.csv 320행과 재계산", float(np.abs(redo-old).max()), tol=1e-6)

print("")
print("="*62)
print("전부 통과. 파이프라인에 구멍 없음." if not FAIL else "★실패 %d건: %s"%(len(FAIL),FAIL))
