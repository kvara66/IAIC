"""해상도 차이가 상관계수를 만든 건지 확인한다.

문제:
  같은 리사이즈 로직을 써도 모델 입력 크기가 달라 내용 해상도가 크게 다르다.
    DINO 518 -> 내용 324x432
    후보 224 -> 내용 140x187      가로세로 2.3배, 픽셀 5.3배 차이

  그러면 "판단 방식이 달라서" 가 아니라 "덜 보여서" 못 따라가는 걸 수도 있다.

두 방향으로 확인한다.
  1) DINO 를 224 로 낮춰서 후보와 같은 해상도로 맞춘다
  2) 후보를 518 로 올려서 DINO 와 같은 해상도로 맞춘다
  (timm 이 위치 임베딩을 보간해 준다)

⚠ 최종 답의 기준은 반드시 DINO@518 이다. 채점기가 그걸 쓴다.
  여기 결과는 원인 진단용이다.
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, timm
sys.path.insert(0,"/workspace/dinoproxy")
from common import (EVAL_H, EVAL_W, model_input_size, normalize_stats,
                    extract_features, cosine, content_fraction)
torch.set_num_threads(8)
P=Path("/workspace/dinopairs"); dev=torch.device("cuda")
M={"dino":"vit_small_patch14_dinov2.lvd142m","clip_l14":"vit_large_patch14_clip_224.openai",
   "clip_b16":"vit_base_patch16_clip_224.openai","siglip_b16":"vit_base_patch16_siglip_224",
   "mae_b16":"vit_base_patch16_224.mae"}
import imageio.v2 as iio
def side(d):
    fs=sorted((P/d).glob("*.png"))
    return torch.from_numpy(np.stack([iio.imread(f)[...,:3] for f in fs])).permute(0,3,1,2).contiguous(),[f.stem for f in fs]
gt,ids=side("gt"); pr,_=side("pred")
strat={r["pair_id"]:r["strategy"] for r in csv.DictReader(open(P/"manifest.csv",encoding="utf-8"))}
keep=[i for i,p in enumerate(ids) if strat.get(p)!="identity"]
def pear(x,y):
    x=x-x.mean(); y=y-y.mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))

def run(name,size=None):
    kw=dict(pretrained=True,num_classes=0)
    if size: kw["img_size"]=size
    m=timm.create_model(M[name],**kw).to(dev).eval()
    s=model_input_size(m,0); mm,ss=normalize_stats(m)
    fg=extract_features(gt,m,dev,s,mm,ss,8); fp=extract_features(pr,m,dev,s,mm,ss,8)
    d=(1.0-cosine(fg,fp)).numpy(); del m; torch.cuda.empty_cache()
    rh=max(1,round(EVAL_H*min(s/EVAL_H,s/EVAL_W)))
    return d,s,rh

print("=== 기준 두 가지 ===")
d518,s1,rh1=run("dino"); d224,s2,rh2=run("dino",224)
print("  DINO @518  내용 %dx%d" % (rh1,s1))
print("  DINO @224  내용 %dx%d" % (rh2,s2))
print("  두 DINO 끼리 상관 r=%.4f  <- 해상도만 바꿔도 이만큼 달라진다" % pear(d518[keep],d224[keep]))

for base,tag in [(d518,"DINO@518 (채점기 기준)"),(d224,"DINO@224 (해상도 맞춤)")]:
    print("")
    print("=== 후보 @224 vs %s ===" % tag)
    for n in ["siglip_b16","clip_b16","clip_l14","mae_b16"]:
        d,_,_=run(n)
        print("   %-12s r=%.4f" % (n,pear(base[keep],d[keep])))

print("")
print("=== 후보를 518 로 올리면 (DINO@518 기준) ===")
for n in ["siglip_b16","clip_b16","clip_l14","mae_b16"]:
    try:
        d,s,rh=run(n,518)
        print("   %-12s @%d 내용 %dx%d  r=%.4f" % (n,s,rh,s,pear(d518[keep],d[keep])))
    except Exception as e:
        print("   %-12s 518 실패: %s" % (n,str(e)[:80]))
