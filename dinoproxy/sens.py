"""판단으로 정한 것이 결론을 바꾸는지 검정한다.

채점기는 DINO 만 돌리므로 후보 모델에는 "채점기와 대조" 가 불가능하다.
그래서 아래 두 가지는 내가 정한 것이다. 정말 상관없는지 실제로 확인한다.

  A) 정규화 상수
     지금: 각 모델 자기 pretrained_cfg 값
     대안: 채점기가 DINO 에 쓰듯 ImageNet 을 전 모델에 강제
     -> 두 경우의 순위가 같은가?

  B) 기하 처리가 정말 의도대로인가
     패딩 영역의 실제 값이 (0-mean)/std 인지, 내용이 가운데인지 숫자로 확인.
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, timm
sys.path.insert(0, "/workspace/dinoproxy")
from common import (EVAL_H, EVAL_W, stage2_resize_pad, model_input_size,
                    normalize_stats, pool_output, extract_features, cosine,
                    IMAGENET_MEAN, IMAGENET_STD)
torch.set_num_threads(8)

P = Path("/workspace/dinopairs")
MODELS = {"dino":"vit_small_patch14_dinov2.lvd142m",
          "clip_l14":"vit_large_patch14_clip_224.openai",
          "clip_b16":"vit_base_patch16_clip_224.openai",
          "siglip_b16":"vit_base_patch16_siglip_224",
          "mae_b16":"vit_base_patch16_224.mae"}
dev = torch.device("cuda")

import imageio.v2 as iio
def side(d):
    fs = sorted((P/d).glob("*.png"))
    a = np.stack([iio.imread(f)[...,:3] for f in fs])
    return torch.from_numpy(a).permute(0,3,1,2).contiguous(), [f.stem for f in fs]
gt,ids = side("gt"); pr,_ = side("pred")
strat = {r["pair_id"]:r["strategy"] for r in csv.DictReader(open(P/"manifest.csv",encoding="utf-8"))}
keep = [i for i,p in enumerate(ids) if strat.get(p)!="identity"]

def pear(x,y):
    x=x-x.mean(); y=y-y.mean()
    return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))

# ---------- B) 기하 처리 확인 ----------
print("=== B) 기하 처리가 의도대로인가 ===")
for name,size in [("dino",518),("clip_b16",224)]:
    x = stage2_resize_pad(gt[:1], size, 0.0)
    rh = max(1, round(EVAL_H*min(size/EVAL_H, size/EVAL_W)))
    pt = (size-rh)//2
    top_is_black = float(x[0,:,:pt,:].abs().max())
    bot_is_black = float(x[0,:,pt+rh:,:].abs().max())
    print("  %-10s size=%d  내용행 %d~%d (높이 %d)  위여백 최대 %.1e  아래여백 최대 %.1e  %s"
          % (name,size,pt,pt+rh,rh,top_is_black,bot_is_black,
             "패딩=0 확인" if max(top_is_black,bot_is_black)<1e-9 else "★이상"))
    m,s = normalize_stats(timm.create_model(MODELS[name],pretrained=False,num_classes=0))
    xn = (x-m)/s
    exp = float(((0.0-m)/s).flatten()[0])
    got = float(xn[0,0,0,0])
    print("             정규화 후 패딩값 %.4f (기대 %.4f)  %s"
          % (got,exp,"일치" if abs(got-exp)<1e-5 else "★불일치"))

# ---------- A) 정규화를 ImageNet 으로 강제하면 ----------
print("")
print("=== A) 정규화 상수를 바꾸면 순위가 바뀌는가 ===")
res = {}
for name,tn in MODELS.items():
    m = timm.create_model(tn,pretrained=True,num_classes=0).to(dev).eval()
    size = model_input_size(m,0)
    own_m,own_s = normalize_stats(m)
    for tag,(mm,ss) in {"자기상수":(own_m,own_s),"ImageNet강제":(IMAGENET_MEAN,IMAGENET_STD)}.items():
        fg = extract_features(gt,m,dev,size,mm,ss,16)
        fp = extract_features(pr,m,dev,size,mm,ss,16)
        res[(name,tag)] = (1.0-cosine(fg,fp)).numpy()
    del m; torch.cuda.empty_cache()

for tag in ["자기상수","ImageNet강제"]:
    b = res[("dino",tag)][keep]
    print("")
    print("  [%s]  (DINO 도 같은 상수를 씀)" % tag)
    rr = sorted(((pear(b,res[(n,tag)][keep]),n) for n in MODELS if n!="dino"), reverse=True)
    for i,(r,n) in enumerate(rr,1):
        print("    %d위 %-12s r=%.4f" % (i,n,r))

b_own = res[("dino","자기상수")][keep]
print("")
print("  [DINO 는 자기상수(=ImageNet) 고정, 후보만 ImageNet 강제]")
rr = sorted(((pear(b_own,res[(n,"ImageNet강제")][keep]),n) for n in MODELS if n!="dino"), reverse=True)
for i,(r,n) in enumerate(rr,1):
    print("    %d위 %-12s r=%.4f" % (i,n,r))
