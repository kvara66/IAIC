"""정지 비율을 바꿔가며 제출용 영상 집합을 다시 만든다.

이미 만들어둔 6000스텝 생성 영상 216개를 재활용하므로 추론이 필요 없다.
입력 액션의 움직임 크기로 정렬해서, 적게 움직이는 쪽부터 pct% 를 원본 정지 프레임으로 바꾼다.
"""
import sys, shutil
import numpy as np
from pathlib import Path
from PIL import Image

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
from ldwma.datasets.lerobot_so100 import preprocess_video
from eval.feature_csv_utils import save_video_tensor

pct = int(sys.argv[1])
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/runs/submit_p6k/videos")
OUTROOT = sys.argv[3] if len(sys.argv) > 3 else "mix"
IMG = Path("/workspace/open/data/eval/images")
ACT = Path("/workspace/open/data/eval/actions")

OUT = Path(f"/workspace/runs/{OUTROOT}_{pct:03d}/videos")
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.mp4"):
    f.unlink()

vids = sorted(SRC.glob("*.mp4"))
if len(vids) != 216:
    print(f"[경고] 원본 영상이 {len(vids)}개")

mot = {v.stem: float(np.abs(np.diff(np.load(ACT / (v.stem + ".npy")).astype(np.float64), axis=0)).sum())
       for v in vids}
order = sorted(vids, key=lambda v: mot[v.stem])      # 덜 움직이는 것부터
nstat = round(len(vids) * pct / 100)
static_set = {v.stem for v in order[:nstat]}

for v in vids:
    if v.stem in static_set:
        img = np.asarray(Image.open(IMG / (v.stem + ".png")).convert("RGB"))
        t = preprocess_video(np.repeat(img[None], 16, axis=0), 320, 512, True)
        save_video_tensor(t, OUT / v.name, 6)
    else:
        shutil.copy2(v, OUT / v.name)

made = len(list(OUT.glob("*.mp4")))
print(f"  정지 {nstat}개 / 생성 {len(vids)-nstat}개 / 최종 {made}개")
if made != 216:
    print(f"  [실패] 216개가 아님")
    sys.exit(1)
