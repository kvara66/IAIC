"""정지 영상 216개를 입력 PNG 에서 바로 만든다.

mixfrac.py 는 기존 mp4 목록에서 샘플 id 를 읽는 구조라, 영상이 없는 상태에서는 못 쓴다.
정지 영상은 모델과 무관하고(입력 이미지를 16번 반복) 합칠 때 반드시 필요한 후보라
어떤 상황에서도 다시 만들 수 있어야 한다.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
from ldwma.datasets.lerobot_so100 import preprocess_video
from eval.feature_csv_utils import save_video_tensor

IMG = Path("/workspace/open/data/eval/images")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/pure_sta_v/videos")
OUT.mkdir(parents=True, exist_ok=True)

pngs = sorted(IMG.glob("*.png"))
made = 0
for p in pngs:
    dst = OUT / (p.stem + ".mp4")
    if dst.exists():
        made += 1
        continue
    img = np.asarray(Image.open(p).convert("RGB"))
    t = preprocess_video(np.repeat(img[None], 16, axis=0), 320, 512, True)
    save_video_tensor(t, dst, 6)
    made += 1
print(f"  정지 영상 {made}개 (입력 PNG {len(pngs)}개)")
if made != 216:
    sys.exit(1)
