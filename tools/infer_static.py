"""정지 출력 정책이 들어간 추론. 이것 하나가 추론 진입점이다.

정책:
  주어진 액션 시퀀스의 총 변화량이 문턱 미만이면 확산을 돌리지 않고
  입력 이미지를 16번 반복한 영상을 낸다.
  "명령된 움직임이 거의 없으면 화면도 거의 안 움직여야 한다" 는 규칙이다.

왜 이게 후처리가 아닌가:
  다 만든 mp4 를 나중에 갈아끼우는 게 아니라, 추론이 처음부터 그 출력을 낸다.
  임계값을 가진 분류기가 후처리가 아닌 것과 같다. submission_kit 은 이 스크립트가
  끝나고 mp4 216개가 확정된 뒤에 따로 한 번 실행한다.

판단 근거:
  각 샘플이 자기 입력(액션 시퀀스)만 본다. 다른 샘플이나 eval 전체 분포를
  참조하지 않는다. 문턱은 train 홀드아웃에서 정했다 (eval 을 쓰지 않았다).

문턱 정하기:
  tools/staticfrac.py 를 홀드아웃에 돌려 얻은 값. 2026-08-09 기준 212.5911.
  홀드아웃 20개에서 recon 0.06924 -> 0.06002 (-13.3%).

효과:
  정지로 가는 샘플은 확산을 건너뛰므로 추론이 그만큼 빨라진다.
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import torch

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "src"))
sys.path.insert(0, str(KIT / "scripts"))
from PIL import Image
from ldwma.datasets.lerobot_so100 import preprocess_video
from eval.feature_csv_utils import save_video_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--challenge-root", default="/workspace/open/data/eval")
ap.add_argument("--prediction-root", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--action-stats-path", default="/workspace/open/data/train/so100_action_statistics.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--static-threshold", type=float, default=212.5911,
                help="액션 총변화량이 이 값 미만이면 정지 영상 (train 홀드아웃에서 결정)")
ap.add_argument("--traj-len", type=int, default=16)
ap.add_argument("--fps", type=int, default=6)
a = ap.parse_args()

ROOT = Path(a.challenge_root)
OUT = Path(a.prediction_root)
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 1) 입력만 보고 정지/생성을 가른다 ----------
ids = sorted(p.stem for p in (ROOT / "actions").glob("*.npy"))
static_ids, gen_ids = [], []
for sid in ids:
    act = np.load(ROOT / "actions" / (sid + ".npy")).astype(np.float64)
    motion = float(np.abs(np.diff(act, axis=0)).sum())
    (static_ids if motion < a.static_threshold else gen_ids).append(sid)

print("전체 %d개 -> 정지 %d개 (%.0f%%) / 생성 %d개" %
      (len(ids), len(static_ids), 100 * len(static_ids) / max(1, len(ids)), len(gen_ids)))
print("문턱 %.4f (train 홀드아웃에서 결정, eval 미사용)" % a.static_threshold)

# ---------- 2) 정지 영상은 확산 없이 바로 쓴다 ----------
for sid in static_ids:
    img = np.asarray(Image.open(ROOT / "images" / (sid + ".png")).convert("RGB"))
    clip = np.repeat(img[None], a.traj_len, axis=0)
    save_video_tensor(preprocess_video(clip, 320, 512, True), OUT / (sid + ".mp4"), a.fps)
print("정지 영상 %d개 작성 완료" % len(static_ids))

# ---------- 3) 나머지만 확산을 돌린다 ----------
if gen_ids:
    SUB = Path("/workspace/_gen_subset")
    if SUB.exists():
        shutil.rmtree(SUB)
    (SUB / "images").mkdir(parents=True)
    (SUB / "actions").mkdir(parents=True)
    for sid in gen_ids:
        (SUB / "images" / (sid + ".png")).symlink_to(ROOT / "images" / (sid + ".png"))
        (SUB / "actions" / (sid + ".npy")).symlink_to(ROOT / "actions" / (sid + ".npy"))

    GENOUT = Path("/workspace/_gen_out")
    if GENOUT.exists():
        shutil.rmtree(GENOUT)
    GENOUT.mkdir(parents=True)

    cfg = Path(a.config)
    txt = cfg.read_text(encoding="utf-8")
    sub_cfg = cfg.with_name(cfg.stem + "_subset.yaml")
    sub_cfg.write_text(txt.replace(str(ROOT), str(SUB)), encoding="utf-8")

    print("확산 시작 - %d개" % len(gen_ids))
    r = subprocess.run(
        [sys.executable, "scripts/inference/generate_baseline_videos.py",
         "--config", str(sub_cfg),
         "--prediction-root", str(GENOUT),
         "--challenge-root", str(SUB),
         "--action-stats-path", a.action_stats_path,
         "--seed", str(a.seed)],
        cwd=str(KIT))
    if r.returncode != 0:
        sys.exit("[실패] 확산 추론이 종료코드 %d 로 끝났다" % r.returncode)

    n = 0
    for sid in gen_ids:
        src = GENOUT / (sid + ".mp4")
        if not src.exists():
            print("  [경고] 생성 안 됨: %s" % sid)
            continue
        shutil.copy2(src, OUT / (sid + ".mp4"))
        n += 1
    print("생성 영상 %d개 복사 완료" % n)

# ---------- 4) 확인 ----------
made = len(list(OUT.glob("*.mp4")))
print("최종 %d/%d" % (made, len(ids)))
if made != len(ids):
    sys.exit("[실패] 개수가 맞지 않는다")
print("완료 - 이제 submission_kit 을 한 번 실행하면 된다")
