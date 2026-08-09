"""액션을 첫 프레임 기준으로 중심화한다. 원본 대비 최소 수정.

왜 필요한가 (2026-08-09 실측):
  액션 절대값에는 "어느 연구실 데이터인가" 라는 오프셋이 섞여 있다.
  차원 4(손목 회전)의 데이터셋별 평균이 -109.5 에서 +184.0 까지 흩어지는데,
  이건 움직임이 아니라 손목 장착 방향 차이다.

  그리고 eval 은 그 오프셋이 train 과 크게 다르다.

    train 통계로 정규화했을 때 eval 평균의 이동
      절대값 그대로   [-0.04 -0.09 -0.01  0.24  1.26 -0.04]   최대 1.26 시그마
      첫 프레임 중심화 [ 0.01 -0.11  0.04 -0.11  0.02  0.45]   최대 0.45 시그마

  차원 4 의 1.26 시그마가 0.02 로 사라진다. eval 평균(+64.4) 이상인 train
  에피소드는 6.6% 뿐이라, 모델이 거의 못 본 구간의 입력을 받고 있었다.

  중심화해도 정보 손실이 없다. 시작 자세는 첫 프레임 이미지가 알려준다.

⚠ 정규화 통계를 반드시 같이 바꿔야 한다:
  중심화하면 값의 크기가 훨씬 작아지는데 절대값 기준 std 로 나누면
  액션 신호가 오히려 더 약해진다. mkcenterstats.py 로 중심화용 통계를
  만들어 --action-stats-path 로 넘길 것.

쓰는 법:
  학습 yaml:  data.params.action_center: True
  추론 yaml:  data.params.action_center: True   (없으면 학습 설정을 따라감)
  값이 없거나 False 면 원본과 완전히 동일하게 동작한다.
"""
import shutil
import sys
import ast
from pathlib import Path

K = Path("/workspace/open/baseline/challenge_kit")
DS = K / "src/ldwma/datasets/lerobot_so100.py"
DM = K / "src/ldwma/lightning/data_modules/lerobot_so100.py"
FC = K / "scripts/eval/feature_csv_utils.py"
GB = K / "scripts/inference/generate_baseline_videos.py"

CENTER_DS = (
    "        if getattr(self, 'action_center', False):\n"
    "            # 데이터셋별 손목 오프셋을 뺀다. 차원4 의 데이터셋별 평균이\n"
    "            # -109.5 ~ +184.0 으로 흩어지는데 이건 움직임이 아니라 장착 방향이다.\n"
    "            actions = actions - actions[:1]\n"
)

EDITS = [
    # ---------- 데이터셋 (학습) ----------
    (DS,
     "        action_shift: int = 0,\n",
     "        action_shift: int = 0,\n        action_center: bool = False,\n", 1),

    (DS,
     "        self.action_shift = action_shift\n",
     "        self.action_shift = action_shift\n        self.action_center = action_center\n", 1),

    # action_shift 다음에 중심화한다. 순서를 고정해 두 옵션이 겹쳐도 결과가 정해지게.
    (DS,
     "            actions = np.concatenate([np.repeat(actions[:1], k, axis=0), actions[:-k]], axis=0)\n",
     "            actions = np.concatenate([np.repeat(actions[:1], k, axis=0), actions[:-k]], axis=0)\n"
     + CENTER_DS, 1),

    # ---------- 데이터모듈 ----------
    (DM,
     "        action_shift: int = 0,\n",
     "        action_shift: int = 0,\n        action_center: bool = False,\n", 1),

    (DM,
     "        self.action_shift = action_shift\n",
     "        self.action_shift = action_shift\n        self.action_center = action_center\n", 1),

    (DM,
     "            action_shift=self.action_shift,\n            root=self.root,\n",
     "            action_shift=self.action_shift,\n            action_center=self.action_center,\n            root=self.root,\n", 0),

    # ---------- 추론 (조건 입력 쪽만) ----------
    (FC,
     "    action_shift: int = 0,\n) -> dict:\n",
     "    action_shift: int = 0,\n    action_center: bool = False,\n) -> dict:\n", 1),

    (FC,
     "            action = np.concatenate([np.repeat(action[:1], action_shift, axis=0),\n"
     "                                     action[:-action_shift]], axis=0)\n",
     "            action = np.concatenate([np.repeat(action[:1], action_shift, axis=0),\n"
     "                                     action[:-action_shift]], axis=0)\n"
     "        if action_center:\n"
     "            # 학습과 반드시 동일하게. 어긋나면 조건이 통째로 무의미해진다.\n"
     "            action = action - action[:1]\n", 1),

    # ---------- 추론 스크립트 ----------
    (GB,
     '        "action_shift": config_value(config, "data.params.action_shift",\n'
     '                                    train_value("data.params.action_shift", 0)),\n',
     '        "action_shift": config_value(config, "data.params.action_shift",\n'
     '                                    train_value("data.params.action_shift", 0)),\n'
     '        "action_center": config_value(config, "data.params.action_center",\n'
     '                                    train_value("data.params.action_center", False)),\n', 1),

    (GB,
     '    parser.add_argument("--action-shift", type=int, default=defaults["action_shift"])\n',
     '    parser.add_argument("--action-shift", type=int, default=defaults["action_shift"])\n'
     '    parser.add_argument("--action-center", action="store_true", default=bool(defaults["action_center"]))\n', 1),

    (GB,
     "            device,\n            args.action_shift,\n        )\n",
     "            device,\n            args.action_shift,\n            args.action_center,\n        )\n", 1),
]

for p in {e[0] for e in EDITS}:
    b = p.with_suffix(p.suffix + ".orig_center")
    if not b.exists():
        shutil.copy2(p, b)
        print("백업 %s" % b.name)

cache = {}
fail = False
for p, old, new, cnt in EDITS:
    if p not in cache:
        cache[p] = p.with_suffix(p.suffix + ".orig_center").read_text(encoding="utf-8")
    n = cache[p].count(old)
    if cnt == 0:                      # 데이터모듈은 train/val 두 군데
        if n < 1:
            print("[실패] %s: 못 찾음 %r" % (p.name, old[:60])); fail = True; continue
    elif n != cnt:
        print("[실패] %s: 일치 %d개 (기대 %d)" % (p.name, n, cnt))
        print("        %r" % old[:70]); fail = True; continue
    cache[p] = cache[p].replace(old, new)

if fail:
    sys.exit(1)

for p, txt in cache.items():
    ast.parse(txt)
    p.write_text(txt, encoding="utf-8")
    print("적용 %-28s 문법 OK  action_center %d곳" % (p.name, txt.count("action_center")))

print("")
print("action_center 를 설정하지 않으면 원본과 동일하게 동작한다.")
print("되돌리려면 각 파일의 .orig_center 백업을 복사할 것.")
