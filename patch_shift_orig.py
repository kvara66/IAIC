"""데이콘 원본 소스에 action_shift 를 넣는다. (예전 pod 의 수정본이 아니라 원본 기준)

근거(실측): 홀드아웃 정답 영상에서 화면 변화량과 액션 변화량의 상관이
            lag 0 에서 0.410, lag +1 에서 0.537. 화면이 액션보다 한 프레임 늦다.
            실제 로봇의 구동 지연으로 보인다.

문제: action_embed 이 nn.Linear 라 프레임마다 독립이다. t 번 프레임에서 a[t] 만 보는데
      그 프레임에 그려야 할 움직임은 a[t-1] 이 만든 것이다. 모델이 스스로 못 당겨온다.

해결: 액션을 한 칸 밀어서 넣는다. 첫 프레임은 이전이 없으므로 a[0] 을 반복.

주의 1: 학습과 추론이 어긋나면 조건이 통째로 무의미해진다.
        그래서 eval yaml 에 없으면 model_config_file(학습 설정) 값을 따라가게 한다.
주의 2: feature_csv_utils.py 에는 액션을 읽는 곳이 두 군데다.
        하나는 조건 입력(build_inference_batch), 다른 하나는 채점용이다.
        채점용은 절대 밀면 안 된다 - 주어진 정답 액션과 비교해야 하므로.
        그래서 ndim 검사가 있는 조건 입력 쪽만 앵커로 잡았다.

action_shift: 0 이면 기존과 완전히 동일하다.
"""
import shutil, sys, ast
from pathlib import Path

K = Path("/workspace/open/baseline/challenge_kit")
DS = K / "src/ldwma/datasets/lerobot_so100.py"
DM = K / "src/ldwma/lightning/data_modules/lerobot_so100.py"
FC = K / "scripts/eval/feature_csv_utils.py"
GB = K / "scripts/inference/generate_baseline_videos.py"

SHIFT_DS = (
    "        if getattr(self, 'action_shift', 0) > 0:\n"
    "            # 화면 변화가 액션 변화보다 한 프레임 늦다(상관 lag+1 0.537 > lag0 0.410).\n"
    "            # action_embed 이 프레임마다 독립이라 모델이 스스로 못 당겨온다. 미리 밀어서 준다.\n"
    "            k = self.action_shift\n"
    "            actions = np.concatenate([np.repeat(actions[:1], k, axis=0), actions[:-k]], axis=0)\n"
)

EDITS = [
    # ---------- 데이터셋 (학습 경로) ----------
    (DS,
     "        action_mean: Sequence[float] | None = None,\n"
     "        action_std: Sequence[float] | None = None,\n",
     "        action_mean: Sequence[float] | None = None,\n"
     "        action_std: Sequence[float] | None = None,\n"
     "        action_shift: int = 0,\n", 1),

    (DS,
     "        self.action_std = torch.tensor(action_std, dtype=torch.float32) if action_std is not None else None\n",
     "        self.action_std = torch.tensor(action_std, dtype=torch.float32) if action_std is not None else None\n"
     "        self.action_shift = action_shift\n", 1),

    (DS,
     '        actions = np.stack(table["action"].iloc[frame_indices].to_numpy()).astype(np.float32)\n'
     "        if actions.shape[-1] != 6:\n"
     "            raise ValueError(f\"Expected SO-100 action dim 6, got {actions.shape[-1]} from {example['data_ref']}.\")\n",
     '        actions = np.stack(table["action"].iloc[frame_indices].to_numpy()).astype(np.float32)\n'
     "        if actions.shape[-1] != 6:\n"
     "            raise ValueError(f\"Expected SO-100 action dim 6, got {actions.shape[-1]} from {example['data_ref']}.\")\n"
     + SHIFT_DS, 1),

    # ---------- 데이터모듈 ----------
    (DM,
     "        action_stats_path: str | None = None,\n",
     "        action_stats_path: str | None = None,\n        action_shift: int = 0,\n", 1),

    (DM,
     "        self.action_stats_path = action_stats_path\n",
     "        self.action_stats_path = action_stats_path\n        self.action_shift = action_shift\n", 1),

    (DM,
     "        self.train_dataset = LeRobotSO100Dataset(\n            root=self.root,\n",
     "        self.train_dataset = LeRobotSO100Dataset(\n            root=self.root,\n"
     "            action_shift=self.action_shift,\n", 1),

    (DM,
     "        self.val_dataset = LeRobotSO100Dataset(\n            root=self.root,\n",
     "        self.val_dataset = LeRobotSO100Dataset(\n            root=self.root,\n"
     "            action_shift=self.action_shift,\n", 1),

    # ---------- 추론 경로 (조건 입력 쪽만) ----------
    (FC,
     "    action_std: torch.Tensor | None,\n    device: torch.device,\n) -> dict:\n",
     "    action_std: torch.Tensor | None,\n    device: torch.device,\n    action_shift: int = 0,\n) -> dict:\n", 1),

    (FC,
     '        action = np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32)\n'
     "        if action.ndim != 2:\n"
     '            raise ValueError(f"{sample_id}: action must have shape (T, A), got {action.shape}.")\n',
     '        action = np.load(challenge_root / "actions" / f"{sample_id}.npy").astype(np.float32)\n'
     "        if action.ndim != 2:\n"
     '            raise ValueError(f"{sample_id}: action must have shape (T, A), got {action.shape}.")\n'
     "        if action_shift > 0:\n"
     "            # 학습과 동일하게 밀어야 한다. 어긋나면 조건이 통째로 무의미해진다.\n"
     "            action = np.concatenate([np.repeat(action[:1], action_shift, axis=0),\n"
     "                                     action[:-action_shift]], axis=0)\n", 1),

    # ---------- 추론 스크립트 ----------
    (GB,
     "def load_defaults(config_path: str) -> dict:\n    config = OmegaConf.load(config_path)\n    return {\n",
     "def load_defaults(config_path: str) -> dict:\n    config = OmegaConf.load(config_path)\n"
     "    # action_shift 가 학습과 어긋나면 조건이 무의미해지는데 에러가 안 나서 잡기 어렵다.\n"
     "    # eval yaml 에 없으면 학습 설정(체크포인트와 짝) 값을 따라가게 해서 어긋날 수 없게 한다.\n"
     "    _mcf = OmegaConf.select(config, \"model_config_file\")\n"
     "    _train = OmegaConf.load(_mcf) if _mcf else None\n"
     "\n"
     "    def train_value(key, default):\n"
     "        if _train is None:\n"
     "            return default\n"
     "        v = OmegaConf.select(_train, key)\n"
     "        return default if v is None else v\n"
     "\n"
     "    return {\n", 1),

    (GB,
     '        "precision": 16,\n    }\n',
     '        "precision": 16,\n'
     '        "action_shift": config_value(config, "data.params.action_shift",\n'
     '                                    train_value("data.params.action_shift", 0)),\n    }\n', 1),

    (GB,
     '    parser.add_argument("--seed", type=int, default=0)\n',
     '    parser.add_argument("--action-shift", type=int, default=defaults["action_shift"])\n'
     '    parser.add_argument("--seed", type=int, default=0)\n', 1),

    (GB,
     "            action_mean,\n            action_std,\n            device,\n        )\n",
     "            action_mean,\n            action_std,\n            device,\n            args.action_shift,\n        )\n", 1),
]

# 원본 백업 (한 번만)
for p in {e[0] for e in EDITS}:
    b = p.with_suffix(p.suffix + ".orig")
    if not b.exists():
        shutil.copy2(p, b)
        print("백업 %s" % b.name)

cache = {}
for p, old, new, cnt in EDITS:
    if p not in cache:
        cache[p] = p.with_suffix(p.suffix + ".orig").read_text(encoding="utf-8")
    n = cache[p].count(old)
    if n != cnt:
        print("[실패] %s: 일치 %d개 (기대 %d개)" % (p.name, n, cnt))
        print("        찾던 것: %r" % old[:80])
        sys.exit(1)
    cache[p] = cache[p].replace(old, new)

for p, txt in cache.items():
    ast.parse(txt)
    p.write_text(txt, encoding="utf-8")
    print("적용 %s  문법 OK" % p.name)

print()
import subprocess
for p in cache:
    r = subprocess.run(["grep", "-c", "action_shift", str(p)], capture_output=True, text=True)
    print("  %-32s action_shift %s 곳" % (p.name, r.stdout.strip()))
