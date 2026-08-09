"""액션 임베딩이 시간 임베딩 대비 얼마나 커졌는지 잰다. GPU 불필요.

왜 보나:
  UNet 은 emb = time_emb + act_emb 로 둘을 더해 쓴다. 그런데 action_embed 에만
  lr 을 1000배 줬으므로 계속 자란다. step 667 에서 이미 초기값의 2.8배다
  (std 0.016 -> 0.045).

  너무 커지면 액션 신호가 시간 임베딩을 압도해서 확산 과정 자체가 망가진다.
  체크포인트마다 이 비율을 재면 폭주 여부를 알 수 있고, 다음 학습의 lr 배수를
  정하는 근거가 된다.

무엇을 재나:
  실제 홀드아웃 액션을 정규화해 action_embed 에 통과시킨 출력의 크기와,
  실제 확산 타임스텝을 time_embed 에 통과시킨 출력의 크기를 비교한다.
  가중치 노름이 아니라 실제 출력이라 의미가 분명하다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

KIT = Path("/workspace/open/baseline/challenge_kit")
sys.path.insert(0, str(KIT / "libs/dynamicrafter"))
from lvdm.modules.networks.openaimodel3d import timestep_embedding

CK = sys.argv[1]
STATS = sys.argv[2] if len(sys.argv) > 2 else "/workspace/open/data/train/so100_action_statistics.json"
HOLD = Path("/workspace/holdout/actions")

sd = torch.load(CK, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
P = "model.diffusion_model."


def build(prefix, in_dim, hid, out_dim):
    m = nn.Sequential(nn.Linear(in_dim, hid), nn.SiLU(), nn.Linear(hid, out_dim))
    m[0].weight.data = sd[prefix + "0.weight"].float()
    m[0].bias.data = sd[prefix + "0.bias"].float()
    m[2].weight.data = sd[prefix + "2.weight"].float()
    m[2].bias.data = sd[prefix + "2.bias"].float()
    return m.eval()


act_mlp = build(P + "action_embed.", 6, 1280, 1280)
time_mlp = build(P + "time_embed.", 320, 1280, 1280)

# 실제 액션을 정규화해 통과시킨다
st = json.load(open(STATS))
mu = torch.tensor(st["mean"], dtype=torch.float32)
sg = torch.tensor(st["std"], dtype=torch.float32)
acts = np.stack([np.load(p) for p in sorted(HOLD.glob("*.npy"))]).astype(np.float32)
a = (torch.from_numpy(acts) - mu) / sg                     # (N, 16, 6)

# 실제 확산 타임스텝 (1000 스텝 중 고르게)
t = torch.linspace(0, 999, 50).long()
with torch.no_grad():
    ae = act_mlp(a.reshape(-1, 6))
    te = time_mlp(timestep_embedding(t, 320, repeat_only=False).float())

an = ae.norm(dim=1)
tn = te.norm(dim=1)
null = sd.get(P + "null_action_emb")

print("체크포인트 %s" % Path(CK).name)
print("")
print("  액션 임베딩 노름   평균 %8.3f   표준편차 %7.3f" % (an.mean(), an.std()))
print("  시간 임베딩 노름   평균 %8.3f   표준편차 %7.3f" % (tn.mean(), tn.std()))
print("  비율 (액션/시간)   %.4f" % (an.mean() / tn.mean()))
if null is not None:
    print("  null_action_emb 노름 %.4f" % null.float().norm())
print("")
print("  읽는 법: 비율이 1 을 크게 넘으면 액션이 시간 신호를 압도한다.")
print("           0.05 미만이면 사실상 무시되고 있다.")
