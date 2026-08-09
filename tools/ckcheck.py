"""체크포인트의 action_embed 가 실제로 학습됐는지 본다.

판정 기준:
  PyTorch Linear 기본 초기화는 U(-1/sqrt(fan_in), +1/sqrt(fan_in)) 이고
  그 분포의 표준편차는 (2/sqrt(fan_in)) / sqrt(12) 이다.

    Linear(6, 1280)      기대 std 0.235702
    Linear(1280, 1280)   기대 std 0.016139

  6000스텝 학습본은 실측이 0.235320 / 0.016136 으로 소수점 4자리까지 일치했다.
  즉 전혀 안 움직였다. 학습이 됐다면 이 값에서 벗어나야 한다.

종료코드 0 = 움직임, 1 = 안 움직임 (중단 판정용)
"""
import math
import sys
import torch

path = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0   # 몇 % 이상 벗어나면 학습된 것으로

ck = torch.load(path, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)


def init_std(fan_in):
    return (2.0 / math.sqrt(fan_in)) / math.sqrt(12.0)


rows = []
for k, v in sd.items():
    if "action_embed" not in k or not k.endswith("weight"):
        continue
    t = v.float()
    fan_in = t.shape[1]
    exp = init_std(fan_in)
    got = t.std().item()
    dev = 100.0 * abs(got - exp) / exp
    rows.append((k, fan_in, exp, got, dev))

if not rows:
    print("[실패] action_embed 를 못 찾았다")
    sys.exit(2)

print("%-42s %8s %10s %10s %8s" % ("층", "fan_in", "기대std", "실측std", "차이"))
print("-" * 84)
moved = False
for k, fan_in, exp, got, dev in rows:
    mark = "  <- 움직임" if dev >= thresh else ""
    if dev >= thresh:
        moved = True
    print("%-42s %8d %10.6f %10.6f %7.2f%%%s" % (k.replace("model.diffusion_model.", ""), fan_in, exp, got, dev, mark))

nz = sd.get("model.diffusion_model.null_action_emb")
if nz is not None:
    print("")
    print("null_action_emb  std %.6f  absmax %.6f" % (nz.float().std().item(), nz.float().abs().max().item()))

print("")
if moved:
    print("판정: 학습됨. 계속 진행한다.")
    sys.exit(0)
print("판정: 초기화 상태 그대로다. 패치가 안 먹었다. 중단할 것.")
sys.exit(1)
