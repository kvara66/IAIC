"""[폐기됨 — 이 패치는 아무 일도 하지 않습니다. tools/patch_actcfg2.py 를 쓰세요]

    이 1차 패치는 act 가 kwargs 에 있는 줄 알고 kwargs.get("act") 를 확인하는데,
    실제로 act 는 조건 딕셔너리 c 안에 있습니다(ddpm3d.py:1232 cond = {"act": ...}).
    그래서 조건이 항상 거짓이라 패치 전체가 건너뛰어집니다.
    증거: action_guidance_scale 2.0 / 3.0 / 미설정의 결과가 소수점까지 동일했습니다.

    수정본은 tools/patch_actcfg2.py 이고, 경로를 실제로 탔는지 확인하도록
    "[액션CFG]" 로그 한 줄을 찍습니다. 안 찍히면 또 안 먹은 것입니다.

--- 이하 원문 (기록용) ---

DDIM 샘플러에 액션 CFG 를 넣는다. 원본 대비 최소 수정.

왜 필요한가:
  지금 샘플러는 조건부·무조건부 양쪽 모두 dropout_actions=False 로 호출하고
  act 는 **kwargs 로 양쪽에 똑같이 들어간다. 즉 guidance 를 올려도
  이미지 조건만 증폭되고 액션 조건은 전혀 증폭되지 않는다.

  홀드아웃 측정으로 확인한 것: 모델은 액션의 타이밍은 맞히는데(정답·생성 모두
  lag +1 에서 최고 상관) 세기가 약하다(달성률 43%). 그래서 필요한 건
  "언제" 가 아니라 "얼마나 강하게" 다.

무엇을 하나:
  이미지 조건은 그대로 두고 액션만 증폭한다.
    e_t_full  = UNet(이미지 조건 + 액션)
    e_t_noact = UNet(이미지 조건 + 액션 없음)
    e_t = e_t_noact + act_scale * (e_t_full - e_t_noact)

  act=None 이면 UNet 이 self.null_action_emb 를 쓴다. 이건 이미 모델에 있는
  nn.Parameter 인데 action_dropout_prob=0.0 이라 학습 중 한 번도 안 쓰여
  정확히 0 으로 남아 있다. 즉 "액션 변조 없음" 상태다. 모델이 학습 때 본 적은
  없으므로 잘 되리란 보장은 없다 - 그래서 홀드아웃으로 먼저 잰다.

비용:
  UNet 호출이 1회 -> 2회. 2배 느려진다.
  50스텝 + 액션CFG 는 1시간 제한을 넘으므로 25스텝으로 써야 한다.

쓰는 법:
  eval yaml 의 ddim_kwargs 에 action_guidance_scale 을 넣는다.
    ddim_kwargs:
      ddim_steps: 25
      unconditional_guidance_scale: 1.0
      action_guidance_scale: 2.0
  값이 없거나 1.0 이면 원본과 완전히 동일하게 동작한다.
"""
import shutil
import sys
import ast
from pathlib import Path

DDIM = Path("/workspace/open/baseline/challenge_kit/libs/dynamicrafter/lvdm/models/samplers/ddim.py")

ANCHOR = """        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)  # unet denoiser
        else:"""

NEW = """        # ---- 액션 CFG (원본에 없던 경로) ----
        # 이미지 조건은 그대로 두고 액션만 증폭한다. act=None 이면 UNet 이
        # null_action_emb 를 써서 "액션 변조 없음" 예측을 낸다.
        _act_scale = kwargs.pop("action_guidance_scale", 1.0)
        if _act_scale is not None and abs(float(_act_scale) - 1.0) > 1e-6 and kwargs.get("act", None) is not None:
            e_t_full, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)
            _noact = dict(kwargs)
            _noact["act"] = None
            e_t_noact, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **_noact)
            e_t = e_t_noact + float(_act_scale) * (e_t_full - e_t_noact)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)  # unet denoiser
        else:"""

if not DDIM.exists():
    sys.exit("[실패] ddim.py 가 없다: %s" % DDIM)

bak = DDIM.with_suffix(".py.orig")
if not bak.exists():
    shutil.copy2(DDIM, bak)
    print("원본 백업 -> %s" % bak.name)

src = bak.read_text(encoding="utf-8")
n = src.count(ANCHOR)
if n != 1:
    print("[실패] 앵커가 %d개 (1개여야 한다)" % n)
    print("        찾던 것:")
    print(ANCHOR)
    sys.exit(1)

out = src.replace(ANCHOR, NEW)
ast.parse(out)
DDIM.write_text(out, encoding="utf-8")
print("적용 완료, 문법 OK")
print("")
print("확인:")
print("  action_guidance_scale 등장 %d회" % out.count("action_guidance_scale"))
print("  되돌리려면: cp %s %s" % (bak, DDIM))
