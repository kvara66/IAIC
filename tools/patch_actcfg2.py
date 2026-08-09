"""DDIM 샘플러에 액션 CFG 를 넣는다. (1차 패치의 배선 오류 수정본)

1차 패치가 왜 안 먹었나:
  act 가 kwargs 에 있는 줄 알고 kwargs.get("act") 를 봤는데, 실제로는
  조건 딕셔너리 c 안에 들어 있다.

    ddpm3d.py:1232   cond = {"act": batch["act"]}
    ddpm3d.py:790    x_recon = self.model(x_noisy, t, **cond, **kwargs)

  c["act"] 가 **cond 로 펼쳐져 UNet 의 act= 인자가 된다.
  prepare_batch_for_inference 가 돌려주는 kwargs 에는 clean_cond 와 fs 뿐이다.
  그래서 조건이 항상 거짓이었고 패치가 통째로 건너뛰어졌다.
  (증거: action_scale 2.0 / 3.0 / 없음 의 결과가 소수점까지 동일했다)

무엇을 하나:
  이미지 조건은 그대로 두고 액션만 증폭한다. c 를 복사해 act 만 None 으로 바꾼다.
    e_t_full  = UNet(c 그대로)
    e_t_noact = UNet(c 에서 act 만 None)
    e_t = e_t_noact + scale * (e_t_full - e_t_noact)

  act=None 이면 UNet 이 self.null_action_emb 를 쓴다. 이번 학습에
  action_dropout_prob 0.1 을 켜서 그 파라미터가 실제로 학습됐다(노름 5.33).

비용: UNet 호출이 1회 -> 2회. 25스텝이 50스텝 1회와 같은 비용이다.

쓰는 법: eval yaml 의 ddim_kwargs 에 action_guidance_scale 을 넣는다.
        값이 없거나 1.0 이면 원본과 완전히 동일하게 동작한다.
        경로를 실제로 탔는지는 로그의 "[액션CFG]" 한 줄로 확인한다.
"""
import shutil
import sys
import ast
from pathlib import Path

DDIM = Path("/workspace/open/baseline/challenge_kit/libs/dynamicrafter/lvdm/models/samplers/ddim.py")
BAK = DDIM.with_suffix(".py.orig")

ANCHOR = """        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)  # unet denoiser
        else:"""

NEW = """        # ---- 액션 CFG (원본에 없던 경로) ----
        # act 는 kwargs 가 아니라 조건 딕셔너리 c 안에 있다(ddpm3d.py:1232).
        # 이미지 조건은 그대로 두고 액션만 증폭한다.
        _act_scale = kwargs.pop("action_guidance_scale", 1.0)
        _has_act = isinstance(c, dict) and c.get("act", None) is not None
        if _act_scale is not None and abs(float(_act_scale) - 1.0) > 1e-6 and _has_act:
            if not getattr(self, "_actcfg_logged", False):
                print("[액션CFG] scale=%s 로 동작 중" % _act_scale)
                self._actcfg_logged = True
            e_t_full, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)
            _c_noact = dict(c)
            _c_noact["act"] = None
            e_t_noact, _, _ = self.model.model_predictions(x, t, _c_noact, dropout_actions=False, **kwargs)
            e_t = e_t_noact + float(_act_scale) * (e_t_full - e_t_noact)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t, _, _ = self.model.model_predictions(x, t, c, dropout_actions=False, **kwargs)  # unet denoiser
        else:"""

if not BAK.exists():
    sys.exit("[실패] 원본 백업 %s 가 없다" % BAK.name)

src = BAK.read_text(encoding="utf-8")          # 항상 원본에서 다시 시작
n = src.count(ANCHOR)
if n != 1:
    sys.exit("[실패] 앵커가 %d개 (1개여야 한다)" % n)

out = src.replace(ANCHOR, NEW)
ast.parse(out)
DDIM.write_text(out, encoding="utf-8")
print("적용 완료, 문법 OK")
print("  원본에서 다시 적용했으므로 1차 패치의 잘못된 코드는 남지 않는다")
print("  action_guidance_scale 등장 %d회" % out.count("action_guidance_scale"))
print('  조건을 c 에서 확인: %s' % ('있음' if 'c.get(' in out else '없음'))
