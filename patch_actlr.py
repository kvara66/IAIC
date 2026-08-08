"""action_embed 에 별도 학습률을 준다. 원본 대비 최소 수정.

왜 필요한가 (2026-08-09 실측):
  6000스텝 학습한 체크포인트의 action_embed 가중치가 초기화 분포 그대로다.

    Linear(6, 1280)      기대 std 0.235702   실측 0.235320
    Linear(1280, 1280)   기대 std 0.016139   실측 0.016136   소수점 4자리까지 일치

  action_embed 는 DynamiCrafter 백본에 없는 새 모듈이라 무작위로 시작하는데,
  전 파라미터가 lr 1e-5 한 그룹으로 묶여 있어 1.14 에폭 동안 거의 안 움직였다.
  액션 조건화가 사실상 무작위 투영 상태다.

  이건 예전에 시간축 어텐션이 "안 깨어난" 실패와 같은 원인이다.
  그때 얻은 교훈이 "새 모듈만 lr 1e-3 을 따로 주라" 였다.

무엇을 하나:
  configure_optimizers 에서 파라미터를 이름으로 갈라 두 그룹으로 만든다.
    action_embed / null_action_emb  ->  lr x mult
    나머지                          ->  lr

쓰는 법:
  환경변수로 켠다. 설정하지 않으면 원본과 완전히 동일하게 동작한다.
    export IAIC_ACTION_LR_MULT=100     # 1e-5 -> 1e-3

  코드 제출 때는 학습 스크립트에 이 변수를 명시해 재현 가능하게 둘 것.
"""
import shutil
import sys
import ast
from pathlib import Path

DDPM = Path("/workspace/open/baseline/challenge_kit/libs/dynamicrafter/lvdm/models/ddpm3d.py")

ANCHOR = '''    def configure_optimizers(self):
        """configure_optimizers for LatentDiffusion"""
        lr = self.learning_rate
        params = self.get_param_list()
'''

NEW = '''    def configure_optimizers(self):
        """configure_optimizers for LatentDiffusion"""
        lr = self.learning_rate
        params = self.get_param_list()

        # ---- 액션 모듈에만 별도 학습률 (원본에 없던 경로) ----
        # action_embed 는 백본에 없는 새 모듈이라 무작위로 시작하는데, 전 파라미터가
        # 한 그룹으로 묶여 lr 1e-5 를 받으면 1 에폭 동안 초기화 상태에 머문다.
        # 실측: 6000스텝 뒤 가중치 std 가 초기화 분포와 소수점 4자리까지 일치했다.
        import os as _os

        _mult = float(_os.environ.get("IAIC_ACTION_LR_MULT", "1") or 1)
        if abs(_mult - 1.0) > 1e-9:
            _act, _base = [], []
            for _n, _p in self.model.named_parameters():
                if not _p.requires_grad:
                    continue
                if "action_embed" in _n or "null_action_emb" in _n:
                    _act.append(_p)
                else:
                    _base.append(_p)
            _extra = [p for p in params if not any(p is q for q in _act) and not any(p is q for q in _base)]
            params = [{"params": _base, "lr": lr}, {"params": _act, "lr": lr * _mult}]
            if _extra:
                params.append({"params": _extra, "lr": lr})
            mainlogger.info(
                "@Training action_embed lr x%.1f  (%d개 파라미터, 나머지 %d개)"
                % (_mult, len(_act), len(_base))
            )
'''

if not DDPM.exists():
    sys.exit("[실패] ddpm3d.py 가 없다")

bak = DDPM.with_suffix(".py.orig")
if not bak.exists():
    shutil.copy2(DDPM, bak)
    print("원본 백업 -> %s" % bak.name)

src = bak.read_text(encoding="utf-8")
n = src.count(ANCHOR)
if n != 1:
    print("[실패] 앵커가 %d개 (1개여야 한다)" % n)
    sys.exit(1)

out = src.replace(ANCHOR, NEW)
ast.parse(out)
DDPM.write_text(out, encoding="utf-8")
print("적용 완료, 문법 OK")
print("  IAIC_ACTION_LR_MULT 미설정 시 원본과 동일하게 동작")
print("  되돌리려면: cp %s %s" % (bak.name, DDPM.name))
