#!/bin/bash
# 새 포드에서 dinoproxy 를 돌리기 위한 최소 환경.
# 베이스라인 학습용 setup_env.sh 와 달리 여기서는 torch + timm + av 만 있으면 된다.
#
# 주의: 반드시 EU-SE-1 리전에 띄워야 기존 네트워크 볼륨이 붙는다.
#       (마운트가 mfs#eu-se-1.runpod.net 이다)

set -e
echo "===== 파이썬/토치 확인 ====="
python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo ""
echo "===== 필요한 패키지 설치 ====="
pip install -q --no-input timm av imageio numpy || {
  echo "[실패] pip 설치 실패"; exit 1; }

echo ""
echo "===== 임포트 확인 ====="
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "timm", "av", "imageio", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print("  %-10s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-10s 실패 %s" % (m, e)); ok = False
sys.exit(0 if ok else 1)
PY

echo ""
echo "===== 볼륨 확인 ====="
for p in /workspace/open/data/train /workspace/open/submission_kit; do
  if [ -d "$p" ]; then
    echo "  있음   $p"
  else
    echo "  ★없음  $p   <- 볼륨이 안 붙었다. 리전을 확인할 것 (EU-SE-1)"
  fi
done
N=$(find /workspace/open/data/train -name "*.mp4" 2>/dev/null | wc -l)
echo "  train mp4 $N 개"

echo ""
echo "준비 완료. 다음 순서로 돌린다:"
echo "  cd /workspace/dinoproxy"
echo "  python verify.py --kit /workspace/open/submission_kit --model"
echo "  python mkpairs.py --out /workspace/dinopairs"
echo "  python extract.py --pairs /workspace/dinopairs --determinism"
echo "  python corr.py    --pairs /workspace/dinopairs"
