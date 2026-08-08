#!/bin/bash
# 포드를 재시작할 때마다 돌린다.  bash /workspace/setup_env.sh
#
# RunPod 은 /workspace 만 영구 볼륨이고 나머지 컨테이너 파일시스템은 재시작 때
# 초기화된다. pip 로 깐 패키지는 /usr/local 에 들어가므로 전부 날아간다.
# 매번 다섯 군데에서 막혔던 것을 여기 정리해둔다.

set -u
P="pip install --quiet --break-system-packages"   # PEP 668 차단 우회

echo "=== 현재 torch (건드리지 않는다) ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# requirements.txt 의 torch 핀(2.7.1)을 따르면 이미 깔린 2.8.0 을 내려버린다.
# torch 는 베이스 이미지 것을 쓰고 나머지만 넣는다.
echo "=== 기본 패키지 ==="
$P pandas pyarrow kornia "av>=10" "omegaconf==2.1.1" "einops==0.6.0" "imageio>=2.9.0,<3" \
   opencv-python "PyYAML>=6.0.1,<7" "requests>=2.28" "tqdm>=4.60" \
   "pillow>=9.5,<12" "torchmetrics>=0.11,<2" "timm>=0.9,<2"

# 이 두 개는 최신 버전을 깔면 임포트 단계에서 죽는다. 버전을 반드시 고정한다.
echo "=== 버전 고정 패키지 ==="
$P "huggingface-hub==0.25.2" "transformers==4.48.3" "open_clip_torch==2.22.0"

# --no-deps 없이 깔면 torch 를 제 맘대로 바꾼다.
echo "=== pytorch-lightning (--no-deps 필수) ==="
$P --no-deps "pytorch-lightning==1.9.3"
$P "lightning-utilities" "fsspec[http]"     # pl 이 실제로 쓰는 것만 따로

echo ""
echo "=== 확인 ==="
python - <<'EOF'
import importlib, sys
mods = ["torch","numpy","pandas","pyarrow","kornia","PIL","av","omegaconf","einops",
        "pytorch_lightning","transformers","open_clip","imageio","cv2","timm"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print(f"  {m:22s} OK")
    except Exception as e:
        bad.append(m); print(f"  {m:22s} 실패 {type(e).__name__}")
import torch
print(f"\n  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
sys.exit(1 if bad else 0)
EOF
RC=$?

echo ""
echo "=== PYTHONPATH (shared_libs 가 아니라 shared_libs/video_utils 까지 들어가야 한다) ==="
cat > /workspace/env.sh <<'EOF'
KIT=/workspace/open/baseline/challenge_kit
export PYTHONPATH="$KIT/src:$KIT/libs/dynamicrafter:$KIT/../shared_libs/video_utils"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
echo "  /workspace/env.sh 생성.  source /workspace/env.sh 로 적용"

if [ "$RC" -eq 0 ]; then
  echo ""
  echo "환경 준비 완료"
else
  echo ""
  echo "[실패] 위에 실패한 모듈이 있다"
fi
exit $RC
