"""트리플 체크 - common.py 의 재구현이 채점기 원본과 수치적으로 같은지 증명한다.

이 스크립트가 하는 일
  submission_kit/feature_csv_utils.py 의 함수와 common.py 의 함수에
  같은 입력을 넣어 출력 차이가 0 인지 본다. 코드 동일성 증명이다.

이 스크립트가 하지 않는 일
  submission_kit 을 고치지 않는다 (읽기만 한다).
  eval 데이터를 건드리지 않는다. 무작위 텐서와 train 프레임만 쓴다.
  여기서 나온 값을 모델 선택이나 학습에 쓰지 않는다. 오직 재구현 검증용이다.

왜 필요한가
  전처리가 한 군데라도 어긋나면 "DINO 가 무엇을 보는가" 자체가 달라져서
  상관계수 전체가 무의미해진다. 그런데 어긋나도 에러가 안 나고 그럴듯한
  숫자가 나온다. 그래서 숫자로 증명해 두어야 한다.

검사 항목
  A  정규화 상수                IMAGENET_MEAN / STD
  B  1단계 preprocess_video     여러 해상도에서 최대 오차 0
  C  1단계 to_eval_uint8        uint8 왕복까지 완전 일치
  D  2단계 _resize_pad_frame_batch  여러 입력 크기에서 최대 오차 0
  E  resolve_dino_image_size    같은 값을 돌려주나
  F  _normalize_image_model_output  여러 출력 모양에서 일치
  G  (선택 --model) DINO 실제 특징이 채점기 경로와 완전히 같은가

사용:
  python verify.py --kit /workspace/open/submission_kit
  python verify.py --kit /workspace/open/submission_kit --model      # G 까지
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

FAIL = []


def check(label: str, diff: float, tol: float = 0.0) -> None:
    ok = diff <= tol
    print("  %-46s 최대오차 %.3e   %s" % (label, diff, "통과" if ok else "★실패★"))
    if not ok:
        FAIL.append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default="/workspace/open/submission_kit")
    ap.add_argument("--model", action="store_true", help="DINO 실물까지 대조 (G)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    kit = Path(args.kit)
    if not (kit / "feature_csv_utils.py").exists():
        raise SystemExit("[실패] 채점기를 못 찾았다: %s" % kit)
    sys.path.insert(0, str(kit))
    import feature_csv_utils as K       # 읽기만 한다. 절대 고치지 않는다.

    torch.manual_seed(0)
    np.random.seed(0)

    print("=== A. 정규화 상수 ===")
    check("IMAGENET_MEAN", float((common.IMAGENET_MEAN - K.IMAGENET_MEAN).abs().max()))
    check("IMAGENET_STD", float((common.IMAGENET_STD - K.IMAGENET_STD).abs().max()))

    print("")
    print("=== B. 1단계 preprocess_video ===")
    for (h, w) in [(480, 640), (320, 512), (400, 680), (256, 256), (720, 1280), (100, 999)]:
        v = np.random.randint(0, 256, (4, h, w, 3), dtype=np.uint8)
        for pad in (True, False):
            a = common.stage1_video_norm(v, 320, 512, pad)
            b = K.preprocess_video(v, 320, 512, pad)
            check("%dx%d pad=%s  모양 %s" % (h, w, pad, tuple(a.shape)),
                  float((a - b).abs().max()))

    print("")
    print("=== C. 1단계 to_eval_uint8 (uint8 왕복) ===")
    for (h, w) in [(480, 640), (320, 512), (200, 300)]:
        v = np.random.randint(0, 256, (3, h, w, 3), dtype=np.uint8)
        a = common.stage1_to_uint8(v, 320, 512, True)
        b = K.to_eval_uint8(v, 320, 512, True)
        check("%dx%d  dtype %s" % (h, w, a.dtype),
              float((a.int() - b.int()).abs().max()))

    print("")
    print("=== D. 2단계 _resize_pad_frame_batch ===")
    for size in [518, 384, 336, 224, 112]:
        f = torch.randint(0, 256, (5, 3, 320, 512), dtype=torch.uint8)
        a = common.stage2_resize_pad(f, size, 0.0)
        b = K._resize_pad_frame_batch(f, size, 0.0)
        frac = common.content_fraction(320, 512, size)
        check("size=%-4d  모양 %s  내용 %.1f%%" % (size, tuple(a.shape), frac * 100),
              float((a - b).abs().max()))

    print("")
    print("=== E. resolve_dino_image_size ===")

    class _PE:
        def __init__(self, s):
            self.img_size = s

    class _M(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            self.patch_embed = _PE(s)

    for s, req in [((518, 518), 0), ((224, 224), 0), ((224, 224), 336), (518, 0), (None, 0)]:
        m = _M(s)
        a = common.model_input_size(m, req)
        b = K.resolve_dino_image_size(m, req)
        check("img_size=%s requested=%s -> %s" % (s, req, a), float(abs(a - b)))

    print("")
    print("=== F. _normalize_image_model_output ===")
    cases = {
        "(N,D) 그대로": torch.randn(7, 384),
        "(N,T,D) -> CLS": torch.randn(7, 1370, 384),
        "(N,C,H,W) -> 평균": torch.randn(7, 384, 4, 4),
        "tuple": (torch.randn(7, 384), torch.randn(7, 8)),
        "dict x_norm_clstoken": {"x_norm_clstoken": torch.randn(7, 384)},
        "dict features": {"features": torch.randn(7, 512)},
    }
    for label, o in cases.items():
        a = common.pool_output(o)
        b = K._normalize_image_model_output(o)
        check("%-24s -> %s" % (label, tuple(a.shape)), float((a - b).abs().max()))

    # ---------------------------------------------------------------- G
    if args.model:
        print("")
        print("=== G. DINO 실물 대조 (전체 경로) ===")
        device = torch.device(args.device)
        model = K.load_dino_model(device, "vit_small_patch14_dinov2.lvd142m", pretrained=True)
        size_k = K.resolve_dino_image_size(model, requested_size=0)
        size_c = common.model_input_size(model, 0)
        check("입력 크기 (채점기 %d, 우리 %d)" % (size_k, size_c), float(abs(size_k - size_c)))

        mean, std = common.normalize_stats(model)
        check("정규화 mean 이 ImageNet 과 같은가",
              float((mean - common.IMAGENET_MEAN).abs().max()))
        check("정규화 std 가 ImageNet 과 같은가",
              float((std - common.IMAGENET_STD).abs().max()))

        videos = torch.randint(0, 256, (2, 4, 320, 512, 3), dtype=torch.uint8)
        kit_feat = K.extract_dino_features(videos, model, device, size_k)   # (B,T,D)

        b, t = videos.shape[0], videos.shape[1]
        frames = videos.permute(0, 1, 4, 2, 3).reshape(b * t, 3, 320, 512)
        our_feat = common.extract_features(frames, model, device, size_c, mean, std, 32)
        our_feat = our_feat.reshape(b, t, -1)

        check("특징 모양 %s vs %s" % (tuple(our_feat.shape), tuple(kit_feat.shape)),
              float(abs(our_feat.numel() - kit_feat.numel())))
        check("특징 값", float((our_feat - kit_feat).abs().max()), tol=1e-4)

        cs = torch.nn.functional.cosine_similarity(
            our_feat.reshape(-1, our_feat.shape[-1]),
            kit_feat.reshape(-1, kit_feat.shape[-1]), dim=-1)
        check("특징 코사인 유사도가 1 인가", float((1 - cs).abs().max()), tol=1e-5)

    print("")
    print("=" * 62)
    if FAIL:
        print("★ 실패 %d건: %s" % (len(FAIL), FAIL))
        print("  재구현이 채점기와 다르다. 이 상태로는 상관계수를 믿을 수 없다.")
        return 1
    print("전부 통과. common.py 는 채점기 전처리와 수치적으로 동일하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
