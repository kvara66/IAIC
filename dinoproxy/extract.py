"""모델별로 300쌍의 코사인 유사도를 뽑아 CSV 로 저장한다.

각 모델은 자기 입력 크기와 자기 정규화 상수를 쓰되,
기하 처리(비율 유지 리사이즈 + 중앙 정렬 + 검은 패딩)는 채점기와 완전히 동일하다.
common.py 참고.

내장 검증 (매 실행마다 자동으로 돈다)
  1) identity 쌍의 코사인 거리가 0 인가          -> 아니면 전처리/추출이 틀렸다
  2) 특징에 NaN/Inf 가 있는가
  3) 두 번 돌려 같은 값이 나오는가 (--determinism)
  4) 각 모델의 입력 크기·정규화 상수·특징 차원·검은 여백 비율을 찍는다

사용:
  python extract.py --pairs /workspace/dinopairs --device cuda
  python extract.py --pairs /workspace/dinopairs --models dino,clip_l14 --determinism
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVAL_H, EVAL_W, content_fraction, cosine, extract_features,
                    model_input_size, normalize_stats)


def limit_threads(n: int) -> None:
    """torch CPU 스레드를 제한한다.

    ⚠ 실측(2026-08-10): 메모리가 작은 컨테이너에서 기본 48스레드로 두면
      480x640 한 장 전처리에 5.06초가 걸린다. 1스레드면 0.083초다. 61배 차이다.
      작은 텐서에 스레드를 많이 붙이면 경합 비용이 계산 비용을 압도한다.
      스레드 수는 결과값에 영향을 주지 않는다(bilinear interpolate 는 결정적).
    """
    import torch
    torch.set_num_threads(max(1, n))
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None


# 이름 -> (timm 이름, 대체 후보들)
MODELS: dict[str, list[str]] = {
    # 기준. 채점기가 쓰는 바로 그 모델 (feature_csv_utils.py 373행)
    "dino":       ["vit_small_patch14_dinov2.lvd142m"],
    "clip_l14":   ["vit_large_patch14_clip_224.openai",
                   "vit_large_patch14_clip_224.laion2b",
                   "vit_large_patch14_clip_336.openai"],
    "clip_b16":   ["vit_base_patch16_clip_224.openai",
                   "vit_base_patch16_clip_224.laion2b"],
    "siglip_b16": ["vit_base_patch16_siglip_224",
                   "vit_base_patch16_siglip_224.webli"],
    "mae_b16":    ["vit_base_patch16_224.mae"],
}


def build(name: str, device: torch.device):
    import timm
    last = None
    for cand in MODELS[name]:
        try:
            m = timm.create_model(cand, pretrained=True, num_classes=0)
            m.to(device).eval()
            return m, cand
        except Exception as e:
            last = e
            print("   시도 실패 %s: %s" % (cand, str(e)[:120]))
    # 이름이 바뀌었을 수 있으니 후보를 찾아 보여준다
    try:
        import timm
        key = name.split("_")[0]
        avail = [n for n in timm.list_models(pretrained=True) if key in n][:15]
        print("   timm 에서 '%s' 가 들어간 사용 가능 모델: %s" % (key, avail))
    except Exception:
        pass
    raise RuntimeError("%s 를 못 만들었다: %s" % (name, last))


def load_side(d: Path) -> tuple[torch.Tensor, list[str]]:
    """PNG 를 (N,3,320,512) uint8 텐서로. 파일명 순서 = pair_id 순서."""
    import imageio.v2 as imageio
    files = sorted(d.glob("*.png"))
    if not files:
        raise SystemExit("[실패] PNG 가 없다: %s" % d)
    arrs = []
    for f in files:
        a = imageio.imread(f)
        if a.shape[:2] != (EVAL_H, EVAL_W):
            raise SystemExit("[실패] %s 크기가 %s 다. 320x512 여야 한다." % (f.name, a.shape[:2]))
        arrs.append(a[..., :3])
    t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
    return t, [f.stem for f in files]


def run_model(name: str, gt: torch.Tensor, pred: torch.Tensor,
              device: torch.device, batch: int):
    model, timm_name = build(name, device)
    size = model_input_size(model, 0)          # 채점기와 동일하게 0 을 넘긴다
    mean, std = normalize_stats(model)
    frac = content_fraction(EVAL_H, EVAL_W, size)

    print("   timm 이름     %s" % timm_name)
    print("   입력 크기     %d x %d" % (size, size))
    print("   정규화 mean   %s" % np.round(mean.flatten().numpy(), 4).tolist())
    print("           std   %s" % np.round(std.flatten().numpy(), 4).tolist())
    print("   내용 면적     %.1f%%   (검은 여백 %.1f%%)" % (frac * 100, (1 - frac) * 100))

    # OOM 이면 배치를 반씩 줄여 다시 시도한다. 작은 GPU 나 CPU 에서도 돌게.
    b = batch
    while True:
        try:
            fg = extract_features(gt, model, device, size, mean, std, b)
            fp = extract_features(pred, model, device, size, mean, std, b)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or b <= 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            b = max(1, b // 2)
            print("   메모리 부족 -> 배치 %d 로 줄여 다시 시도" % b)
    if b != batch:
        print("   실제 배치     %d" % b)
    print("   특징 차원     %d" % fg.shape[1])

    bad = int((~torch.isfinite(fg)).sum() + (~torch.isfinite(fp)).sum())
    if bad:
        raise SystemExit("[실패] %s 특징에 NaN/Inf 가 %d개 있다" % (name, bad))

    sim = cosine(fg, fp)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sim, fg, fp, dict(timm_name=timm_name, size=size, dim=int(fg.shape[1]),
                             content=frac)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 채점기는 32 를 쓰지만 DINO 는 518x518(1370토큰)이라 작은 GPU 에서 넘칠 수 있다.
    # 배치 크기는 특징값에 영향을 주지 않는다(모델이 eval 모드, 샘플 간 독립).
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--determinism", action="store_true", help="같은 모델을 두 번 돌려 값이 같은지 확인")
    # 느린 CPU 에서 먼저 소수만 돌려 시간을 재보는 용도. 0 이면 전부.
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU 스레드. 2단계 리사이즈가 CPU 에서 돈다")
    args = ap.parse_args()
    limit_threads(args.threads)

    pairs = Path(args.pairs)
    device = torch.device(args.device)
    outdir = pairs / "feats"
    outdir.mkdir(exist_ok=True)

    gt, gt_ids = load_side(pairs / "gt")
    pred, pred_ids = load_side(pairs / "pred")
    if gt_ids != pred_ids:
        raise SystemExit("[실패] gt 와 pred 의 pair_id 가 다르다. 짝이 어긋났다.")
    if args.limit:
        gt, pred, gt_ids = gt[:args.limit], pred[:args.limit], gt_ids[:args.limit]
        print("⚠ --limit %d : 속도 측정용 부분 실행. 최종 결과로 쓰지 말 것." % args.limit)
    print("쌍 %d개, 장치 %s" % (len(gt_ids), device))

    strategy = {}
    mf = pairs / "manifest.csv"
    if mf.exists():
        with mf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                strategy[r["pair_id"]] = r["strategy"]
    ident_idx = [i for i, p in enumerate(gt_ids) if strategy.get(p) == "identity"]

    summary = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print("")
        print("=== %s ===" % name)
        _t0 = time.time()
        sim, fg, fp, info = run_model(name, gt, pred, device, args.batch)
        print("   소요          %.1f초  (%.2f초/장)" % (time.time()-_t0, (time.time()-_t0)/max(1,2*len(gt_ids))))

        # --- 검증 1: identity 쌍은 거리가 0 이어야 한다 ---
        if ident_idx:
            d = (1.0 - sim[ident_idx]).abs().max().item()
            ok = d < 1e-5
            print("   검증 identity  최대 거리 %.3e   %s" % (d, "통과" if ok else "★실패★"))
            if not ok:
                raise SystemExit("[실패] %s: 같은 그림인데 거리가 0 이 아니다. 전처리가 틀렸다." % name)
        else:
            print("   검증 identity  검산쌍이 없다 (mkpairs 에서 --n-identity 를 켤 것)")

        # --- 검증 3: 결정성 ---
        if args.determinism:
            sim2, _, _, _ = run_model(name, gt, pred, device, args.batch)
            d = (sim - sim2).abs().max().item()
            print("   검증 결정성    최대 차이 %.3e   %s" % (d, "통과" if d < 1e-6 else "★실패★"))
            if d >= 1e-6:
                raise SystemExit("[실패] %s: 두 번 돌린 값이 다르다." % name)

        path = outdir / ("%s.csv" % name)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "cos_sim", "cos_dist"])
            for pid, s in zip(gt_ids, sim.tolist()):
                w.writerow([pid, "%.8f" % s, "%.8f" % (1.0 - s)])
        np.save(outdir / ("%s_gt.npy" % name), fg.numpy())
        np.save(outdir / ("%s_pred.npy" % name), fp.numpy())

        s = sim.numpy()
        print("   유사도  평균 %.4f  최소 %.4f  최대 %.4f  표준편차 %.4f"
              % (s.mean(), s.min(), s.max(), s.std()))
        print("   저장 %s" % path.name)
        summary.append((name, info, float(s.mean()), float(s.std())))

    print("")
    print("=" * 74)
    print("%-12s %-34s %6s %6s %8s %8s" % ("이름", "timm", "크기", "차원", "평균유사", "표준편차"))
    print("-" * 74)
    for name, info, m, sd in summary:
        print("%-12s %-34s %6d %6d %8.4f %8.4f"
              % (name, info["timm_name"], info["size"], info["dim"], m, sd))
    print("")
    print("다음: python corr.py --pairs %s" % pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
