"""정답 300장 + 예측 300장 = 600장으로 300쌍을 만든다.
1단계 전처리(320x512)까지 끝낸 uint8 PNG 로 저장한다.

왜 1단계를 여기서 끝내나:
  1단계(영상 규격 맞추기)는 모델과 무관하다. 여기서 한 번만 해두면 양쪽이
  반드시 동일하게 처리되고, 모델 5개를 돌릴 때 매번 다시 안 해도 된다.
  extract.py 는 2단계(모델별 입력 크기)만 한다.


★ 쌍을 어떻게 뽑느냐 — 여기가 결과를 좌우한다

  그냥 무작위로 두 장을 짝지으면 분포가 봉우리 두 개가 된다.
  우연히 같은 에피소드에서 걸린 몇 쌍은 유사도 0.99, 나머지는 0.4 근처에 뭉친다.
  그러면 두 덩어리를 갈라놓기만 해도 상관계수가 0.95 가 나온다.
  실제 판단력과 무관하게 부풀려지고, 모델 순위가 의미를 잃는다.

  반대로 "다른 데이터셋끼리만" 뽑으면 전부 0.3~0.6 좁은 구간에 몰린다.
  범위가 좁으면 상관계수가 표본 잡음에 흔들린다(range restriction).

  그래서 유사도가 고르게 퍼지도록 세 층으로 나눠 뽑는다.

    near   같은 에피소드의 다른 프레임          유사도 높음
    mid    같은 데이터셋의 다른 에피소드        중간
    far    다른 데이터셋                        낮음

  층 정보는 manifest 에 남겨서 corr.py 가 구간별로도 상관계수를 낸다.
  세 구간 모두에서 이기는 모델이면 확실한 답이고,
  구간마다 순위가 뒤집히면 그것도 알아야 할 사실이다.

  --strategy random 을 주면 층 없이 완전 무작위로도 뽑는다(원래 계획 그대로).


검산쌍 (identity)
  같은 프레임을 양쪽에 넣은 쌍을 섞는다. 모든 모델에서 코사인 거리가 0 이어야 한다.
  안 나오면 전처리나 특징 추출이 조용히 틀린 것이다. 이게 없으면 틀린 결과를 믿게 된다.


데이터셋/에피소드 판정
  train_root/<데이터셋이름>/**/<...>.mp4  구조를 가정한다.
  데이터셋 = train_root 바로 아래 폴더 이름, 에피소드 = mp4 파일 하나.
  실제 구조가 다르면 --dataset-depth 로 조정한다.

사용:
  python mkpairs.py --out /workspace/dinopairs --train-root /workspace/open/data/train
  python mkpairs.py --out /workspace/dinopairs --strategy random     # 완전 무작위
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_H, EVAL_W, stage1_to_uint8


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


_NFRAMES: dict[str, int] = {}


def frame_count(path: Path) -> int:
    """프레임 수를 싸게 알아낸다.

    ⚠ 전부 디코드해서 세면 영상 하나당 수 초가 걸린다. 620번 하면 몇십 분이다.
      메타데이터(stream.frames) -> 길이x프레임율 -> 그래도 없으면 패킷 수 순으로 본다.
      CPU 로 돌릴 때 이 차이가 크다.
    """
    k = str(path)
    if k in _NFRAMES:
        return _NFRAMES[k]
    n = 0
    try:
        with av.open(k) as c:
            s = c.streams.video[0]
            if s.frames:
                n = int(s.frames)
            elif s.duration is not None and s.time_base and s.average_rate:
                n = int(float(s.duration * s.time_base) * float(s.average_rate))
            elif c.duration and s.average_rate:
                n = int((c.duration / 1_000_000) * float(s.average_rate))
            else:
                n = sum(1 for _ in c.demux(s) if not _.is_corrupt)
    except Exception:
        n = 0
    _NFRAMES[k] = max(0, n)
    return _NFRAMES[k]


def read_frames(path: Path, idxs: list[int]) -> dict[int, np.ndarray]:
    want, out, i = set(idxs), {}, 0
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
            i += 1
    return out


def one_frame(path: Path, rng: random.Random) -> np.ndarray | None:
    try:
        n = frame_count(path)
        if n < 1:
            return None
        t = rng.randrange(0, n)
        return read_frames(path, [t]).get(t)
    except Exception:
        return None


def many_frames(path: Path, k: int, rng: random.Random) -> list[np.ndarray]:
    """한 번 열어서 서로 다른 프레임 k 장을 가져온다. 파일 열기 비용을 k 로 나눈다."""
    try:
        n = frame_count(path)
        if n < 1:
            return []
        idxs = sorted(rng.sample(range(n), min(k, n)))
        got = read_frames(path, idxs)
        return [got[i] for i in idxs if i in got]
    except Exception:
        return []


def canon(frame_hwc: np.ndarray) -> np.ndarray:
    """1단계를 태워 320x512 uint8 로. (H,W,3) -> (320,512,3)"""
    return stage1_to_uint8(frame_hwc[None], EVAL_H, EVAL_W, pad=True)[0].numpy()


def save_png(arr: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)


def write_pkl(out: Path, rows: list[dict]) -> None:
    """PNG 를 모아 pairs.pkl 로 저장한다. 620쌍이면 약 610MB 를 쓴다."""
    import pickle
    import imageio.v2 as _iio
    gt = np.stack([_iio.imread(out / "gt" / ("%s.png" % r["pair_id"]))[..., :3] for r in rows])
    pr = np.stack([_iio.imread(out / "pred" / ("%s.png" % r["pair_id"]))[..., :3] for r in rows])
    with (out / "pairs.pkl").open("wb") as f:
        pickle.dump({"gt": gt, "pred": pr, "manifest": rows,
                     "note": "1단계(320x512 비율유지+검은패딩) 까지 끝낸 uint8. "
                             "extract.py 가 2단계(모델별 입력크기)만 한다."}, f)
    print("pkl 저장  gt %s  pred %s  ->  %s" % (gt.shape, pr.shape, out / "pairs.pkl"))


def group_by_dataset(mp4s: list[Path], root: Path, depth: int) -> dict[str, list[Path]]:
    g: dict[str, list[Path]] = defaultdict(list)
    for p in mp4s:
        try:
            rel = p.relative_to(root).parts
        except ValueError:
            rel = p.parts
        key = "/".join(rel[:depth]) if len(rel) >= depth else rel[0]
        g[key].append(p)
    return dict(g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-root", default="/workspace/open/data/train")
    # 기본은 완전 무작위 (팀 결정, 2026-08-09).
    # stratified 는 유사도 구간별로 순위가 뒤집히는지 보고 싶을 때만 쓴다.
    ap.add_argument("--strategy", default="random", choices=["random", "stratified"])
    ap.add_argument("--n-near", type=int, default=100)
    ap.add_argument("--n-mid", type=int, default=100)
    ap.add_argument("--n-far", type=int, default=100)
    ap.add_argument("--n", type=int, default=300, help="--strategy random 일 때 쌍 개수")
    ap.add_argument("--n-identity", type=int, default=20, help="검산용 동일쌍 (0 이면 끔)")
    # 1 이면 영상 하나당 한 장(가장 무작위). CPU 로 돌릴 때 5~10 으로 올리면
    # 파일 여는 비용이 그만큼 줄어 훨씬 빠르다. 뽑은 뒤 전체를 섞으므로 짝은 여전히 무작위다.
    ap.add_argument("--frames-per-video", type=int, default=1)
    ap.add_argument("--pkl", action="store_true", help="pairs.pkl 도 만든다 (메모리 약 610MB)")
    ap.add_argument("--pkl-only", action="store_true", help="이미 만든 PNG 로 pkl 만 만든다")
    ap.add_argument("--dataset-depth", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU 스레드. 많으면 오히려 느리다")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    limit_threads(args.threads)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root = Path(args.train_root)
    out = Path(args.out)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)

    if args.pkl_only:
        import csv as _csv
        with (out / "manifest.csv").open(encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        write_pkl(out, rows)
        return 0

    mp4s = sorted(root.rglob("*.mp4"))
    print("train mp4 %d개 발견: %s" % (len(mp4s), root))
    if not mp4s:
        print("[실패] train 영상을 못 찾았다")
        return 1

    groups = group_by_dataset(mp4s, root, args.dataset_depth)
    multi = [k for k, v in groups.items() if len(v) >= 2]
    print("데이터셋 %d개 (에피소드 2개 이상인 것 %d개)" % (len(groups), len(multi)))
    for k in list(groups)[:3]:
        print("   예: %-40s 에피소드 %d개" % (k, len(groups[k])))
    if len(groups) < 2:
        print("   ⚠ 데이터셋이 하나로 잡혔다. --dataset-depth 를 조정할 것.")

    rows: list[dict] = []

    def add(gt: np.ndarray, pred: np.ndarray, strategy: str, detail: str) -> None:
        i = len(rows)
        save_png(canon(gt), out / "gt" / ("%04d.png" % i))
        save_png(canon(pred), out / "pred" / ("%04d.png" % i))
        rows.append({"pair_id": "%04d" % i, "strategy": strategy, "detail": detail})

    if args.strategy == "stratified":
        # ---- near : 같은 에피소드의 다른 프레임 ----
        made = tries = 0
        while made < args.n_near and tries < args.n_near * 30:
            tries += 1
            v = rng.choice(mp4s)
            try:
                n = frame_count(v)
                if n < 3:
                    continue
                a, b = rng.sample(range(n), 2)
                fr = read_frames(v, [a, b])
                if a in fr and b in fr:
                    add(fr[a], fr[b], "near", "%s_f%d_f%d" % (v.stem, a, b))
                    made += 1
            except Exception:
                continue
        print("[near ] %d쌍  같은 에피소드" % made)

        # ---- mid : 같은 데이터셋의 다른 에피소드 ----
        made = tries = 0
        while made < args.n_mid and tries < args.n_mid * 30 and multi:
            tries += 1
            key = rng.choice(multi)
            va, vb = rng.sample(groups[key], 2)
            fa, fb = one_frame(va, rng), one_frame(vb, rng)
            if fa is not None and fb is not None:
                add(fa, fb, "mid", key[:40])
                made += 1
        print("[mid  ] %d쌍  같은 데이터셋 다른 에피소드" % made)

        # ---- far : 다른 데이터셋 ----
        made = tries = 0
        keys = list(groups)
        while made < args.n_far and tries < args.n_far * 30 and len(keys) >= 2:
            tries += 1
            ka, kb = rng.sample(keys, 2)
            fa = one_frame(rng.choice(groups[ka]), rng)
            fb = one_frame(rng.choice(groups[kb]), rng)
            if fa is not None and fb is not None:
                add(fa, fb, "far", "%s|%s" % (ka[:20], kb[:20]))
                made += 1
        print("[far  ] %d쌍  다른 데이터셋" % made)
    else:
        # ---- 완전 무작위 : 정답 300장 + 예측 300장 ----
        #
        # ⚠ 600장을 다 모았다가 짝짓지 않는다. 480x640 x 600장 = 553MB 라
        #   메모리가 작은 환경에서 죽는다. 두 장씩 뽑아 즉시 저장한다.
        #   전부 무작위이므로 "600장 뽑아 앞300/뒤300 으로 짝" 과 통계적으로 동일하다.
        #   (같은 영상의 같은 프레임이 양쪽에 걸리면 다시 뽑는다)
        made, tries, t0 = 0, 0, time.time()
        while made < args.n and tries < args.n * 60:
            tries += 1
            va, vb = rng.choice(mp4s), rng.choice(mp4s)
            ta, tb = frame_count(va), frame_count(vb)
            if ta < 1 or tb < 1:
                continue
            ia, ib = rng.randrange(ta), rng.randrange(tb)
            if va == vb and ia == ib:
                continue                      # 우연한 동일쌍은 검산쌍과 헷갈리니 제외
            fa = read_frames(va, [ia]).get(ia)
            fb = read_frames(vb, [ib]).get(ib)
            if fa is None or fb is None:
                continue
            add(fa, fb, "random", "%s#%d|%s#%d" % (va.stem, ia, vb.stem, ib))
            made += 1
            del fa, fb
            if made % 25 == 0:
                el = time.time() - t0
                rate = made / max(el, 1e-6)
                print("   %d/%d쌍  %.1f쌍/초  남은시간 약 %.0f초"
                      % (made, args.n, rate, (args.n - made) / max(rate, 1e-6)))
        print("[random] %d쌍  (%.0f초)" % (made, time.time() - t0))

    # ---- identity : 검산 ----
    made = tries = 0
    while made < args.n_identity and tries < args.n_identity * 30:
        tries += 1
        f = one_frame(rng.choice(mp4s), rng)
        if f is not None:
            add(f, f, "identity", "same")
            made += 1
    if made:
        print("[ident] %d쌍  <- 모든 모델에서 코사인 거리 0 이어야 한다" % made)

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "strategy", "detail"])
        w.writeheader()
        w.writerows(rows)

    # pkl 로도 저장한다. 팀 원안이 "pkl 파일" 이라 형식을 맞춘다. PNG 와 내용은 같다.
    # ⚠ 620쌍을 한꺼번에 배열로 올리면 약 610MB 를 쓴다. 메모리가 작은 환경에서는
    #   --pkl 없이 PNG 만 만들고, 나중에 여유 있는 곳에서 --pkl-only 로 만든다.
    if args.pkl:
        write_pkl(out, rows)

    print("")
    print("총 %d쌍  ->  %s" % (len(rows), out))
    for k, v in Counter(r["strategy"] for r in rows).most_common():
        print("   %-10s %d" % (k, v))
    print("")
    print("⚠ 이 폴더를 그대로 보관할 것. 모든 모델이 반드시 같은 쌍을 봐야 한다.")
    print("다음: python extract.py --pairs %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
