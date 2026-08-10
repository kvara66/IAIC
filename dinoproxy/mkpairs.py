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
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_H, EVAL_W, stage1_to_uint8


def frame_count(path: Path) -> int:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return int(s.frames) if s.frames else sum(1 for _ in c.decode(s))


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


def canon(frame_hwc: np.ndarray) -> np.ndarray:
    """1단계를 태워 320x512 uint8 로. (H,W,3) -> (320,512,3)"""
    return stage1_to_uint8(frame_hwc[None], EVAL_H, EVAL_W, pad=True)[0].numpy()


def save_png(arr: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)


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
    ap.add_argument("--dataset-depth", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root = Path(args.train_root)
    out = Path(args.out)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "pred").mkdir(parents=True, exist_ok=True)

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
        # ---- 완전 무작위 : 600장 뽑아 앞 300 / 뒤 300 으로 짝 ----
        got, tries = [], 0
        while len(got) < args.n * 2 and tries < args.n * 40:
            tries += 1
            f = one_frame(rng.choice(mp4s), rng)
            if f is not None:
                got.append(f)
        half = len(got) // 2
        for i in range(half):
            add(got[i], got[half + i], "random", "idx%d" % i)
        print("[random] %d쌍" % half)

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
