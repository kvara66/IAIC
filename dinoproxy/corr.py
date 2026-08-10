"""모델별 코사인 거리 CSV 를 DINO 것과 대조해 상관관계를 낸다.

무엇을 보나:
  DINO 가 매긴 300개의 코사인 거리와, 후보 모델이 매긴 300개를 비교한다.
  상관계수가 높을수록 "DINO 와 같은 기준으로 닮음을 판단한다"는 뜻이다.

  Pearson   값 자체가 함께 움직이는 정도.   로스로 쓸 거면 이쪽이 중요하다.
  Spearman  순위만 함께 움직이는 정도.      단조 관계면 1 에 가깝다.
  둘이 크게 다르면 관계가 비선형이라는 뜻이다.

검증 (자동)
  1) dino 대 dino 의 상관계수가 정확히 1.0 인가  -> 아니면 파이프라인이 틀렸다
  2) identity 쌍의 거리가 모든 모델에서 0 인가
  3) 유사도 분포가 한쪽에 몰려 있지 않은가 (범위가 좁으면 상관계수가 불안정하다)
  4) 봉우리가 둘인가 (그러면 상관계수가 부풀려진다)

scipy 없이 numpy 만으로 계산한다. Spearman 은 순위에 Pearson 을 적용한 것이다.

사용:
  python corr.py --pairs /workspace/dinopairs
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def rankdata(a: np.ndarray) -> np.ndarray:
    """동점은 평균 순위를 준다 (scipy.stats.rankdata 의 'average' 와 동일)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def load(path: Path) -> dict[str, float]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["pair_id"]] = float(r["cos_dist"])
    return out


def bimodality(x: np.ndarray) -> float:
    """Sarle 의 bimodality coefficient. 0.555 를 넘으면 봉우리가 둘일 가능성이 크다."""
    n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return float("nan")
    z = (x - m) / s
    g1 = (z ** 3).mean()
    g2 = (z ** 4).mean() - 3.0
    return float((g1 ** 2 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def hist_line(x: np.ndarray, bins: int = 20, width: int = 46) -> list[str]:
    h, edges = np.histogram(x, bins=bins)
    top = max(1, h.max())
    return ["   %5.3f~%5.3f | %-*s %d" % (edges[i], edges[i + 1], width,
                                          "#" * int(round(h[i] / top * width)), h[i])
            for i in range(bins)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base", default="dino")
    # 검산쌍은 기본으로 뺀다. 거리 0 이 한 점에 몰려 상관계수를 부풀리기 때문이다.
    ap.add_argument("--keep-identity", action="store_true",
                    help="검산쌍도 상관계수 계산에 포함 (권장하지 않음)")
    args = ap.parse_args()
    args.exclude_identity = not args.keep_identity

    pairs = Path(args.pairs)
    feats = pairs / "feats"
    csvs = sorted(feats.glob("*.csv"))
    if not csvs:
        raise SystemExit("[실패] %s 에 CSV 가 없다. extract.py 를 먼저 돌릴 것." % feats)

    data = {p.stem: load(p) for p in csvs}
    if args.base not in data:
        raise SystemExit("[실패] 기준 '%s' 의 CSV 가 없다. 있는 것: %s"
                         % (args.base, list(data)))

    strategy = {}
    mf = pairs / "manifest.csv"
    if mf.exists():
        with mf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                strategy[r["pair_id"]] = r["strategy"]

    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    ident = [p for p in ids if strategy.get(p) == "identity"]
    use = [p for p in ids if not (args.exclude_identity and strategy.get(p) == "identity")]
    print("쌍 %d개 (검산쌍 %d개 제외 -> %d개로 상관계수 계산)"
          % (len(ids), len(ident), len(use)))

    base = np.array([data[args.base][p] for p in use])

    # ---------------- 검증 ----------------
    print("")
    print("=== 검증 ===")
    r_self = pearson(base, base)
    print("  1) %s 대 %s Pearson = %.10f   %s"
          % (args.base, args.base, r_self, "통과" if abs(r_self - 1) < 1e-9 else "★실패★"))
    if abs(r_self - 1) >= 1e-9:
        raise SystemExit("[실패] 자기 자신과의 상관계수가 1 이 아니다. 계산이 틀렸다.")

    if ident:
        bad = []
        for name, d in data.items():
            mx = max(abs(d[p]) for p in ident)
            if mx > 1e-5:
                bad.append((name, mx))
        if bad:
            print("  2) identity 쌍  ★실패★  " +
                  ", ".join("%s=%.2e" % (n, v) for n, v in bad))
            raise SystemExit("[실패] 같은 그림인데 거리가 0 이 아닌 모델이 있다.")
        print("  2) identity 쌍  모든 모델에서 거리 < 1e-5   통과")
    else:
        print("  2) identity 쌍  없음 (mkpairs 의 --n-identity 를 켜면 검산된다)")

    rng_ = base.max() - base.min()
    print("  3) %s 거리 범위 %.4f ~ %.4f (폭 %.4f)   %s"
          % (args.base, base.min(), base.max(), rng_,
             "통과" if rng_ > 0.15 else "⚠ 범위가 좁다. 상관계수가 불안정할 수 있다"))

    bc = bimodality(base)
    print("  4) 봉우리 계수 %.3f   %s"
          % (bc, "통과 (단봉)" if bc < 0.555 else
             "⚠ 봉우리가 둘일 수 있다. 상관계수가 부풀려질 수 있으니 아래 히스토그램 확인"))

    print("")
    print("=== %s 코사인 거리 분포 ===" % args.base)
    for line in hist_line(base):
        print(line)

    # ---------------- 상관계수 ----------------
    cands = [n for n in sorted(data) if n != args.base]
    print("")
    print("=== DINO 와의 상관관계 (전체 %d쌍) ===" % len(use))
    print("%-12s %10s %10s %12s" % ("모델", "Pearson", "Spearman", "평균거리"))
    print("-" * 48)
    res = []
    for name in cands:
        v = np.array([data[name][p] for p in use])
        pr, sp = pearson(base, v), spearman(base, v)
        res.append((name, pr, sp, float(v.mean())))
    for name, pr, sp, mu in sorted(res, key=lambda t: -t[1]):
        print("%-12s %10.4f %10.4f %12.4f" % (name, pr, sp, mu))

    # ---------------- 전략별 ----------------
    kinds = sorted({strategy.get(p, "?") for p in use})
    if len(kinds) > 1:
        print("")
        print("=== 쌍 종류별 Pearson ===")
        print("%-12s %s" % ("모델", "".join("%12s" % k for k in kinds)))
        print("-" * (12 + 12 * len(kinds)))
        for name, pr, _, _ in sorted(res, key=lambda t: -t[1]):
            cells = []
            for k in kinds:
                sub = [p for p in use if strategy.get(p, "?") == k]
                if len(sub) < 8:
                    cells.append("%12s" % "-")
                    continue
                b = np.array([data[args.base][p] for p in sub])
                v = np.array([data[name][p] for p in sub])
                cells.append("%12.4f" % pearson(b, v))
            print("%-12s %s" % (name, "".join(cells)))

    # ---------------- 구간별 ----------------
    print("")
    print("=== DINO 거리 사분위 구간별 Pearson (구간 안에서도 순위를 지키나) ===")
    q = np.quantile(base, [0.25, 0.5, 0.75])
    bands = [("가장 닮음 Q1", base <= q[0]),
             ("Q2", (base > q[0]) & (base <= q[1])),
             ("Q3", (base > q[1]) & (base <= q[2])),
             ("가장 다름 Q4", base > q[2])]
    print("%-12s %s" % ("모델", "".join("%14s" % b[0] for b in bands)))
    print("-" * (12 + 14 * len(bands)))
    for name, pr, _, _ in sorted(res, key=lambda t: -t[1]):
        v = np.array([data[name][p] for p in use])
        cells = []
        for _, m in bands:
            cells.append("%14.4f" % pearson(base[m], v[m]) if m.sum() >= 8 else "%14s" % "-")
        print("%-12s %s" % (name, "".join(cells)))

    best = max(res, key=lambda t: t[1])
    print("")
    print("=" * 60)
    print("전체 Pearson 기준 1위: %s  (r = %.4f, Spearman %.4f)"
          % (best[0], best[1], best[2]))
    print("")
    print("읽는 법")
    print("  - 전체 상관계수가 높아도 구간별로 뒤집히면 그 모델은 특정 구간에서만 맞는 것이다.")
    print("  - Pearson 과 Spearman 차이가 크면 관계가 비선형이다. 로스로 쓸 때 문제가 된다.")
    print("  - 봉우리 계수 경고가 떴으면 전체 상관계수를 그대로 믿지 말고 구간별을 볼 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
