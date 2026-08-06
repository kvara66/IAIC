"""제출 CSV 에서 Action 성분을 실측한다. 평가 액션이 입력이라 자체 채점이 가능하다."""
import csv, json, statistics, sys, pathlib

csv.field_size_limit(10 ** 9)
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(f"  [실패] CSV 없음: {p}")
    sys.exit(1)
v = [json.loads(r["feature_json"])[0][0]
     for r in csv.DictReader(open(p, encoding="utf-8"))
     if r["feature_component"] == "Action Component"]
m = statistics.mean(v)
print(f"  CSV {p.stat().st_size:,} 바이트, n={len(v)}")
print(f"  >>> Action MAE = {m:.6f}   가중 = {0.4*m:.6f}")
print(f"      (2000스텝 0.408450 / 6000스텝 50%정지 0.404856 대비 {m-0.404856:+.6f})")
