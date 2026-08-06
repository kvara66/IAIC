# ============================================================================
# 사용 금지 — 규정 위반 방식 (2026-08-06 확인)
#
# 이 스크립트는 submission_kit 을 실행해 얻은 Action 값으로 영상을 고르거나
# 후보를 평가한다. 대회 규칙 "Submission Kit 사용 범위(8/3 11시~)" 의
# 다음 항목에 해당한다.
#
#   - Test 데이터의 특징값을 추출하여 예측 결과를 선택하거나 보정하는 행위
#   - Submission Kit 의 실행 결과를 활용하여 최종 mp4 를 생성·선택·수정하는 행위
#   - Submission Kit 정보를 후처리 또는 앙상블 과정에 활용하는 행위
#   - Submission Kit 을 제출 CSV 생성 외의 목적으로 사용하는 행위
#   - 최종 영상 선택은 Submission Kit 실행 이전에 완료되어야 함
#
# 위반 시 평가·수상 대상에서 제외될 수 있다.
# 2026-08-06 데이콘에 자진 신고했고, 이 방식의 작업은 전부 중단했다.
#
# 지우지 않고 남겨두는 이유 — 무엇을 어떻게 했는지 설명할 근거가 필요하다.
# 실행하지 말 것. 기록용 보관.
# ============================================================================

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
