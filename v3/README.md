# Action Extractor 학습 (extractor_v3)

## 필요 파일 (이 브랜치에 포함됨)
- `train_extractor.py` — 학습 스크립트 (에피소드 단위 train/val 분리 + 320×512 채점기 리사이즈 전처리 + 100 epoch)
- `dataset.py` — `EpisodeIndex`, `LeRobotSequenceDataset`
- `action_extractor_indep.py` — 모델 구조 (`ActionExtractorIndep`)
- `episode_index_filtered.pkl` — 필터링된 에피소드 인덱스

## 필요하지만 이 브랜치엔 없는 것 (기존 데이터 재사용)
- `../data/train/` — 실제 학습 데이터(parquet + mp4). extractor_v2 학습 때 쓰던 것과 동일한 경로/데이터.
- `../data/train/so100_action_statistics.json` — action 정규화 stats

이 파일들이 있는 위치에 맞게 아래 실행 시 `--index`, `--action-stats` 경로를 조정하세요.
(기본값은 `episode_index_filtered.pkl`이 현재 디렉토리에, action stats가 `../data/train/so100_action_statistics.json`에 있다고 가정합니다.)

## 필요 패키지
```
torch, torchvision, av, pandas, numpy
```

## 실행
```bash
cd v3   # 이 파일들이 있는 디렉토리
python train_extractor.py --output-dir runs/extractor_v3 --amp
```

## 완료 후
`runs/extractor_v3/best.pt`를 본 프로젝트(v3_real 브랜치 or S3) 쪽에 전달해주세요.
