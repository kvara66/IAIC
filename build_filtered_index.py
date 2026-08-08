"""
두 필터 합쳐서 filtered EpisodeIndex pkl 생성:
  1. 이미지 k=2: C1(체스 stereo) 제거
  2. 액션 k=2 t-SNE: C0(아래 클러스터) 제거
"""

import pickle
import pandas as pd
from pathlib import Path

TRAIN_ROOT   = Path(__file__).parent.parent / "data" / "train"
INDEX_IN     = Path(__file__).parent / "episode_index.pkl"
INDEX_OUT    = Path(__file__).parent / "episode_index_filtered.pkl"
OUTPUT_DIR   = Path(__file__).parent / "cluster_output"

# ── 필터 1: 이미지 클러스터 — C1이 제거 대상(체스)
img_df = pd.read_csv(OUTPUT_DIR / "episode_image_cluster.csv")
# mp4_path → cluster, C1=제거
remove_mp4 = set(img_df[img_df["cluster"] == 1]["mp4_path"].tolist())
print(f"[이미지 필터] 제거 mp4: {len(remove_mp4)}개")

# ── 필터 2: 액션 클러스터 — C0이 제거 대상
act_df = pd.read_csv(OUTPUT_DIR / "episode_action_cluster.csv")
# path(parquet) → episode_cluster, C0=제거
remove_parquet = set(act_df[act_df["episode_cluster"] == 0]["path"].tolist())
keep_parquet   = set(act_df[act_df["episode_cluster"] == 1]["path"].tolist())
print(f"[액션 필터] 제거 parquet: {len(remove_parquet)}개 / 유지: {len(keep_parquet)}개")

# ── episode_index.pkl 로드
with open(INDEX_IN, "rb") as f:
    entries = pickle.load(f)
print(f"\n[원본] {len(entries)}개 에피소드")

import json
info_cache: dict[str, dict] = {}

def get_parquet_path(dataset_root: Path, ep_idx: int) -> Path | None:
    key = str(dataset_root)
    if key not in info_cache:
        try:
            info_cache[key] = json.loads((dataset_root / "meta" / "info.json").read_text())
        except Exception:
            return None
    info = info_cache[key]
    chunks_size = info.get("chunks_size", 1000)
    chunk = ep_idx // chunks_size
    return dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)

def get_mp4_path(dataset_root: Path, ep_idx: int) -> Path | None:
    key = str(dataset_root)
    if key not in info_cache:
        return None
    info = info_cache[key]
    chunks_size = info.get("chunks_size", 1000)
    chunk = ep_idx // chunks_size
    VIDEO_KEY = "observation.images.image"
    return dataset_root / info["video_path"].format(
        episode_chunk=chunk, episode_index=ep_idx, video_key=VIDEO_KEY
    )

V3_DIR = Path(__file__).parent  # v3/ 폴더 (pkl이 저장된 기준 디렉토리)

kept = []
n_img = 0
n_act = 0
n_unknown = 0  # 액션 CSV에 없는 에피소드 → 유지

for i, (dataset_root, ep_idx, n_frames) in enumerate(entries):
    # 상대 경로 → 절대 경로 변환
    dataset_root = (V3_DIR / dataset_root).resolve()
    if i % 2000 == 0:
        print(f"  처리 중 {i}/{len(entries)} ...")

    # info.json 먼저 로드
    get_parquet_path(dataset_root, ep_idx)

    # 필터 1: 이미지 (mp4 경로 기준)
    mp4 = get_mp4_path(dataset_root, ep_idx)
    if mp4 and str(mp4) in remove_mp4:
        n_img += 1
        continue

    # 필터 2: 액션 (parquet 경로 기준)
    pq = get_parquet_path(dataset_root, ep_idx)
    if pq:
        pq_str = str(pq)
        if pq_str in remove_parquet:
            n_act += 1
            continue
        elif pq_str not in keep_parquet:
            n_unknown += 1  # 서브샘플에 없던 에피소드 → 유지

    kept.append((dataset_root, ep_idx, n_frames))

print(f"\n[결과]")
print(f"  이미지 필터 제거:  {n_img}개")
print(f"  액션 필터 제거:    {n_act}개")
print(f"  액션 CSV 미포함:   {n_unknown}개 (유지)")
print(f"  최종 유지:         {len(kept)}개 ({len(kept)/len(entries)*100:.1f}%)")

with open(INDEX_OUT, "wb") as f:
    pickle.dump(kept, f)
print(f"\n[저장] {INDEX_OUT}")
