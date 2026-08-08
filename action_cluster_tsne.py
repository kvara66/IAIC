"""
Train action 시퀀스를 sliding window로 자르고 t-SNE 클러스터링.
Eval 데이터를 같은 공간에 투영해서 분포 비교.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# ── 설정 ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
TRAIN_ROOT = DATA_ROOT / "train"
EVAL_ACTIONS = DATA_ROOT / "eval" / "actions"
OUTPUT_DIR = Path(__file__).parent / "cluster_output"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 16
STRIDE = 8            # 슬라이딩 간격 (작을수록 더 많은 윈도우)
MAX_WINDOWS = 50000   # t-SNE 메모리/속도 한계, None이면 전부
N_CLUSTERS = 2
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 40
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ───────────────────────────────────────────────────────────────────────────


def load_train_windows():
    """Train parquet에서 (16,6) sliding window 추출 → (N, 96), 경로도 추적"""
    parquet_files = sorted(TRAIN_ROOT.rglob("*.parquet"))
    print(f"[Train] parquet 파일 수: {len(parquet_files)}")

    windows = []
    window_paths = []  # 각 윈도우가 어느 parquet에서 왔는지 추적

    for pq_path in parquet_files:
        try:
            df = pd.read_parquet(pq_path, columns=["observation.state"])
        except Exception:
            continue

        try:
            actions = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 6)
        except Exception:
            continue

        T = len(actions)
        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            windows.append(actions[start:start + WINDOW_SIZE].flatten())
            window_paths.append(str(pq_path))

    windows = np.array(windows, dtype=np.float32)
    print(f"[Train] 총 윈도우 수: {len(windows)}")
    return windows, window_paths


def load_eval_windows():
    """Eval npy (216, 16, 6) → (216, 96)"""
    npy_files = sorted(EVAL_ACTIONS.glob("*.npy"))
    data = []
    for f in npy_files:
        arr = np.load(f).astype(np.float32)  # (16, 6)
        data.append(arr.flatten())
    data = np.array(data, dtype=np.float32)  # (216, 96)
    print(f"[Eval]  샘플 수: {len(data)}")
    return data


def main():
    # 1. 데이터 로드
    train_windows, window_paths = load_train_windows()
    eval_windows = load_eval_windows()

    # 2. 서브샘플 (t-SNE 속도) — 경로도 같이 서브샘플
    if MAX_WINDOWS and len(train_windows) > MAX_WINDOWS:
        idx = np.random.choice(len(train_windows), MAX_WINDOWS, replace=False)
        train_sub = train_windows[idx]
        sub_paths = [window_paths[i] for i in idx]
        print(f"[Train] 서브샘플: {len(train_sub)}")
    else:
        train_sub = train_windows
        sub_paths = window_paths

    # 3. 정규화 (train 기준 fit)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    eval_scaled = scaler.transform(eval_windows)

    # 4. PCA
    print(f"[PCA]   {train_scaled.shape[1]}D → {PCA_COMPONENTS}D ...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    train_pca = pca.fit_transform(train_scaled)
    eval_pca = pca.transform(eval_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[PCA]   설명 분산: {explained:.1%}")

    # 5. t-SNE 먼저 (train + eval 합쳐서)
    combined = np.vstack([train_pca, eval_pca])
    print(f"\n[t-SNE] {len(combined)}개 포인트 ...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=1000,
        random_state=RANDOM_SEED,
        verbose=1,
    )
    combined_2d = tsne.fit_transform(combined)

    train_2d = combined_2d[: len(train_pca)]
    eval_2d = combined_2d[len(train_pca):]

    # 6. KMeans를 t-SNE 2D 공간에서 실행 → 경계가 시각적 구조와 일치
    print(f"[KMeans] k={N_CLUSTERS} (t-SNE 공간) ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    train_labels = kmeans.fit_predict(train_2d)
    eval_labels = kmeans.predict(eval_2d)

    # Eval 클러스터 분포 출력
    unique, counts = np.unique(eval_labels, return_counts=True)
    print("\n[Eval] 클러스터 분포:")
    for c, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  cluster {c:2d}: {cnt:3d}개 ({cnt/len(eval_labels)*100:.1f}%)")

    # 7. 시각화
    colors = cm.tab10(np.linspace(0, 1, N_CLUSTERS))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # (A) Train 클러스터
    ax = axes[0]
    for c in range(N_CLUSTERS):
        mask = train_labels == c
        ax.scatter(
            train_2d[mask, 0], train_2d[mask, 1],
            s=3, alpha=0.4, color=colors[c], label=f"C{c}"
        )
    ax.set_title("Train Action Windows — K-Means Clusters")
    ax.legend(markerscale=4, loc="upper right", fontsize=8)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # (B) Train 배경 + Eval 오버레이
    ax = axes[1]
    ax.scatter(train_2d[:, 0], train_2d[:, 1], s=2, alpha=0.2, color="lightgray", label="Train")
    for c in range(N_CLUSTERS):
        mask = eval_labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(
            eval_2d[mask, 0], eval_2d[mask, 1],
            s=40, alpha=0.9, color=colors[c],
            edgecolors="black", linewidths=0.5,
            label=f"Eval C{c} ({mask.sum()})"
        )
    ax.set_title("Eval Samples on Train Cluster Space")
    ax.legend(markerscale=1.5, loc="upper right", fontsize=8)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "action_cluster_tsne.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[저장] {out_path}")
    plt.show()

    # 8. 클러스터별 액션 통계 저장
    stats_rows = []
    for c in range(N_CLUSTERS):
        mask = train_labels == c
        seqs = train_sub[mask].reshape(-1, WINDOW_SIZE, 6)  # (M, 16, 6)
        mean_action = seqs.mean(axis=(0, 1))
        std_action = seqs.std(axis=(0, 1))
        eval_count = (eval_labels == c).sum()
        stats_rows.append({
            "cluster": c,
            "train_count": int(mask.sum()),
            "eval_count": int(eval_count),
            **{f"mean_{i}": float(mean_action[i]) for i in range(6)},
            **{f"std_{i}": float(std_action[i]) for i in range(6)},
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = OUTPUT_DIR / "cluster_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[저장] {stats_path}")
    print(stats_df[["cluster", "train_count", "eval_count"]].to_string(index=False))

    # 9. 에피소드별 클러스터 저장 (다수결: 윈도우 과반이 C1이면 에피소드도 C1)
    path_cluster_df = pd.DataFrame({"path": sub_paths, "cluster": train_labels})
    # 에피소드(parquet 파일) 별로 C1 비율 계산
    ep_stats = path_cluster_df.groupby("path")["cluster"].agg(
        total="count",
        c1_count=lambda x: (x == 1).sum()
    ).reset_index()
    ep_stats["c1_ratio"] = ep_stats["c1_count"] / ep_stats["total"]
    # C1 비율 > 0.5이면 C1 에피소드
    ep_stats["episode_cluster"] = (ep_stats["c1_ratio"] > 0.5).astype(int)

    ep_path = OUTPUT_DIR / "episode_action_cluster.csv"
    ep_stats.to_csv(ep_path, index=False)
    c1_eps = (ep_stats["episode_cluster"] == 1).sum()
    print(f"\n[에피소드] C1(유지): {c1_eps}개 / C0(제거): {len(ep_stats)-c1_eps}개")
    print(f"[저장] {ep_path}")


if __name__ == "__main__":
    main()
