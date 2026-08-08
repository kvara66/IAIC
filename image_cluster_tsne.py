"""
Train 에피소드 첫 프레임 + Eval 이미지를 DINO로 임베딩 후 KMeans 클러스터링.
액션 클러스터링과 동일한 파이프라인으로 이미지 분포 비교.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── 설정 ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
TRAIN_ROOT = DATA_ROOT / "train"
EVAL_IMAGES = DATA_ROOT / "eval" / "images"
OUTPUT_DIR = Path(__file__).parent / "cluster_output"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_TRAIN = 5000   # 에피소드 서브샘플 (None이면 전부 11132개)
N_CLUSTERS = 2
PCA_COMPONENTS = 50
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

np.random.seed(RANDOM_SEED)
print(f"[Device] {DEVICE}")
# ───────────────────────────────────────────────────────────────────────────


def build_dino():
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,  # feature extractor
    ).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf


def extract_first_frame(mp4_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def embed_images(model, transform, image_list: list[np.ndarray]) -> np.ndarray:
    """numpy RGB 이미지 리스트 → (N, D) DINO 임베딩"""
    all_feats = []
    for i in range(0, len(image_list), BATCH_SIZE):
        batch = image_list[i:i + BATCH_SIZE]
        tensors = torch.stack([transform(Image.fromarray(img)) for img in batch]).to(DEVICE)
        with torch.no_grad():
            feats = model(tensors).cpu().numpy()
        all_feats.append(feats)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  임베딩 {i+len(batch)}/{len(image_list)} ...")
    return np.vstack(all_feats)


def load_train_frames():
    mp4_files = sorted(TRAIN_ROOT.rglob("*.mp4"))
    print(f"[Train] mp4 파일 수: {len(mp4_files)}")

    if MAX_TRAIN and len(mp4_files) > MAX_TRAIN:
        rng = np.random.default_rng(RANDOM_SEED)
        mp4_files = [mp4_files[i] for i in rng.choice(len(mp4_files), MAX_TRAIN, replace=False)]
        print(f"[Train] 서브샘플: {len(mp4_files)}")

    frames = []
    paths = []
    for p in mp4_files:
        img = extract_first_frame(p)
        if img is not None:
            frames.append(img)
            paths.append(p)
    print(f"[Train] 로드 완료: {len(frames)}개")
    return frames, paths


def load_eval_frames():
    img_files = sorted(EVAL_IMAGES.glob("*.png"))
    frames = []
    for f in img_files:
        img = np.array(Image.open(f).convert("RGB"))
        frames.append(img)
    print(f"[Eval]  로드 완료: {len(frames)}개")
    return frames


def main():
    # 1. 모델 로드
    print("[DINO] 모델 로드 ...")
    model, transform = build_dino()

    # 2. 이미지 로드
    train_frames, train_paths = load_train_frames()
    eval_frames = load_eval_frames()

    # 3. 임베딩
    print("[Train] DINO 임베딩 추출 ...")
    train_emb = embed_images(model, transform, train_frames)   # (N, 384)
    print("[Eval]  DINO 임베딩 추출 ...")
    eval_emb = embed_images(model, transform, eval_frames)     # (216, 384)
    print(f"[임베딩] train={train_emb.shape}, eval={eval_emb.shape}")

    # 4. 정규화 + PCA
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_emb)
    eval_scaled = scaler.transform(eval_emb)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    train_pca = pca.fit_transform(train_scaled)
    eval_pca = pca.transform(eval_scaled)
    print(f"[PCA]   설명 분산: {pca.explained_variance_ratio_.sum():.1%}")

    # 5. KMeans
    print(f"[KMeans] k={N_CLUSTERS} ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    train_labels = kmeans.fit_predict(train_pca)
    eval_labels = kmeans.predict(eval_pca)

    # Eval 클러스터 분포
    unique, counts = np.unique(eval_labels, return_counts=True)
    print("\n[Eval] 클러스터 분포:")
    for c, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  cluster {c:2d}: {cnt:3d}개 ({cnt/len(eval_labels)*100:.1f}%)")

    # Train 클러스터 분포
    print("\n[Train] 클러스터 분포:")
    for c in range(N_CLUSTERS):
        cnt = (train_labels == c).sum()
        print(f"  cluster {c:2d}: {cnt:4d}개 ({cnt/len(train_labels)*100:.1f}%)")

    # 6. 시각화 (PCA 2D)
    colors = cm.tab10(np.linspace(0, 1, N_CLUSTERS))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PCA dim1 vs dim2로 빠르게 시각화
    ax = axes[0]
    for c in range(N_CLUSTERS):
        mask = train_labels == c
        ax.scatter(train_pca[mask, 0], train_pca[mask, 1], s=5, alpha=0.3, color=colors[c], label=f"C{c}")
    ax.set_title("Train Image Embeddings — KMeans (PCA 2D)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[1]
    ax.scatter(train_pca[:, 0], train_pca[:, 1], s=3, alpha=0.15, color="lightgray", label="Train")
    for c in range(N_CLUSTERS):
        mask = eval_labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(eval_pca[mask, 0], eval_pca[mask, 1], s=60, alpha=0.9, color=colors[c],
                   edgecolors="black", linewidths=0.5, label=f"Eval C{c} ({mask.sum()})")
    ax.set_title("Eval Images on Train Cluster Space (PCA 2D)")
    ax.legend(markerscale=1.5, fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "image_cluster_pca.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[저장] {out_path}")
    plt.show()

    # 7. 통계 저장
    rows = []
    for c in range(N_CLUSTERS):
        train_cnt = int((train_labels == c).sum())
        eval_cnt = int((eval_labels == c).sum())
        rows.append({"cluster": c, "train_count": train_cnt, "eval_count": eval_cnt,
                     "eval_ratio": round(eval_cnt / len(eval_labels) * 100, 1)})
    df = pd.DataFrame(rows).sort_values("eval_count", ascending=False)
    df.to_csv(OUTPUT_DIR / "image_cluster_stats.csv", index=False)
    print(df.to_string(index=False))

    # 에피소드(mp4)별 클러스터 저장 — C0이 eval 없는 제거 대상
    # train_paths는 mp4 경로, 각 mp4 = 1 에피소드
    ep_df = pd.DataFrame({
        "mp4_path": [str(p) for p in train_paths],
        "cluster": train_labels,
    })
    ep_path = OUTPUT_DIR / "episode_image_cluster.csv"
    ep_df.to_csv(ep_path, index=False)

    c0_eps = (train_labels == 0).sum()
    c1_eps = (train_labels == 1).sum()
    print(f"\n[에피소드] C0(eval 없음, 제거): {c0_eps}개 / C1(eval 있음, 유지): {c1_eps}개")
    print(f"[저장] {ep_path}")

    # eval 없는 클러스터 contributor 분석
    zero_eval_clusters = [c for c in range(N_CLUSTERS) if (eval_labels == c).sum() == 0]
    for c in zero_eval_clusters:
        mask = train_labels == c
        cluster_paths = ep_df[mask]["mp4_path"].tolist()
        contributors = {}
        for p in cluster_paths:
            parts = Path(p).relative_to(TRAIN_ROOT).parts
            key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]
            contributors[key] = contributors.get(key, 0) + 1
        print(f"\n  Cluster {c} 제거 대상 top contributors:")
        for k, v in sorted(contributors.items(), key=lambda x: -x[1])[:5]:
            print(f"    {k}: {v}개")


if __name__ == "__main__":
    main()
