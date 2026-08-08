"""
ActionExtractorIndep 학습 스크립트.
LeRobotSequenceDataset 재사용, episode_index_filtered.pkl 기준.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import EpisodeIndex, LeRobotSequenceDataset
from action_extractor_indep import ActionExtractorIndep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index",        default="episode_index_filtered.pkl")
    p.add_argument("--action-stats", default="../data/train/so100_action_statistics.json")
    p.add_argument("--output-dir",   default="runs/extractor_v1")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--val-ratio",    type=float, default=0.05)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--amp",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    index = EpisodeIndex.load(args.index)
    dataset = LeRobotSequenceDataset(
        index,
        action_stats_path=args.action_stats,
        samples_per_epoch=args.samples_per_epoch,
    )

    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── 모델 ─────────────────────────────────────────────────────────────
    # Blackwell GPU(sm_120)에서 flatten_parameters()가 cuDNN 커널 없어 뻗음
    with torch.backends.cudnn.flags(enabled=False):
        model = ActionExtractorIndep().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] {total/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_mae = float("inf")

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_mae = 0.0
        t0 = time.time()

        for batch in train_loader:
            frames = batch["frames"].to(device)   # (B, 16, 3, H, W)
            states = batch["states"].to(device)   # (B, 16, 6) normalized

            # extractor 입력: (B, 3, T, H, W)
            video = frames.permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(video)               # (B, 16, 6)
                loss = F.mse_loss(pred, states)
                mae  = F.l1_loss(pred, states)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_mae += mae.item()

        train_mae /= len(train_loader)

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                states = batch["states"].to(device)
                video  = frames.permute(0, 2, 1, 3, 4)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred = model(video)
                val_mae += F.l1_loss(pred, states).item()
        val_mae /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train_mae={train_mae:.4f}  val_mae={val_mae:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        # Best 체크포인트 저장
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_mae": val_mae,
            }, out_dir / "best.pt")
            print(f"  → best saved (val_mae={val_mae:.4f})")

        # 주기적 체크포인트
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_mae": val_mae,
            }, out_dir / f"epoch_{epoch:03d}.pt")

    print(f"\n[완료] best val_mae={best_val_mae:.4f}")
    print(f"[저장] {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
