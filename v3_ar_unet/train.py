"""
Stage 2 training: 16-frame teacher forcing with gradient-checkpointed rollout.

Loss = 0.3 * DINO + 0.3 * R3D + 0.4 * Action, single backward at sequence end.
Each step (model forward + DINO + Action) is wrapped in torch.utils.checkpoint
so per-step activations aren't kept in memory simultaneously — this is what
lets R3D (needs the full 16-frame video) receive real gradient without OOM,
instead of the old detached/monitoring-only R3D.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import EpisodeIndex, LeRobotSequenceDataset
from loss import DINOLoss, R3DLoss, ActionMAELoss
from model import ImageEditingModel

SEQ_LEN = 16
W_DINO   = 0.3
W_R3D    = 0.3
W_ACTION = 0.4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root",        default="../data/train")
    p.add_argument("--action-stats",      default="../data/train/so100_action_statistics.json")
    p.add_argument("--action-ckpt",       default="runs/extractor_v2/best.pt")
    p.add_argument("--index-cache",       default="episode_index_filtered.pkl")
    p.add_argument("--output-dir",        default="runs/v2_ar_unet")
    # total-steps 기준(2일치 GPU 예산 역산, 20 epoch x 500 step/epoch)으로 재조정.
    # 실제 페이스가 다르면 --total-steps만 바꿔서 재실행.
    p.add_argument("--samples-per-epoch", type=int, default=3_000)
    p.add_argument("--batch-size",        type=int, default=4)
    p.add_argument("--num-workers",       type=int, default=4)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--max-epochs",        type=int, default=27)
    p.add_argument("--total-steps",       type=int, default=20_000,
                    help="LR 스케줄(warmup+cosine decay) 기준 총 스텝 수. 이 스텝 이후 LR은 eta_min에 고정.")
    p.add_argument("--warmup-steps",      type=int, default=150)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument("--log-every",         type=int, default=50)
    p.add_argument("--val-every",         type=int, default=500)
    p.add_argument("--save-every",        type=int, default=2000)
    p.add_argument("--amp",               action="store_true", default=True)
    p.add_argument("--resume",            default=None)
    return p.parse_args()


def get_tf_prob(step: int) -> float:
    if step < 700:
        return 1.0
    elif step < 7000:
        return 1.0 - (step - 700) / 6300
    else:
        return 0.0


def train_step(
    model: ImageEditingModel,
    frames: torch.Tensor,   # (B, 16, 3, H, W)
    states: torch.Tensor,   # (B, 16, 6)
    dino_fn: DINOLoss,
    r3d_fn: R3DLoss,
    action_fn: ActionMAELoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    trainable: list,
    grad_clip: float,
    amp: bool,
    step: int = 0,
) -> dict:
    """
    Gradient-checkpointed rollout:
    - 각 스텝(model forward + DINO)을 checkpoint()로 감싸서 중간 activation을
      저장하지 않고 backward 때 재계산 → 15프레임 그래프를 끝까지 살려둬도
      메모리가 안 터짐
    - R3D + Action은 채점기와 동일하게 16프레임 전체 영상을 한 번에 넣어서 계산
      (Action extractor가 BiLSTM으로 시간축 전체를 보도록 학습됐는데, 예전엔
      프레임 하나씩(T=1) 넣어서 그 맥락이 무의미했음 — 이제 R3D처럼 T=16으로 통일)
    - 시퀀스 끝에서 DINO+R3D+Action 합친 total_loss로 단 한 번 backward
      (R3D/Action 둘 다 실제로 gradient에 관여함)

    Returns logging dict.
    """
    optimizer.zero_grad(set_to_none=True)

    def step_fn(frame_t, state_t, state_t1, gt_next):
        with torch.cuda.amp.autocast(enabled=amp):
            pred, _ = model(frame_t, state_t, state_t1)
            d_loss = dino_fn(pred, gt_next)
        return pred.float(), d_loss.float()

    pred_frames = []
    dino_terms = []

    tf_prob = get_tf_prob(step)
    curr_frame = frames[:, 0]
    for t in range(SEQ_LEN - 1):
        input_frame = frames[:, t] if (tf_prob > 0 and random.random() < tf_prob) else curr_frame
        pred, d_loss = checkpoint(
            step_fn,
            input_frame, states[:, t], states[:, t + 1], frames[:, t + 1],
            use_reentrant=False,
        )
        curr_frame = pred.clamp(-1, 1)
        pred_frames.append(curr_frame)
        dino_terms.append(d_loss)

    dino_avg = torch.stack(dino_terms).mean()

    pred_video = torch.cat(
        [frames[:, :1], torch.stack(pred_frames, dim=1)], dim=1
    )  # (B, 16, 3, H, W), grad-attached
    with torch.cuda.amp.autocast(enabled=amp):
        r3d_loss = r3d_fn(pred_video, frames)
        # action_total: 6개 관절 MAE 평균, 가중치 없음 — 채점기와 동일 방식
        action_total, action_per_dim = action_fn(pred_video, states)  # (B,16,3,H,W) vs (B,16,6)

    total_loss = W_DINO * dino_avg + W_R3D * r3d_loss + W_ACTION * action_total

    if scaler:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()

    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    logs = {
        "dino_loss":   dino_avg.item(),
        "r3d_loss":    r3d_loss.item(),
        "action_loss": action_total.item(),
        "total_loss":  total_loss.item(),
    }
    # 관절별 개별 MAE(로깅 전용, 학습엔 영향 없음) — 어느 관절이 유독 안 줄어드는지 진단용
    for k, v in action_per_dim.items():
        logs[k] = v.item()
    return logs


def val_step(
    model: ImageEditingModel,
    frames: torch.Tensor,
    states: torch.Tensor,
    dino_fn: DINOLoss,
    r3d_fn: R3DLoss,
    action_fn: ActionMAELoss,
    amp: bool,
) -> float:
    dino_sum = 0.0

    with torch.no_grad():
        pred_frames_det = []
        curr_frame = frames[:, 0]
        for t in range(SEQ_LEN - 1):
            with torch.cuda.amp.autocast(enabled=amp):
                pred, _ = model(curr_frame, states[:, t], states[:, t + 1])
                d_loss = dino_fn(pred, frames[:, t + 1])
            dino_sum += d_loss.item()
            curr_frame = pred.clamp(-1, 1)
            pred_frames_det.append(curr_frame)

        pred_video = torch.cat(
            [frames[:, :1], torch.stack(pred_frames_det, dim=1)], dim=1
        )
        r3d_loss_val = r3d_fn(pred_video, frames).item()
        action_loss_val, _ = action_fn(pred_video, states)
        action_loss_val = action_loss_val.item()

    dino_avg = dino_sum / (SEQ_LEN - 1)
    return W_DINO * dino_avg + W_ACTION * action_loss_val + W_R3D * r3d_loss_val


def save_checkpoint(path: Path, model, action_fn, optimizer, scaler, lr_sched, step, best_loss):
    torch.save({
        "step": step,
        "best_loss": best_loss,
        "unet": model.unet.state_dict(),
        "action_mlp": model.action_mlp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "lr_sched": lr_sched.state_dict(),
    }, path)


def load_checkpoint(path, model, action_fn, optimizer, scaler, lr_sched):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.unet.load_state_dict(ckpt["unet"])
    model.action_mlp.load_state_dict(ckpt["action_mlp"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if "lr_sched" in ckpt:
        lr_sched.load_state_dict(ckpt["lr_sched"])
    return ckpt["step"], ckpt["best_loss"]


def split_index_by_episode(index: EpisodeIndex, val_ratio: float = 0.05, seed: int = 42):
    """에피소드 단위로 train/val을 분리(윈도우 단위로 섞으면 같은 에피소드 인접 구간이
    양쪽에 다 들어가서 val이 held-out이 아니게 됨). 고정 seed라 재실행해도 같은 분할."""
    entries = list(index.entries)
    random.Random(seed).shuffle(entries)
    n_val = max(1, int(len(entries) * val_ratio))

    train_index = EpisodeIndex.__new__(EpisodeIndex)
    train_index.entries = entries[n_val:]
    val_index = EpisodeIndex.__new__(EpisodeIndex)
    val_index.entries = entries[:n_val]
    print(f"[split] train episodes: {len(train_index.entries)}, val episodes: {len(val_index.entries)}")
    return train_index, val_index


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    # ── data ──────────────────────────────────────────────────────────────────
    cache = Path(args.index_cache)
    index = EpisodeIndex.load(cache) if cache.exists() else EpisodeIndex(args.train_root)
    if not cache.exists():
        index.save(cache)

    train_index, val_index = split_index_by_episode(index)
    train_ds = LeRobotSequenceDataset(train_index, args.action_stats, args.samples_per_epoch)
    val_ds   = LeRobotSequenceDataset(val_index, args.action_stats, samples_per_epoch=500)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=2,
                              pin_memory=True)

    # ── model & loss ──────────────────────────────────────────────────────────
    model = ImageEditingModel(cond_dim=512).to(device)

    dino_fn   = DINOLoss(device)
    r3d_fn    = R3DLoss(device)
    action_fn = ActionMAELoss(args.action_ckpt, device)

    trainable = list(model.unet.parameters()) + list(model.action_mlp.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    # warmup(선형) + cosine decay, total-steps 이후엔 step()을 안 불러서 eta_min에 고정(다시 안 올라감)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.total_steps - args.warmup_steps), eta_min=args.lr * 0.01
    )
    lr_sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps]
    )

    step, best_loss = 0, float("inf")
    if args.resume:
        step, best_loss = load_checkpoint(Path(args.resume), model, action_fn, optimizer, scaler, lr_sched)
        print(f"Resumed from step {step}")

    # ── train loop ────────────────────────────────────────────────────────────
    for epoch in range(args.max_epochs):
        model.train()
        t0 = time.time()

        for batch in train_loader:
            frames = batch["frames"].to(device)   # (B, 16, 3, H, W)
            states = batch["states"].to(device)   # (B, 16, 6)

            logs = train_step(
                model, frames, states,
                dino_fn, r3d_fn, action_fn,
                optimizer, scaler, trainable, args.grad_clip, args.amp,
                step=step,
            )

            step += 1

            if step <= args.total_steps:
                lr_sched.step()
            current_lr = optimizer.param_groups[0]["lr"]

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"[ep{epoch} step{step}] "
                      f"total={logs['total_loss']:.4f}  "
                      f"dino={logs['dino_loss']:.4f}  "
                      f"r3d={logs['r3d_loss']:.4f}  "
                      f"action={logs['action_loss']:.4f}  "
                      f"tf={get_tf_prob(step):.2f}  "
                      f"lr={current_lr:.2e}  "
                      f"{elapsed:.1f}s")
                for k, v in logs.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", current_lr, step)
                t0 = time.time()

            # ── validation ────────────────────────────────────────────────────
            if step % args.val_every == 0:
                model.eval()
                val_totals = []
                for vb in val_loader:
                    vf = vb["frames"].to(device)
                    vs = vb["states"].to(device)
                    vl = val_step(model, vf, vs, dino_fn, r3d_fn, action_fn, args.amp)
                    val_totals.append(vl)
                val_loss = sum(val_totals) / len(val_totals)
                writer.add_scalar("val/total_loss", val_loss, step)
                print(f"  [val] step={step}  val_loss={val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(out_dir / "best.ckpt", model, action_fn, optimizer, scaler, lr_sched, step, best_loss)
                    print(f"  → best saved (loss={best_loss:.4f})")
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(out_dir / f"step{step}.ckpt", model, action_fn, optimizer, scaler, lr_sched, step, best_loss)

        save_checkpoint(out_dir / f"epoch{epoch}.ckpt", model, action_fn, optimizer, scaler, lr_sched, step, best_loss)
        print(f"Epoch {epoch} done.")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
