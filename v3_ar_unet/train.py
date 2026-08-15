"""
Non-AR + bigger UNet + State Sequence Transformer training.

train_step: for each t in 1..15, pass full states[0..t] to StateSequenceEncoder.
            t passed as tensor through checkpoint to avoid closure bug.
val_step:   same non-AR generation with state sequence conditioning.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import EpisodeIndex, LeRobotSequenceDataset
from loss import DINOLoss, R3DLoss, ActionMAELoss
from model import ImageEditingModel

SEQ_LEN  = 16
W_DINO   = 0.3
W_R3D    = 0.3
W_ACTION = 0.4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-root",        default="../data/train")
    p.add_argument("--action-stats",      default="../data/train/so100_action_statistics.json")
    p.add_argument("--action-ckpt",       default="runs/extractor_v2/best.pt")
    p.add_argument("--index-cache",       default="episode_index_filtered.pkl")
    p.add_argument("--output-dir",        default="runs/v4_noar_seq")
    p.add_argument("--samples-per-epoch", type=int, default=3_000)
    p.add_argument("--batch-size",        type=int, default=4)
    p.add_argument("--num-workers",       type=int, default=4)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--max-epochs",        type=int, default=100)
    p.add_argument("--total-steps",       type=int, default=20_000)
    p.add_argument("--warmup-steps",      type=int, default=200)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument("--log-every",         type=int, default=50)
    p.add_argument("--val-every",         type=int, default=500)
    p.add_argument("--save-every",        type=int, default=2000)
    p.add_argument("--amp",               action="store_true", default=True)
    p.add_argument("--resume",            default=None)
    return p.parse_args()


def train_step(
    model, frames, states,
    dino_fn, r3d_fn, action_fn,
    optimizer, scaler, trainable, grad_clip, amp,
):
    optimizer.zero_grad(set_to_none=True)

    frame_0 = frames[:, 0]  # (B, 3, H, W)

    # t passed as a LongTensor so checkpoint can track it without closure capture bug
    def step_fn(f0, states_full, gt_i, t_tensor):
        t = int(t_tensor.item())
        with torch.cuda.amp.autocast(enabled=amp):
            pred, _ = model(f0, states_full, t)
            d_loss = dino_fn(pred, gt_i)
        return pred.float(), d_loss.float()

    pred_frames = []
    dino_terms  = []

    for t in range(1, SEQ_LEN):
        t_tensor = torch.tensor(t, dtype=torch.long, device=frame_0.device)
        pred, d_loss = checkpoint(
            step_fn,
            frame_0, states, frames[:, t], t_tensor,
            use_reentrant=False,
        )
        pred_frames.append(pred.clamp(-1, 1))
        dino_terms.append(d_loss)

    dino_avg = torch.stack(dino_terms).mean()

    pred_video = torch.cat(
        [frames[:, :1], torch.stack(pred_frames, dim=1)], dim=1
    )  # (B, 16, 3, H, W)

    with torch.cuda.amp.autocast(enabled=amp):
        r3d_loss = r3d_fn(pred_video, frames)
        action_total, action_per_dim = action_fn(pred_video, states)

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
    for k, v in action_per_dim.items():
        logs[k] = v.item()
    return logs


def val_step(model, frames, states, dino_fn, r3d_fn, action_fn, amp):
    dino_sum = 0.0
    with torch.no_grad():
        frame_0 = frames[:, 0]
        pred_frames_det = []

        for t in range(1, SEQ_LEN):
            with torch.cuda.amp.autocast(enabled=amp):
                pred, _ = model(frame_0, states, t)
                d_loss = dino_fn(pred, frames[:, t])
            dino_sum += d_loss.item()
            pred_frames_det.append(pred.clamp(-1, 1))

        pred_video = torch.cat(
            [frames[:, :1], torch.stack(pred_frames_det, dim=1)], dim=1
        )
        r3d_loss_val = r3d_fn(pred_video, frames).item()
        action_loss_val, _ = action_fn(pred_video, states)
        action_loss_val = action_loss_val.item()

    dino_avg = dino_sum / (SEQ_LEN - 1)
    return W_DINO * dino_avg + W_ACTION * action_loss_val + W_R3D * r3d_loss_val


def save_checkpoint(path, model, optimizer, scaler, lr_sched, step, best_loss):
    torch.save({
        "step": step, "best_loss": best_loss,
        "unet":          model.unet.state_dict(),
        "state_encoder": model.state_encoder.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scaler":        scaler.state_dict() if scaler else None,
        "lr_sched":      lr_sched.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, scaler, lr_sched):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.unet.load_state_dict(ckpt["unet"])
    model.state_encoder.load_state_dict(ckpt["state_encoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if "lr_sched" in ckpt:
        lr_sched.load_state_dict(ckpt["lr_sched"])
    return ckpt["step"], ckpt["best_loss"]


def split_index_by_episode(index, val_ratio=0.05, seed=42):
    import random
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

    cache = Path(args.index_cache)
    index = EpisodeIndex.load(cache) if cache.exists() else EpisodeIndex(args.train_root)
    if not cache.exists():
        index.save(cache)

    train_index, val_index = split_index_by_episode(index)
    train_ds = LeRobotSequenceDataset(train_index, args.action_stats, args.samples_per_epoch)
    val_ds   = LeRobotSequenceDataset(val_index,   args.action_stats, samples_per_epoch=500)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=2, pin_memory=True)

    model = ImageEditingModel(cond_dim=512).to(device)
    dino_fn   = DINOLoss(device)
    r3d_fn    = R3DLoss(device)
    action_fn = ActionMAELoss(args.action_ckpt, device)

    trainable = list(model.unet.parameters()) + list(model.state_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.total_steps - args.warmup_steps), eta_min=args.lr * 0.01)
    lr_sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps])

    step, best_loss = 0, float("inf")
    if args.resume:
        step, best_loss = load_checkpoint(Path(args.resume), model, optimizer, scaler, lr_sched)
        print(f"Resumed from step {step}")

    for epoch in range(args.max_epochs):
        model.train()
        t0 = time.time()

        for batch in train_loader:
            frames = batch["frames"].to(device)
            states = batch["states"].to(device)

            logs = train_step(
                model, frames, states,
                dino_fn, r3d_fn, action_fn,
                optimizer, scaler, trainable, args.grad_clip, args.amp,
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
                      f"lr={current_lr:.2e}  "
                      f"{elapsed:.1f}s")
                for k, v in logs.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", current_lr, step)
                t0 = time.time()

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
                    save_checkpoint(out_dir / "best.ckpt", model, optimizer, scaler, lr_sched, step, best_loss)
                    print(f"  -> best saved (loss={best_loss:.4f})")
                model.train()

            if step % args.save_every == 0:
                save_checkpoint(out_dir / f"step{step}.ckpt", model, optimizer, scaler, lr_sched, step, best_loss)

            if step >= args.total_steps:
                print("Total steps reached.")
                writer.close()
                return

        print(f"Epoch {epoch} done.")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
