"""
Non-AR inference: predict each frame independently from frame_0.

frame_0 (PNG) + states[0..15] (NPY) →
  frame_1 = model(frame_0, state_0, state_1)
  frame_2 = model(frame_0, state_0, state_2)   ← always from frame_0, no AR
  ...
  frame_15 = model(frame_0, state_0, state_15)
  → save [frame_0..frame_15] as sample_XXXXXX.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from model import ImageEditingModel


def load_action_stats(path):
    with open(path) as f:
        d = json.load(f)
    return torch.tensor(d["mean"], dtype=torch.float32), torch.tensor(d["std"], dtype=torch.float32)


def preprocess_frame(img_np):
    t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    return t * 2.0 - 1.0


def save_mp4(frames, path, fps=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames.clamp(-1.0, 1.0)
    arr = ((arr + 1.0) / 2.0 * 255.0).to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    _, h, w, _ = arr.shape

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    for frame_np in arr:
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.unet.load_state_dict(ckpt["unet"])
    model.action_mlp.load_state_dict(ckpt["action_mlp"])
    print(f"[ckpt] step={ckpt.get('step', '?')}, best_loss={ckpt.get('best_loss', float('inf')):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root",    default="../data/eval")
    parser.add_argument("--action-stats", default="../data/train/so100_action_statistics.json")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--output-dir",   default="../submission_kit/input_videos")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--fps",          type=int, default=6)
    parser.add_argument("--amp",          action="store_true", default=True)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    eval_root = Path(args.eval_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_ids  = {p.stem for p in (eval_root / "images").glob("*.png")}
    action_ids = {p.stem for p in (eval_root / "actions").glob("*.npy")}
    sample_ids = sorted(image_ids & action_ids)
    print(f"[eval] {len(sample_ids)} samples found")

    act_mean, act_std = load_action_stats(args.action_stats)
    act_mean = act_mean.to(device)
    act_std  = act_std.to(device)

    model = ImageEditingModel(cond_dim=512).to(device)
    load_checkpoint(Path(args.checkpoint), model)
    model.eval()

    already_done = {p.stem for p in out_dir.glob("*.mp4")}
    todo = [s for s in sample_ids if s not in already_done]
    print(f"[eval] {len(already_done)} already done, {len(todo)} remaining")

    use_amp = args.amp and device.type == "cuda"
    B = args.batch_size

    with torch.no_grad():
        for batch_start in range(0, len(todo), B):
            batch_ids = todo[batch_start : batch_start + B]
            b = len(batch_ids)

            frame0_list = []
            states_list = []
            for sid in batch_ids:
                img_np = np.asarray(Image.open(eval_root / "images" / f"{sid}.png").convert("RGB"))
                frame0_list.append(preprocess_frame(img_np))
                raw = np.load(eval_root / "actions" / f"{sid}.npy").astype(np.float32)
                st  = torch.from_numpy(raw).to(device)
                st  = (st - act_mean) / act_std
                states_list.append(st)

            frame_0 = torch.stack(frame0_list).to(device)  # (B, 3, H, W)
            states  = torch.stack(states_list)              # (B, 16, 6)
            state_0 = states[:, 0]                          # (B, 6) — initial state

            all_frames = [frame_0.cpu()]

            with torch.cuda.amp.autocast(enabled=use_amp):
                for t in range(1, 16):
                    # always predict from frame_0 — no AR chaining
                    t_norm = torch.full((frame_0.shape[0],), t / 15.0, device=frame_0.device)
                    pred, _ = model(frame_0, state_0, states[:, t], t_norm)
                    pred = pred.clamp(-1.0, 1.0)
                    all_frames.append(pred.cpu())

            video_batch = torch.stack(all_frames, dim=1)  # (B, 16, 3, H, W)
            for i, sid in enumerate(batch_ids):
                save_mp4(video_batch[i], out_dir / f"{sid}.mp4", args.fps)

            n_done = len(already_done) + batch_start + b
            print(f"[infer] {batch_start + b}/{len(todo)} done  ({n_done}/{len(sample_ids)} total)")

    print(f"[infer] 완료. 저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
