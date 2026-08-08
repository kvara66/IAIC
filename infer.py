"""
v3/infer.py

eval 데이터 216개 → 16프레임 MP4 생성 (autoregressive chaining)

Flow:
  frame_0 (PNG) + states[0..15] (NPY, raw → z-score) →
  frame_1 = model(frame_0, s0, s1)
  frame_2 = model(frame_1_pred, s1, s2)  ← autoregressive
  ...
  frame_15 = model(frame_14_pred, s14, s15)
  → save [frame_0..frame_15] as sample_XXXXXX.mp4

Usage (서버):
  cd /workspace/v3
  python infer.py \\
    --checkpoint runs/v1/best.ckpt \\
    --eval-root ../data/eval \\
    --action-stats ../data/train/so100_action_statistics.json \\
    --output-dir ../submission_kit/input_videos \\
    --batch-size 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import ImageEditingModel

TARGET_H = 320
TARGET_W = 512


def load_action_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        d = json.load(f)
    return torch.tensor(d["mean"], dtype=torch.float32), torch.tensor(d["std"], dtype=torch.float32)


def preprocess_frame(img_np: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, TARGET_H, TARGET_W) float32 [-1, 1], letterbox"""
    t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    _, _, h, w = t.shape
    scale = min(TARGET_H / h, TARGET_W / w)
    rh = max(1, round(h * scale))
    rw = max(1, round(w * scale))
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pl = (TARGET_W - rw) // 2
    pr = TARGET_W - rw - pl
    pt = (TARGET_H - rh) // 2
    pb = TARGET_H - rh - pt
    t = F.pad(t, (pl, pr, pt, pb), value=0.0)
    return (t.squeeze(0) * 2.0 - 1.0)  # (3, H, W)


def save_mp4(frames: torch.Tensor, path: Path, fps: int = 6) -> None:
    """frames: (T, 3, H, W) float32 [-1, 1]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames.clamp(-1.0, 1.0)
    arr = ((arr + 1.0) / 2.0 * 255.0).to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, 3)

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = TARGET_W
    stream.height = TARGET_H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for frame_np in arr:
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def load_checkpoint(path: Path, model: ImageEditingModel) -> None:
    ckpt = torch.load(path, map_location="cpu")
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

    model = ImageEditingModel(cond_dim=256).to(device)
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
                st  = (st - act_mean) / act_std  # (16, 6)
                states_list.append(st)

            curr_frames = torch.stack(frame0_list).to(device)  # (B, 3, H, W)
            states      = torch.stack(states_list)              # (B, 16, 6)

            all_frames = [curr_frames.cpu()]  # store on CPU to save VRAM

            with torch.cuda.amp.autocast(enabled=use_amp):
                for t in range(15):
                    pred, _ = model(curr_frames, states[:, t], states[:, t + 1])
                    pred = pred.clamp(-1.0, 1.0)
                    all_frames.append(pred.cpu())
                    curr_frames = pred

            # (B, 16, 3, H, W)
            video_batch = torch.stack(all_frames, dim=1)

            for i, sid in enumerate(batch_ids):
                save_mp4(video_batch[i], out_dir / f"{sid}.mp4", args.fps)

            n_done = len(already_done) + batch_start + b
            print(f"[infer] {batch_start + b}/{len(todo)} done  ({n_done}/{len(sample_ids)} total)")

    print(f"[infer] 완료. 저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
