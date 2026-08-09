"""
SD1.5 VAE reconstruction check on robot camera images.
Encodes 5 sample frames and decodes them back.
Saves side-by-side comparison images to check/
"""
import os
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL

TRAIN_ROOT = Path("../data/train")
OUT_DIR = Path("check_vae_out")
N_SAMPLES = 5
TARGET_H, TARGET_W = 320, 512


def find_sample_frames(train_root: Path, n: int) -> list[Path]:
    imgs = list(train_root.rglob("*.mp4"))
    if not imgs:
        raise FileNotFoundError(f"No mp4 found under {train_root}")
    chosen = random.sample(imgs, min(n * 3, len(imgs)))
    frames = []
    for mp4 in chosen:
        try:
            import av
            with av.open(str(mp4)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i == 5:
                        frames.append(frame.to_image())
                        break
        except Exception:
            continue
        if len(frames) >= n:
            break
    return frames[:n]


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((TARGET_W, TARGET_H), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x * 2.0 - 1.0
    return x.permute(2, 0, 1).unsqueeze(0)


def postprocess(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).permute(1, 2, 0).float().clamp(-1, 1)
    t = ((t + 1.0) / 2.0 * 255.0).byte().numpy()
    return Image.fromarray(t)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading SD1.5 VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
    )
    vae.eval()
    vae.requires_grad_(False)

    print("Finding sample frames...")
    frames = find_sample_frames(TRAIN_ROOT, N_SAMPLES)
    if not frames:
        raise RuntimeError("No frames extracted. Check data path.")

    print(f"Running encode→decode on {len(frames)} frames...")
    for i, img in enumerate(frames):
        x = preprocess(img)

        with torch.no_grad():
            latent = vae.encode(x).latent_dist.mean
            recon = vae.decode(latent).sample

        print(f"  [{i}] latent shape: {latent.shape}  min/max: {latent.min():.3f} / {latent.max():.3f}")

        orig_img = postprocess(x)
        recon_img = postprocess(recon)

        combined = Image.new("RGB", (TARGET_W * 2 + 8, TARGET_H), (128, 128, 128))
        combined.paste(orig_img, (0, 0))
        combined.paste(recon_img, (TARGET_W + 8, 0))
        combined.save(OUT_DIR / f"sample_{i:02d}.png")
        print(f"  saved check_vae_out/sample_{i:02d}.png  (left=orig, right=recon)")

    print("\nDone. 결과 확인:")
    print(f"  latent 크기가 (1, 4, {TARGET_H//8}, {TARGET_W//8}) = (1, 4, 40, 64) 이면 정상")
    print("  재구성 이미지가 원본과 유사하면 VAE 사용 가능")
    print("  로봇 엣지/그리퍼 부분이 많이 뭉개지면 VAE가 병목 → 재검토 필요")


if __name__ == "__main__":
    main()
