"""
SDXL VAE vs SD1.5 VAE reconstruction comparison.
Saves side-by-side: orig | SD1.5 recon | SDXL recon
"""
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


def find_sample_frames(train_root: Path, n: int) -> list:
    imgs = list(train_root.rglob("*.mp4"))
    random.seed(42)
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


def recon_mse(orig: torch.Tensor, recon: torch.Tensor) -> float:
    return ((orig - recon) ** 2).mean().item()


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading SD1.5 VAE...")
    vae_15 = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae_15.eval().requires_grad_(False)

    print("Loading SDXL VAE...")
    vae_xl = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae_xl.eval().requires_grad_(False)

    print("Finding sample frames (same seed as before)...")
    frames = find_sample_frames(TRAIN_ROOT, N_SAMPLES)

    mse_15_total, mse_xl_total = 0.0, 0.0

    for i, img in enumerate(frames):
        x = preprocess(img)

        with torch.no_grad():
            lat_15 = vae_15.encode(x).latent_dist.mean
            rec_15 = vae_15.decode(lat_15).sample

            lat_xl = vae_xl.encode(x).latent_dist.mean
            rec_xl = vae_xl.decode(lat_xl).sample

        mse_15 = recon_mse(x, rec_15.clamp(-1, 1))
        mse_xl = recon_mse(x, rec_xl.clamp(-1, 1))
        mse_15_total += mse_15
        mse_xl_total += mse_xl

        print(f"  [{i}] SD1.5 MSE: {mse_15:.5f}  |  SDXL MSE: {mse_xl:.5f}")

        orig_img = postprocess(x)
        rec15_img = postprocess(rec_15)
        rec_xl_img = postprocess(rec_xl)

        gap = 4
        combined = Image.new("RGB", (TARGET_W * 3 + gap * 2, TARGET_H), (128, 128, 128))
        combined.paste(orig_img, (0, 0))
        combined.paste(rec15_img, (TARGET_W + gap, 0))
        combined.paste(rec_xl_img, (TARGET_W * 2 + gap * 2, 0))
        combined.save(OUT_DIR / f"compare_{i:02d}.png")
        print(f"     saved compare_{i:02d}.png  (left=orig, mid=SD1.5, right=SDXL)")

    print(f"\n평균 MSE  SD1.5: {mse_15_total/N_SAMPLES:.5f}  |  SDXL: {mse_xl_total/N_SAMPLES:.5f}")
    print("숫자 낮을수록 복원 품질 좋음")


if __name__ == "__main__":
    main()
