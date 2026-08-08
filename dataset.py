"""
LeRobot v2.1 sequence dataset for image editing regression.
Returns 16-frame sequences with corresponding observation.state values.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VIDEO_KEY = "observation.images.image"


def _load_action_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        d = json.load(f)
    return torch.tensor(d["mean"], dtype=torch.float32), torch.tensor(d["std"], dtype=torch.float32)


def _preprocess_frame(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 in [-1, 1], 원본 해상도 유지"""
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    return t * 2.0 - 1.0


class EpisodeIndex:
    """Scans train root and builds a flat list of (dataset_root, episode_index, n_frames)."""

    def __init__(self, train_root: str | Path):
        self.entries: list[tuple[Path, int, int]] = []
        train_root = Path(train_root)
        for info_path in sorted(train_root.rglob("meta/info.json")):
            dataset_root = info_path.parent.parent
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                continue
            chunks_size = info.get("chunks_size", 1000)
            total_episodes = info.get("total_episodes", 0)
            for ep_idx in range(total_episodes):
                chunk = ep_idx // chunks_size
                parquet_path = dataset_root / info["data_path"].format(
                    episode_chunk=chunk, episode_index=ep_idx
                )
                if not parquet_path.exists():
                    continue
                try:
                    import pandas as pd
                    df = pd.read_parquet(parquet_path, columns=["observation.state"])
                    n_frames = len(df)
                except Exception:
                    continue
                if n_frames < 2:
                    continue
                self.entries.append((dataset_root, ep_idx, n_frames))
        print(f"[EpisodeIndex] {len(self.entries)} episodes indexed")

    def save(self, path: str | Path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeIndex":
        import pickle
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.entries = pickle.load(f)
        print(f"[EpisodeIndex] loaded {len(obj.entries)} episodes")
        return obj


def _read_frames(mp4_path: Path, start: int, length: int) -> list[np.ndarray]:
    """Read `length` consecutive frames starting at `start`. Returns list of (H,W,3) uint8."""
    frames = []
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == length:
                break
    return frames


class LeRobotSequenceDataset(Dataset):
    """
    Each __getitem__ returns a 16-frame sequence sampled randomly from an episode.

    Returns dict:
        frames : (16, 3, H, W) float32 [-1, 1]   — 16 consecutive frames
        states : (16, 6) float32 normalized        — observation.state per frame
    """
    SEQ_LEN = 16

    def __init__(
        self,
        index: EpisodeIndex,
        action_stats_path: str,
        samples_per_epoch: int = 100_000,
    ):
        # 16프레임 이상인 에피소드만
        self.entries = [(r, e, n) for r, e, n in index.entries if n >= self.SEQ_LEN]
        self.samples_per_epoch = samples_per_epoch
        self.action_mean, self.action_std = _load_action_stats(action_stats_path)

        self._weights = torch.tensor(
            [max(n - self.SEQ_LEN + 1, 0) for _, _, n in self.entries], dtype=torch.float32
        )
        self._weights /= self._weights.sum()
        print(f"[LeRobotSequenceDataset] {len(self.entries)} episodes with ≥{self.SEQ_LEN} frames")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> dict:
        for _ in range(10):
            try:
                return self._load_sample()
            except Exception:
                continue
        raise RuntimeError("Failed to load sample after 10 retries")

    def _load_sample(self) -> dict:
        import pandas as pd

        ep_pos = torch.multinomial(self._weights, 1).item()
        dataset_root, ep_idx, n_frames = self.entries[ep_pos]

        start = random.randint(0, n_frames - self.SEQ_LEN)

        if not hasattr(self, "_info_cache"):
            self._info_cache = {}
        info_key = str(dataset_root)
        if info_key not in self._info_cache:
            self._info_cache[info_key] = json.loads((dataset_root / "meta" / "info.json").read_text())
        info = self._info_cache[info_key]
        chunks_size = info.get("chunks_size", 1000)
        chunk = ep_idx // chunks_size

        # load states
        parquet_path = dataset_root / info["data_path"].format(
            episode_chunk=chunk, episode_index=ep_idx
        )
        df = pd.read_parquet(parquet_path, columns=["observation.state"])
        raw_states = torch.from_numpy(
            np.array(df["observation.state"].iloc[start:start + self.SEQ_LEN].tolist(), dtype=np.float32)
        )  # (16, 6)
        states = (raw_states - self.action_mean) / self.action_std

        # load frames
        mp4_path = dataset_root / info["video_path"].format(
            episode_chunk=chunk, episode_index=ep_idx, video_key=VIDEO_KEY
        )
        raw_frames = _read_frames(mp4_path, start, self.SEQ_LEN)
        frames = torch.stack([_preprocess_frame(f) for f in raw_frames])  # (16, 3, H, W)

        return {"frames": frames, "states": states}
