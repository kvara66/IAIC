"""Quick sanity check for LeRobotSequenceDataset"""
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import EpisodeIndex, LeRobotSequenceDataset

TRAIN_ROOT = "../data/train"
ACTION_STATS = "../data/train/so100_action_statistics.json"
INDEX_CACHE = "episode_index.pkl"

cache = Path(INDEX_CACHE)
if cache.exists():
    index = EpisodeIndex.load(cache)
else:
    index = EpisodeIndex(TRAIN_ROOT)
    index.save(cache)

dataset = LeRobotSequenceDataset(index, ACTION_STATS, samples_per_epoch=10)
loader = DataLoader(dataset, batch_size=2, num_workers=0)

batch = next(iter(loader))
print("frames:", batch["frames"].shape)   # (2, 16, 3, 320, 512)
print("states:", batch["states"].shape)   # (2, 16, 6)
print("frames min/max:", batch["frames"].min().item(), batch["frames"].max().item())
print("states sample:", batch["states"][0, 0])
print("OK")
