import sys, traceback
sys.path.insert(0, "/workspace/v3")
from dataset import EpisodeIndex, LeRobotSequenceDataset

index = EpisodeIndex.load("/workspace/v3/episode_index_filtered.pkl")
ds = LeRobotSequenceDataset(
    index,
    action_stats_path="/workspace/data/train/so100_action_statistics.json",
    samples_per_epoch=10,
)
print("First entry:", ds.entries[0])
for attempt in range(5):
    try:
        sample = ds._load_sample()
        print("OK:", sample["frames"].shape, sample["states"].shape)
        break
    except Exception:
        print(f"--- attempt {attempt} failed ---")
        traceback.print_exc()
