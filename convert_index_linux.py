"""
Windows에서 만든 episode_index_filtered.pkl의 WindowsPath를
Linux PosixPath로 변환. 서버에서 한 번만 실행.
"""

import pickle
import pathlib
import sys
from pathlib import PurePosixPath, Path

WIN_PREFIX = r"C:\Users\김승원\Desktop\IAIC"
LINUX_PREFIX = "/workspace"


class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PurePosixPath  # WindowsPath → PurePosixPath로 대체
        return super().find_class(module, name)


def win_to_linux(p) -> Path:
    s = str(p).replace("\\", "/")
    # C:/Users/김승원/Desktop/IAIC/... → /workspace/...
    for win in [
        "C:/Users/김승원/Desktop/IAIC",
        "C:\\Users\\김승원\\Desktop\\IAIC",
    ]:
        if s.startswith(win.replace("\\", "/")):
            s = LINUX_PREFIX + s[len(win.replace("\\", "/")):]
            break
    return Path(s)


with open("episode_index_filtered.pkl", "rb") as f:
    entries = WindowsPathUnpickler(f).load()

converted = []
for (dataset_root, ep_idx, n_frames) in entries:
    converted.append((win_to_linux(dataset_root), ep_idx, n_frames))

with open("episode_index_filtered.pkl", "wb") as f:
    pickle.dump(converted, f)

print(f"변환 완료: {len(converted)}개 에피소드")
print(f"샘플: {converted[0][0]}")
