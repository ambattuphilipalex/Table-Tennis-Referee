from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CACHE_ROOT = DATA_ROOT / "dino_cache"


class CachedBallDataset(Dataset):
    def __init__(self, cache_path):
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.tokens = blob["tokens"]  # (M, N, D) fp16, memory-mapped
        self.ball = blob["ball"]      # (M, 2)   fp32
        self.frames = blob["frames"]  # (M,)     int64
        self.meta = blob.get("meta", {})

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i].float(), self.ball[i].float()


def build_split(names, root=CACHE_ROOT):
    datasets = []
    for name in names:
        cache_path = Path(root) / name / "cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cache for {name!r} at {cache_path}")
        datasets.append(CachedBallDataset(cache_path))
    return ConcatDataset(datasets)