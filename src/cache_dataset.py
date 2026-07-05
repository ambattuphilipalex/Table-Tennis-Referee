from pathlib import Path
import json
import torch
from torch.utils.data import ConcatDataset, Dataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CACHE_ROOT = DATA_ROOT / "dino_cache"


class CachedBallDataset(Dataset):
    def __init__(self, cache_path):
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.tokens = blob["tokens"]        
        self.ball = blob["ball"]
        self.frames = blob["frames"]
        self.meta = blob.get("meta", {})

        game = cache_path.parent.name
        bad_path = cache_path.parents[2] / "bad_label_candidates" / f"{game}.json"
        bad = set(json.load(open(bad_path))) if bad_path.exists() else set()
        self.idx = [i for i, f in enumerate(self.frames.tolist()) if int(f) not in bad]
        if bad:
            print(f"{game}: keeping {len(self.idx)}/{len(self.frames)} frames")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tokens[j].float(), self.ball[j].float()


def build_split(names, root=CACHE_ROOT):
    datasets = []
    for name in names:
        cache_path = Path(root) / name / "cache.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cache for {name!r} at {cache_path}")
        datasets.append(CachedBallDataset(cache_path))
    return ConcatDataset(datasets)