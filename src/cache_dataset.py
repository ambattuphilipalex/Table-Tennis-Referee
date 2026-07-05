from pathlib import Path
import json
import torch
from torch.utils.data import ConcatDataset, Dataset

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
CACHE_ROOT = DATA_ROOT / "dino_cache"


class CachedBallDataset(Dataset):
    def __init__(self, cache_path):
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        tokens = blob["tokens"]
        ball = blob["ball"]
        frames = blob["frames"]
        self.meta = blob.get("meta", {})

        game = cache_path.parent.name
        bad_path = cache_path.parents[2] / "bad_label_candidates" / f"{game}.json"
        if bad_path.exists():
            bad = set(json.load(open(bad_path)))
            if bad:
                keep = torch.tensor([int(f) not in bad for f in frames.tolist()])
                tokens, ball, frames = tokens[keep], ball[keep], frames[keep]

        self.tokens = tokens
        self.ball = ball
        self.frames = frames

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