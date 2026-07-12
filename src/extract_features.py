import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ball_head import BallHead
from cache_dataset import CACHE_ROOT, CachedBallDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def forward_cls(model: BallHead, tokens: torch.Tensor, device: str) -> torch.Tensor:
    x = model.proj(tokens.to(device))
    cls = model.cls.expand(x.size(0), -1, -1)
    x = model.blocks(torch.cat([cls, x], dim=1))
    x = model.norm(x)
    return x[:, 0]


def run_game(model: BallHead, game: str, cache_root: Path, device: str, batch_size: int) -> None:
    cache_path = cache_root / game / "cache.pt"
    if not cache_path.exists():
        print(f"{game}: no cache at {cache_path}, skip")
        return

    ds = CachedBallDataset(cache_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    cls_chunks = []
    for tokens, _ in loader:
        cls_chunks.append(forward_cls(model, tokens, device).float().cpu())

    features = torch.cat(cls_chunks)  # [M, 384]

    frames = torch.tensor([int(ds.frames[j]) for j in ds.idx], dtype=torch.int64)

    out_path = cache_path.parent / "cls_token_features.pt"
    torch.save({"features": features, "frames": frames}, out_path)
    print(f"{game}: wrote {out_path}  features {tuple(features.shape)} "
          f"({len(ds)} frames, bad-label-filtered)")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = BallHead().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded ckpt {args.ckpt} (epoch {ckpt.get('epoch')})  device: {device}")

    for game in args.games:
        run_game(model, game, args.cache_root, device, args.batch_size)


if __name__ == "__main__":
    main()