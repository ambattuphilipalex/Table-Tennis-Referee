import argparse
import csv
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from ball_head import BallHead
from cache_dataset import CACHE_ROOT, CachedBallDataset
from metrics import DEFAULT_THRESHOLDS, ORIG_WH, denormalize, mean_predictor_baseline, pixel_errors, summarize
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--games", nargs="+", default=["game_5"])
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    return p.parse_args()


def load_dataset(game: str, cache_root: Path) -> CachedBallDataset:
    cache_path = cache_root / game / "cache.pt"
    if not cache_path.exists():
        raise SystemExit(f"No cache for {game!r} at {cache_path}.\n")
    ds = CachedBallDataset(cache_path)
    return ds


@torch.no_grad()
def predict(model: torch.nn.Module, ds: CachedBallDataset, device: str) -> torch.Tensor:
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))
    preds = []
    for tokens, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            pred = model(tokens.to(device, non_blocking=True))
        preds.append(pred.float().cpu())
    return torch.cat(preds)


def write_predictions_csv(path: Path, frames: torch.Tensor, pred_norm: torch.Tensor,
                          wh: Sequence[float], errs_px: torch.Tensor | None) -> None:
    pred_px = denormalize(pred_norm, wh)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x_norm", "y_norm", "x_px", "y_px", "err_px"])
        for i in range(frames.shape[0]):
            err = f"{errs_px[i]:.2f}" if errs_px is not None else ""
            w.writerow([int(frames[i]), f"{pred_norm[i, 0]:.6f}",
                        f"{pred_norm[i, 1]:.6f}", f"{pred_px[i, 0]:.2f}",
                        f"{pred_px[i, 1]:.2f}", err])


def format_row(name: str, s: dict[str, float], thresholds: Sequence[float]) -> str:
    cells = [f"{name:<12s}", f"{s['n']:>7d}", f"{s['mean_px']:>9.1f}",
             f"{s['median_px']:>9.1f}", f"{s['p90_px']:>9.1f}"]
    cells += [f"{s[f'pce@{t:g}']:>8.3f}" for t in thresholds]
    return "  ".join(cells)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = BallHead().to(device).eval()
    model.load_state_dict(ckpt["model"])
    out_dir = args.out if args.out is not None else args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt: {args.ckpt} (epoch {ckpt.get('epoch')})  device: {device}")

    per_game: dict[str, dict[str, float]] = {}
    all_errs = []
    for game in args.games:
        ds = load_dataset(game, args.cache_root)
        wh = ds.meta.get("orig_wh", ORIG_WH)
        keep = torch.as_tensor(ds.idx, dtype=torch.long)
        pred = predict(model, ds, device)
        assert len(pred) == len(keep), f"{game}: {len(pred)} preds vs {len(keep)} rows"
        errs = pixel_errors(pred, ds.ball[keep].float(), wh)
        per_game[game] = summarize(errs, args.thresholds)
        all_errs.append(errs)
        csv_path = out_dir / f"{game}_predictions.csv"
        write_predictions_csv(csv_path, ds.frames[keep], pred, wh, errs)  # <- was ds.frames
        print(f"wrote {csv_path}  ({len(keep)} rows)")

    overall = summarize(torch.cat(all_errs), args.thresholds)

    ds_cache: dict[str, CachedBallDataset] = {}

    def filtered_ball(g: str) -> torch.Tensor:
        ds = ds_cache.setdefault(g, load_dataset(g, args.cache_root))
        return ds.ball[torch.as_tensor(ds.idx, dtype=torch.long)].float()

    train_games = ckpt["config"]["train_games"]
    train_ball = torch.cat([filtered_ball(g) for g in train_games])
    eval_ball = torch.cat([filtered_ball(g) for g in args.games])
    baseline = mean_predictor_baseline(train_ball, eval_ball, thresholds=args.thresholds)

    header = ["game".ljust(12), "n".rjust(7), "mean_px".rjust(9),
              "median_px".rjust(9), "p90_px".rjust(9)]
    header += [f"PCE@{t:g}".rjust(8) for t in args.thresholds]
    print("\n" + "  ".join(header))
    for game, s in per_game.items():
        print(format_row(game, s, args.thresholds))
    if len(args.games) > 1:
        print(format_row("overall", overall, args.thresholds))
    if baseline is not None:
        print(format_row("mean-pred", baseline, args.thresholds)
              + f"   <- constant baseline (train mean of {' '.join(train_games)})")


if __name__ == "__main__":
    main()
