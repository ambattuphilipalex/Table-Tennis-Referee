from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from ball_head import BallHead
from cache_dataset import CACHE_ROOT, build_split
from metrics import DEFAULT_THRESHOLDS, format_summary, mean_predictor_baseline, pixel_errors, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]

PLOT = {"surface": "#fcfcfb", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "muted": "#898781", "ink": "#0b0b0b", "secondary": "#52514e",
        "blue": "#2a78d6", "aqua": "#1baf7a"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-games", nargs="+",
                   default=["game_1", "game_2", "game_3", "game_4"])
    p.add_argument("--val-games", nargs="+", default=["game_5"])
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    p.add_argument("--beta", type=float, default=0.05,
                   help="smooth_l1 beta; coords live in [0,1], so torch's "
                        "default 1.0 would never leave the quadratic region")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--tag", default="baseline", help="run-directory name suffix")
    return p.parse_args()


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name == "cls" else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """Linear warmup to 1, then cosine decay to ~0."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return f


@torch.no_grad()
def run_validation(model: torch.nn.Module, loader: DataLoader, device: str,
                   use_amp: bool) -> dict[str, float]:
    model.eval()
    errs = []
    for tokens, ball in loader:
        tokens = tokens.to(device, non_blocking=True)
        ball = ball.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(tokens)
        errs.append(pixel_errors(pred.float(), ball).cpu())
    return summarize(torch.cat(errs), DEFAULT_THRESHOLDS)


def plot_curves(history: list[dict], baseline_mean_px: float, out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    marker = "o" if len(epochs) < 2 else None
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    fig.patch.set_facecolor(PLOT["surface"])
    for ax, title, ylab in ((ax_l, "Train loss", "loss"),
                            (ax_r, "Val pixel error", "error (px)")):
        ax.set_facecolor(PLOT["surface"])
        ax.set_yscale("log")
        ax.grid(axis="y", color=PLOT["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(PLOT["axis"])
        ax.tick_params(colors=PLOT["muted"], labelsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=PLOT["ink"])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("epoch", fontsize=8, color=PLOT["muted"])
        ax.set_ylabel(ylab, fontsize=8, color=PLOT["muted"])

    loss = [h["train_loss"] for h in history]
    ax_l.plot(epochs, loss, color=PLOT["blue"], linewidth=2, marker=marker)
    ax_l.annotate(f"{loss[-1]:.2e}", (epochs[-1], loss[-1]), fontsize=8,
                  color=PLOT["secondary"], xytext=(4, 4), textcoords="offset points")

    for key, color, label in (("mean_px", PLOT["blue"], "mean"),
                              ("median_px", PLOT["aqua"], "median")):
        ys = [h["val"][key] for h in history]
        ax_r.plot(epochs, ys, color=color, linewidth=2, marker=marker, label=label)
        ax_r.annotate(f"{label} {ys[-1]:.1f}", (epochs[-1], ys[-1]), fontsize=8,
                      color=PLOT["secondary"], xytext=(4, 4), textcoords="offset points")
    ax_r.axhline(baseline_mean_px, color=PLOT["muted"], linewidth=1, linestyle="--")
    ax_r.annotate(f"mean-predictor {baseline_mean_px:.0f}px",
                  (epochs[0], baseline_mean_px), fontsize=8,
                  color=PLOT["secondary"], xytext=(4, 4), textcoords="offset points")
    ax_r.legend(frameon=False, fontsize=8, labelcolor=PLOT["secondary"])
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def gather_ball(split: ConcatDataset) -> torch.Tensor:
    return torch.cat([d.ball.float() for d in split.datasets])


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    run_dir = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(device=device, amp=use_amp, run_dir=str(run_dir))
    print("config:", json.dumps(config, indent=2))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    log_path = run_dir / "metrics.jsonl"

    def log_jsonl(obj: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    train_ds = build_split(args.train_games, root=args.cache_root)
    val_ds = build_split(args.val_games, root=args.cache_root)
    print(f"train: {len(train_ds)} frames from {args.train_games}")
    print(f"val:   {len(val_ds)} frames from {args.val_games}")

    train_ball, val_ball = gather_ball(train_ds), gather_ball(val_ds)

    baseline = mean_predictor_baseline(train_ball, val_ball)
    print("mean-predictor baseline on val:", format_summary(baseline))
    log_jsonl({"type": "baseline", "val": baseline})

    workers = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=workers,
                              pin_memory=(device == "cuda"),
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=workers, pin_memory=(device == "cuda"),
                            persistent_workers=workers > 0)

    model = BallHead().to(device)
    print(f"BallHead: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(args.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    def loss_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if args.loss == "mse":
            return F.mse_loss(pred, gt)
        return F.smooth_l1_loss(pred, gt, beta=args.beta)

    best_mean_px = math.inf
    global_step = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        for tokens, ball in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            ball = ball.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(tokens), ball)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            loss_sum += loss.item() * ball.shape[0]
            n_seen += ball.shape[0]
            if global_step % 50 == 0:
                log_jsonl({"type": "train", "step": global_step, "epoch": epoch,
                           "loss": loss.item(), "lr": sched.get_last_lr()[0]})

        epoch_time = time.time() - t0
        train_loss = loss_sum / max(1, n_seen)
        samples_per_s = n_seen / epoch_time
        val = run_validation(model, val_loader, device, use_amp)

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val})
        log_jsonl({"type": "epoch", "epoch": epoch, "train_loss": train_loss,
                   "epoch_time_s": round(epoch_time, 1),
                   "samples_per_s": round(samples_per_s, 1),
                   "lr": sched.get_last_lr()[0], "val": val})
        plot_curves(history, baseline["mean_px"], run_dir / "curves.png")

        print(f"epoch {epoch:3d}/{args.epochs}  loss={train_loss:.5f}  "
              f"{format_summary(val)}  [{epoch_time:.0f}s, {samples_per_s:.0f} samples/s]")

        ckpt = {"model": model.state_dict(), "config": config,
                "epoch": epoch, "val_metrics": val}
        torch.save(ckpt, run_dir / "ckpt_last.pt")
        if val["mean_px"] < best_mean_px:
            best_mean_px = val["mean_px"]
            torch.save(ckpt, run_dir / "ckpt_best.pt")
            print(f"new best: val mean {best_mean_px:.1f}px -> ckpt_best.pt")

    print(f"done. best val mean {best_mean_px:.1f}px "
          f"(baseline {baseline['mean_px']:.1f}px). artifacts in {run_dir}")


if __name__ == "__main__":
    main()
