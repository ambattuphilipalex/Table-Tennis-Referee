import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_stage2 import evaluate_stage2, append_result, print_report
from score_dataset import SequentialClsDataset
from score_head import ScorePredictorClsSequential
from score_dataset_regression import SequentialScoreDataset
from score_head_regression import ScorePredictorSequential
from score_eval_utils import load_gt_events

BASELINE_RUN = "runs/20260704-1040_baseline"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs_stage2"
_DS_CACHE = {}
GRADE_WITH_ZONE_FILTER = True

PLOT = {"surface": "#fcfcfb", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "muted": "#898781", "ink": "#0b0b0b", "secondary": "#52514e",
        "blue": "#2a78d6", "aqua": "#1baf7a", "amber": "#c8811a"}

torch.set_num_threads(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def csv_for(game):
    return f"{BASELINE_RUN}/{game}_predictions.csv"


def annot_dir(game):
    return "test" if game.startswith("test") else "train"


def build_dataset(game, cfg):
    key = (game, cfg["arm"])
    if key in _DS_CACHE:
        return _DS_CACHE[key]
    cache_path = f"data/dino_cache/{game}/cache.pt"
    event_json = f"data/OpenTT/annotations/{annot_dir(game)}/{game}.json"
    csv_path = csv_for(game)
    print(f"Loading {game} ...")
    if cfg["arm"] == "cls":
        ds = SequentialClsDataset(
            cache_path, event_json, csv_path, chunk_len=256, max_frames_ahead=30)
    else:
        ds = SequentialScoreDataset(
            cache_path, event_json, csv_path, chunk_len=256,
            rally_window=200, far_background_weight=0.2)
    ds.game = game
    _DS_CACHE[key] = ds
    return ds


def make_model(feature_dim, cfg, device):
    # cls -> bidirectional nn.GRU (the only surviving variant).
    # reg -> unidirectional GRUCell, unchanged: ScorePredictorSequential has no
    # bidirectional option and must not be given one.
    if cfg["arm"] == "cls":
        return ScorePredictorClsSequential(feature_dim=feature_dim).to(device)
    return ScorePredictorSequential(feature_dim=feature_dim + 3).to(device)


def forward(model, cfg, feats, score_in, hidden):
    """Uniform call signature across the two arms."""
    if cfg["arm"] == "reg":
        _fu, logits, hidden, score = model(feats, score_in, hidden=hidden,
                                           teacher_event_type=None)
        return logits, hidden, score
    logits, hidden, score = model(feats, score_in, hidden=hidden)
    return logits, hidden, score


def run_epoch(model, datasets, optimizer, ce_loss, device, cfg, train):
    model.train() if train else model.eval()
    total_loss, n_chunks = 0.0, 0

    for ds in datasets:
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        hidden, score = None, None

        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None

            feats = batch["features"].to(device)
            event_type = batch["event_type"].to(device)
            score_in = score if score is not None else batch["score_state"].to(device)

            with torch.set_grad_enabled(train):
                logits, hidden, score = forward(model, cfg, feats, score_in, hidden)

                if cfg["arm"] == "reg":
                    fw = batch["frame_weight"].to(device).reshape(-1)
                    per_frame = ce_loss(logits.reshape(-1, 3), event_type.reshape(-1))
                    loss = (per_frame * fw).sum() / fw.sum().clamp(min=1e-8)
                else:
                    loss = ce_loss(logits.reshape(-1, 3), event_type.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_chunks += 1

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def collect_per_frame(model, datasets, device, cfg):
    """-> (per_frame list of (frame, cls, conf), merged coords_dict)."""
    model.eval()
    per_frame, coords = [], {}
    for ds in datasets:
        coords.update(ds.coords_dict)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        hidden, score = None, None
        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None
            feats = batch["features"].to(device)
            frame_numbers = batch["frame_numbers"][0].tolist()
            score_in = score if score is not None else batch["score_state"].to(device)
            logits, hidden, score = forward(model, cfg, feats, score_in, hidden)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = torch.argmax(logits, dim=-1)[0]
            for i, f in enumerate(frame_numbers):
                c = pred[i].item()
                per_frame.append((int(f), int(c), float(probs[i, c])))
    return per_frame, coords


def gt_for(datasets):
    left, right = [], []
    for ds in datasets:
        l, r = load_gt_events(
            f"data/OpenTT/annotations/{annot_dir(ds.game)}/{ds.game}.json")
        left += l
        right += r
    return sorted(left), sorted(right)


def _grade(pf, coords, gt_left, gt_right, cfg, verbose=False):
    return evaluate_stage2(pf, gt_left, gt_right, coords,
                           min_confidence=cfg.get("min_conf", 0.90),
                           use_zone_filter_at_predict=GRADE_WITH_ZONE_FILTER,
                           verbose=verbose)


# ----------------------------------------------------------------- plotting
def plot_curves(history, baseline_macro, out_path):
    """history: list of dicts with epoch, train_loss, val_loss, macro_f1, event_f1."""
    if not history:
        return
    ep = [h["epoch"] for h in history]
    ev = [h for h in history if h.get("macro_f1") is not None]
    ep_v = [h["epoch"] for h in ev]
    marker = "o" if len(ep) < 3 else None

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    fig.patch.set_facecolor(PLOT["surface"])
    for ax, title, ylab in ((ax_l, "Loss", "loss"),
                            (ax_r, "Validation metrics", "score")):
        ax.set_facecolor(PLOT["surface"])
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

    ax_l.set_yscale("log")
    ax_l.plot(ep, [h["train_loss"] for h in history], color=PLOT["blue"],
              linewidth=2, marker=marker, label="train")
    if ev and ev[0].get("val_loss") is not None:
        ax_l.plot(ep_v, [h["val_loss"] for h in ev], color=PLOT["amber"],
                  linewidth=2, marker="o", label="val")
    ax_l.legend(frameon=False, fontsize=8, labelcolor=PLOT["secondary"])

    if ev:
        ax_r.plot(ep_v, [h["macro_f1"] for h in ev], color=PLOT["blue"],
                  linewidth=2, marker="o", label="frame macro F1")
        ax_r.plot(ep_v, [h["event_f1"] for h in ev], color=PLOT["aqua"],
                  linewidth=2, marker="o", label="event F1")
        if baseline_macro is not None:
            ax_r.axhline(baseline_macro, color=PLOT["muted"], linewidth=1,
                         linestyle="--")
            ax_r.annotate(f"majority baseline {baseline_macro:.2f}",
                          (ep_v[0], baseline_macro), fontsize=8,
                          color=PLOT["secondary"], xytext=(4, 4),
                          textcoords="offset points")
        ax_r.set_ylim(0, 1)
        ax_r.legend(frameon=False, fontsize=8, labelcolor=PLOT["secondary"])

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_one(cfg, seed, train_ds, eval_ds, device, seed_dir=None):
    set_seed(seed)

    type_counts = Counter()
    for ds in train_ds:
        for i in range(len(ds)):
            for c in ds[i]["event_type"].tolist():
                type_counts[c] += 1
    total = sum(type_counts.values())
    class_weights = torch.tensor([
        total / (3 * type_counts.get(c, 1)) for c in range(3)]).to(device)

    epochs = cfg.get("epochs", 60)
    val_every = cfg.get("val_every", 5)

    feature_dim = train_ds[0].feature_dim
    model = make_model(feature_dim, cfg, device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    reduction = "none" if cfg["arm"] == "reg" else "mean"
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    gt_left, gt_right = gt_for(eval_ds)
    best_sel, best_state, best_epoch = -1.0, None, -1

    history = []
    metrics_path = seed_dir / "metrics.jsonl" if seed_dir else None
    if metrics_path:
        metrics_path.write_text("")
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"type": "class_counts",
                                "counts": dict(type_counts),
                                "weights": class_weights.tolist()}) + "\n")

    sel_key = cfg.get("select_on", "frame_macro_f1_zone_filtered")

    for epoch in range(epochs):
        tr = run_epoch(model, train_ds, optimizer, ce_loss, device, cfg, train=True)
        scheduler.step()

        row = {"type": "epoch", "epoch": epoch + 1, "train_loss": tr,
               "lr": scheduler.get_last_lr()[0]}

        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            vl = run_epoch(model, eval_ds, optimizer, ce_loss, device, cfg, train=False)
            pf, coords = collect_per_frame(model, eval_ds, device, cfg)
            m = _grade(pf, coords, gt_left, gt_right, cfg)
            sel = m[sel_key]
            print(f"  ep{epoch+1:>3} loss {tr:.4f} | val {vl:.4f}"
                  f" | macroF1 {m['frame_macro_f1']:.4f}"
                  f" (zone {m['frame_macro_f1_zone_filtered']:.4f})"
                  f" | eventF1 {m['event_f1']:.3f}"
                  f" | score {m['pred_score'][0]}-{m['pred_score'][1]}"
                  f" vs {m['true_score'][0]}-{m['true_score'][1]}")

            row.update(val_loss=vl,
                       macro_f1=m["frame_macro_f1"],
                       macro_f1_zone=m["frame_macro_f1_zone_filtered"],
                       event_f1=m["event_f1"],
                       event_precision=m["event_precision"],
                       event_recall=m["event_recall"],
                       pred_score=m["pred_score"],
                       true_score=m["true_score"],
                       scoreline_abs_err=m["scoreline_abs_err_total"])

            if sel > best_sel:
                best_sel, best_epoch = sel, epoch + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "train_loss": tr,
                        "val_loss": row.get("val_loss"),
                        "macro_f1": row.get("macro_f1"),
                        "event_f1": row.get("event_f1")})
        if metrics_path:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    model.load_state_dict(best_state)
    pf, coords = collect_per_frame(model, eval_ds, device, cfg)
    metrics = _grade(pf, coords, gt_left, gt_right, cfg, verbose=True)
    metrics["best_epoch"] = best_epoch
    metrics["select_on"] = sel_key

    if seed_dir:
        plot_curves(history, metrics.get("baseline_majority_frame_macro_f1"),
                    seed_dir / "curves.png")
        (seed_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        torch.save(best_state, seed_dir / "ckpt_best.pt")
        print(f"  -> saved {seed_dir}/ (best epoch {best_epoch}, {sel_key}={best_sel:.4f})")

    return metrics, model


@torch.no_grad()
def eval_on(model, games, cfg, device, label):
    """Evaluate an already-selected model on a further set of games (E0b)."""
    ds = [build_dataset(g, cfg) for g in games]
    pf, coords = collect_per_frame(model, ds, device, cfg)
    gt_left, gt_right = gt_for(ds)
    print(f"\n########## {label}: {'+'.join(games)} ##########")
    return _grade(pf, coords, gt_left, gt_right, cfg, verbose=True)


def summarise(ms, title, out_path=None):
    print(f"\n===== {title} over {len(ms)} seeds =====")
    summary = {"n_seeds": len(ms)}
    for k in ["frame_macro_f1", "frame_macro_f1_zone_filtered", "event_f1",
              "event_precision", "event_recall", "scoreline_abs_err_total"]:
        vals = [m[k] for m in ms]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[k] = {"mean": statistics.mean(vals), "sd": sd,
                      "spread": max(vals) - min(vals), "values": vals}
        print(f"  {k:>32}: mean {statistics.mean(vals):.4f}"
              f"  sd {sd:.4f}  spread {max(vals)-min(vals):.4f}"
              f"  {[round(v, 4) for v in vals]}")
    summary["baseline_majority_frame_macro_f1"] = ms[0]["baseline_majority_frame_macro_f1"]
    summary["baseline_random_event_f1"] = statistics.mean(
        [m["baseline_random_event_f1"] for m in ms])
    print(f"  {'baseline_majority_frame_macro_f1':>32}: "
          f"{summary['baseline_majority_frame_macro_f1']:.4f}")
    print(f"  {'baseline_random_event_f1':>32}: "
          f"{summary['baseline_random_event_f1']:.4f}")
    if out_path:
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--arm", choices=["cls", "reg"], default="cls")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-games", nargs="+", default=["game_1", "game_2", "game_4", "game_5"])
    ap.add_argument("--eval-games", nargs="+", default=["game_3"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.90)
    ap.add_argument("--test-games", nargs="*", default=[])
    ap.add_argument("--select-on", default="frame_macro_f1_zone_filtered",
                    choices=["frame_macro_f1_zone_filtered", "frame_macro_f1", "event_f1"],
                    help="metric used to keep the best checkpoint")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="also write experiments/ckpt/<exp-id>_seed<N>.pth")
    ap.add_argument("--no-run-dir", action="store_true",
                    help="skip creating runs_stage2/<ts>_<exp-id>/")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = {
        "arm": args.arm, "train_games": args.train_games, "eval_games": args.eval_games,
        "epochs": args.epochs, "val_every": args.val_every, "min_conf": args.min_conf,
        "select_on": args.select_on, "note": args.note,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = None
    if not args.no_run_dir:
        run_dir = RUNS_ROOT / f"{datetime.now():%Y%m%d-%H%M}_{args.exp_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        full_cfg = {**cfg, "exp_id": args.exp_id, "seeds": args.seeds,
                    "test_games": args.test_games, "device": device,
                    "baseline_run": BASELINE_RUN,
                    "started": datetime.now().isoformat(timespec="seconds")}
        (run_dir / "config.json").write_text(json.dumps(full_cfg, indent=2) + "\n")

    print(f"=== {args.exp_id} === arm={args.arm} device={device}")
    if run_dir:
        print(f"run dir: {run_dir}")
    print(json.dumps(cfg, indent=2))

    train_ds = [build_dataset(g, cfg) for g in args.train_games]
    eval_ds = [build_dataset(g, cfg) for g in args.eval_games]

    all_m, all_t = [], []
    for seed in args.seeds:
        print(f"\n----- {args.exp_id} seed {seed} -----")
        seed_dir = None
        if run_dir:
            seed_dir = run_dir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

        metrics, model = run_one(cfg, seed, train_ds, eval_ds, device, seed_dir)
        append_result(args.exp_id, args.arm, seed, metrics, cfg,
                      game="+".join(args.eval_games))
        all_m.append(metrics)

        if args.save_ckpt:
            d = Path("experiments/ckpt")
            d.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), d / f"{args.exp_id}_seed{seed}.pth")

        if args.test_games:
            tm = eval_on(model, args.test_games, cfg, device, "HELD-OUT TEST")
            append_result(args.exp_id + "_test", args.arm, seed, tm,
                          {**cfg, "eval_games": args.test_games},
                          game="+".join(args.test_games))
            all_t.append(tm)
            if seed_dir:
                (seed_dir / "test_metrics.json").write_text(
                    json.dumps(tm, indent=2) + "\n")

    summarise(all_m, f"{args.exp_id} SUMMARY",
              run_dir / "summary_val.json" if run_dir else None)
    if all_t:
        summarise(all_t, f"{args.exp_id}_test (HELD-OUT) SUMMARY",
                  run_dir / "summary_test.json" if run_dir else None)

    if run_dir:
        print(f"\nartifacts in {run_dir}")


if __name__ == "__main__":
    main()