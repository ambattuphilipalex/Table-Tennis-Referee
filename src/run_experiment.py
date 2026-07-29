import argparse
import json
import random
import statistics
import sys
from pathlib import Path

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
_DS_CACHE = {}
GRADE_WITH_ZONE_FILTER = True

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


def run_one(cfg, seed, train_ds, eval_ds, device):
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
    best_sel, best_state = -1.0, None

    for epoch in range(epochs):
        tr = run_epoch(model, train_ds, optimizer, ce_loss, device, cfg, train=True)
        scheduler.step()
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            pf, coords = collect_per_frame(model, eval_ds, device, cfg)
            m = _grade(pf, coords, gt_left, gt_right, cfg)
            sel = m["frame_macro_f1_zone_filtered"]
            print(f"  ep{epoch+1:>3} loss {tr:.4f} | macroF1 {m['frame_macro_f1']:.4f}"
                  f" (zone {m['frame_macro_f1_zone_filtered']:.4f})"
                  f" | eventF1 {m['event_f1']:.3f}"
                  f" | score {m['pred_score'][0]}-{m['pred_score'][1]}"
                  f" vs {m['true_score'][0]}-{m['true_score'][1]}")
            if sel > best_sel:
                best_sel = sel
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pf, coords = collect_per_frame(model, eval_ds, device, cfg)
    metrics = _grade(pf, coords, gt_left, gt_right, cfg, verbose=True)
    return metrics, model


@torch.no_grad()
def eval_on(model, games, cfg, device, label):
    """Evaluate an already-selected model on a further set of games (E0b)."""
    ds = [build_dataset(g, cfg) for g in games]
    pf, coords = collect_per_frame(model, ds, device, cfg)
    gt_left, gt_right = gt_for(ds)
    print(f"\n########## {label}: {'+'.join(games)} ##########")
    return _grade(pf, coords, gt_left, gt_right, cfg, verbose=True)


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
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = {
        "arm": args.arm, "train_games": args.train_games, "eval_games": args.eval_games,
        "epochs": args.epochs, "val_every": args.val_every, "min_conf": args.min_conf,
        "note": args.note,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== {args.exp_id} === arm={args.arm} device={device}")
    print(json.dumps(cfg, indent=2))

    train_ds = [build_dataset(g, cfg) for g in args.train_games]
    eval_ds = [build_dataset(g, cfg) for g in args.eval_games]

    all_m, all_t = [], []
    for seed in args.seeds:
        print(f"\n----- {args.exp_id} seed {seed} -----")
        metrics, model = run_one(cfg, seed, train_ds, eval_ds, device)
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

    def summarise(ms, title):
        print(f"\n===== {title} over {len(ms)} seeds =====")
        for k in ["frame_macro_f1", "frame_macro_f1_zone_filtered", "event_f1",
                  "event_precision", "event_recall", "scoreline_abs_err_total"]:
            vals = [m[k] for m in ms]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {k:>32}: mean {statistics.mean(vals):.4f}"
                  f"  sd {sd:.4f}  spread {max(vals)-min(vals):.4f}"
                  f"  {[round(v, 4) for v in vals]}")
        print(f"  {'baseline_majority_frame_macro_f1':>32}: "
              f"{ms[0]['baseline_majority_frame_macro_f1']:.4f}")
        print(f"  {'baseline_random_event_f1':>32}: "
              f"{statistics.mean([m['baseline_random_event_f1'] for m in ms]):.4f}")

    summarise(all_m, f"{args.exp_id} SUMMARY")
    if all_t:
        summarise(all_t, f"{args.exp_id}_test (HELD-OUT) SUMMARY")


if __name__ == "__main__":
    main()
