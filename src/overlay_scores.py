"""
overlay_scores.py -- draw predicted vs ground-truth scoreline AND predicted vs
ground-truth ball position over a game or test video.

  (b) step-wise score prediction  -> running PRED score
  (c) step-wise score + 2D ball   -> yellow box = predicted ball
  (d) GT vs predicted overlay     -> PRED/TRUE scorelines, red box = true ball

Run from the repo root:
    python src/overlay_scores.py --game game_3 --max-frames 3000
    python src/overlay_scores.py --game test_6
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_eval_utils import load_gt_events

BOX_HALF = 22
BALL_COLOR = (0, 255, 255)      # yellow  - predicted ball
GT_COLOR = (0, 0, 255)          # red     - true ball  (matches Stage-1 overlay.py)
PRED_COLOR = (0, 200, 255)      # orange  - predicted score
TRUE_COLOR = (0, 255, 0)        # green   - true score
LINE_COLOR = (255, 255, 255)
HOLD_FRAMES = 60
CLUSTER_GAP = 120               
CONFLICT_GAP = 300              


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="game_3", help="game_1..game_5 or test_1..test_7")
    p.add_argument("--arm", choices=["reg", "cls"], default="cls")
    p.add_argument("--model", default="score_predictor_cls_sequential.pth")
    p.add_argument("--cache-root", default="data/dino_cache")
    p.add_argument("--runs-dir", default="runs/20260704-1040_baseline")
    p.add_argument("--out", default=None, help="output mp4 (default annotated_<game>.mp4)")
    p.add_argument("--min-conf", type=float, default=0.65)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--fps", type=float, default=None,
                   help="output fps; lower = slow motion (source is ~120)")
    p.add_argument("--only-labeled", action="store_true",
                   help="write only frames that have a ball prediction (short clip, "
                        "box never disappears, but the footage jumps in time)")
    p.add_argument("--shift-by-countdown", action="store_true",
                   help="reg arm only: shift each firing forward by its predicted "
                        "countdown. OFF by default -- that head is not trained in "
                        "the current run_experiment.py loop.")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    return p.parse_args()


def load_gt_ball(split, game):
    """frame_no -> (x_px, y_px) from the original annotations. Skips invalid (-1)."""
    path = Path(f"data/OpenTT/annotations/{split}/{game}_ball.json")
    if not path.exists():
        print(f"  (no ground-truth ball file at {path} -- red box disabled)")
        return {}
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if v["x"] == -1 or v["y"] == -1:
            continue
        out[int(k)] = (int(v["x"]), int(v["y"]))
    return out


def build(args, cache_path, event_json, csv_path, device):
    """Return (dataset, model) for the chosen arm. Built ONCE, before the loop."""
    if args.arm == "cls":
        from score_dataset import SequentialClsDataset
        from score_head import ScorePredictorClsSequential
        ds = SequentialClsDataset(cache_path, event_json, csv_path,
                                  chunk_len=args.chunk_len, max_frames_ahead=30)
        model = ScorePredictorClsSequential(feature_dim=ds.feature_dim).to(device)
    else:
        from score_dataset_regression import SequentialScoreDataset
        from score_head_regression import ScorePredictorSequential
        ds = SequentialScoreDataset(cache_path, event_json, csv_path,
                                    chunk_len=args.chunk_len)
        model = ScorePredictorSequential(feature_dim=ds.feature_dim + 3).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    return ds, model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = "test" if args.game.startswith("test") else "train"
    cache_path = f"{args.cache_root}/{args.game}/cache.pt"
    event_json = f"data/OpenTT/annotations/{split}/{args.game}.json"
    csv_path = f"{args.runs_dir}/{args.game}_predictions.csv"
    video_path = f"data/OpenTT/videos/{split}/{args.game}.mp4"
    out_path = args.out or f"annotated_{args.game}.mp4"

    for f in (cache_path, event_json, csv_path, video_path):
        if not Path(f).exists():
            raise SystemExit(f"missing input: {f}")

    print(f"game={args.game}  arm={args.arm}  device={device}")

    gt_left, gt_right = load_gt_events(event_json)
    gt_left, gt_right = sorted(gt_left), sorted(gt_right)
    print(f"ground truth: {len(gt_left)} left, {len(gt_right)} right")

    gt_ball = load_gt_ball(split, args.game)
    print(f"ground-truth ball positions: {len(gt_ball)} frames")

    dataset, model = build(args, cache_path, event_json, csv_path, device)

    print("running model ...")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None
    raw, per_frame_pred = [], {}

    with torch.no_grad():
        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None

            feats = batch["features"].to(device)
            frame_numbers = batch["frame_numbers"][0].tolist()
            score_in = score if score is not None else batch["score_state"].to(device)

            if args.arm == "cls":
                logits, hidden, score = model(feats, score_in, hidden=hidden)
                countdown = None
            else:
                fu_pred, logits, hidden, score = model(
                    feats, score_in, hidden=hidden, teacher_event_type=None)
                countdown = fu_pred[0, :, 0] * dataset.max_frames_ahead

            probs = torch.softmax(logits, dim=-1)[0]
            pred_cls = torch.argmax(logits, dim=-1)[0]

            for i, real_frame in enumerate(frame_numbers):
                cls = int(pred_cls[i].item())
                conf = float(probs[i, cls].item())
                per_frame_pred[int(real_frame)] = (cls, conf)

                if cls != 0 and conf >= args.min_conf:
                    frame_out = int(real_frame)
                    if args.shift_by_countdown and countdown is not None:
                        fu = float(countdown[i].item())
                        if 0 <= fu <= dataset.max_frames_ahead:
                            frame_out += int(max(0, fu))
                    raw.append({"frame": frame_out, "cls": cls, "conf": conf})

    print(f"raw firings: {len(raw)}")

    raw.sort(key=lambda x: x["frame"])
    dets = []
    for cls in (1, 2):
        same = [p for p in raw if p["cls"] == cls]
        if not same:
            continue
        cluster = [same[0]]
        for p in same[1:]:
            if p["frame"] - cluster[-1]["frame"] <= CLUSTER_GAP:
                cluster.append(p)
            else:
                dets.append(max(cluster, key=lambda x: x["conf"]))
                cluster = [p]
        dets.append(max(cluster, key=lambda x: x["conf"]))

    dets.sort(key=lambda x: x["frame"])
    kept, i = [], 0
    while i < len(dets):
        current, j = dets[i], i + 1
        while j < len(dets) and dets[j]["frame"] - current["frame"] < CONFLICT_GAP:
            if dets[j]["cls"] != current["cls"] and dets[j]["conf"] > current["conf"]:
                current = dets[j]
            j += 1
        kept.append(current)
        i = j
    dets = sorted(kept, key=lambda x: x["frame"])

    pl = sum(1 for d in dets if d["cls"] == 1)
    pr = sum(1 for d in dets if d["cls"] == 2)
    print(f"detections: {len(dets)}   predicted {pl}-{pr}   "
          f"true {len(gt_left)}-{len(gt_right)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps or src_fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {w}x{h} @ {src_fps:.1f}fps -> writing at {fps:.1f}fps, "
          f"{total} frames -> {out_path}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    left = right = gl = gr = 0
    di = li = ri = 0
    frame_no = written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_no >= args.max_frames:
            break

        while di < len(dets) and dets[di]["frame"] <= frame_no:
            if dets[di]["cls"] == 1:
                left += 1
            else:
                right += 1
            di += 1

        while li < len(gt_left) and gt_left[li] <= frame_no:
            gl += 1
            li += 1
        while ri < len(gt_right) and gt_right[ri] <= frame_no:
            gr += 1
            ri += 1

        cv2.putText(frame, f"PRED  {left} - {right}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, PRED_COLOR, 3, cv2.LINE_AA)
        cv2.putText(frame, f"TRUE  {gl} - {gr}", (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, TRUE_COLOR, 3, cv2.LINE_AA)

        for d in dets:
            if d["frame"] <= frame_no < d["frame"] + HOLD_FRAMES:
                label = "LEFT" if d["cls"] == 1 else "RIGHT"
                cv2.putText(frame, f"{label} POINT  ({d['conf']:.2f})", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, PRED_COLOR, 3, cv2.LINE_AA)
                break

        # --- ground-truth ball (red) ---
        gt_xy = gt_ball.get(frame_no)
        if gt_xy is not None:
            gx, gy = gt_xy
            cv2.rectangle(frame, (gx - BOX_HALF, gy - BOX_HALF),
                          (gx + BOX_HALF, gy + BOX_HALF), GT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, "true", (gx - BOX_HALF, gy + BOX_HALF + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)

        # --- predicted ball (yellow) ---
        coords = dataset.coords_dict.get(frame_no)
        px = py = None
        if coords is not None and coords[2] > 0.5:
            px, py = int(coords[0] * w), int(coords[1] * h)
            cv2.rectangle(frame, (px - BOX_HALF, py - BOX_HALF),
                          (px + BOX_HALF, py + BOX_HALF), BALL_COLOR, 2, cv2.LINE_AA)
            cv2.line(frame, (px - 4, py), (px + 4, py), BALL_COLOR, 1, cv2.LINE_AA)
            cv2.line(frame, (px, py - 4), (px, py + 4), BALL_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, "pred", (px - BOX_HALF, py - BOX_HALF - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1, cv2.LINE_AA)

        if gt_xy is not None and px is not None:
            cv2.line(frame, (gx, gy), (px, py), LINE_COLOR, 1, cv2.LINE_AA)
            err = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            cv2.putText(frame, f"{err:.0f}px", (px + BOX_HALF + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 1, cv2.LINE_AA)

        if frame_no in per_frame_pred:
            cls, conf = per_frame_pred[frame_no]
            if cls != 0:
                label = "LEFT" if cls == 1 else "RIGHT"
                cv2.putText(frame, f"frame pred: {label} ({conf:.2f})",
                            (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (210, 210, 210), 2, cv2.LINE_AA)

        if not args.only_labeled or px is not None:
            writer.write(frame)
            written += 1

        frame_no += 1
        if frame_no % 5000 == 0:
            print(f"  {frame_no}/{total}  (written {written})")

    cap.release()
    writer.release()
    print(f"\ndone -> {out_path}   ({written} frames written)")
    print(f"final: predicted {left}-{right}   true {gl}-{gr}")


if __name__ == "__main__":
    main()