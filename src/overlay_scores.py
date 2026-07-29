

import argparse
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_dataset_regression import SequentialScoreDataset
from score_head_regression import ScorePredictorSequential
from score_eval_utils import load_gt_events

BALL_RADIUS = 12
BALL_COLOR = (0, 255, 255)      # yellow  (BGR)
PRED_COLOR = (0, 200, 255)      # orange
TRUE_COLOR = (0, 255, 0)        # green
HOLD_FRAMES = 60                
CLUSTER_GAP = 120               # firings this close are the same point
CONFLICT_GAP = 300              # left/right disagreements this close -> keep best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="test_6",
                   help="game_1..game_5 or test_1..test_7")
    p.add_argument("--arm", choices=["reg", "cls"], default="reg",
                   help="which trained model to use")
    p.add_argument("--model", default="score_predictor_sequential.pth")
    p.add_argument("--cache-root", default="data/dino_cache")
    p.add_argument("--runs-dir", default="runs/20260704-1040_baseline",
                   help="folder holding <game>_predictions.csv")
    p.add_argument("--out", default=None, help="output mp4 (default annotated_<game>.mp4)")
    p.add_argument("--min-conf", type=float, default=0.65)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--shift-by-countdown", action="store_true",
                   help="reg arm only: move each firing forward by its predicted "
                        "countdown. OFF by default because the countdown head is "
                        "not trained in the current run_experiment.py loop.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after N frames (quick preview)")
    return p.parse_args()


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

    dataset = SequentialScoreDataset(cache_path, event_json, csv_path,
                                     chunk_len=args.chunk_len)

    if args.arm == "cls":
        from score_head import ScorePredictorClsSequential
        model = ScorePredictorClsSequential(feature_dim=dataset.feature_dim).to(device)
    else:
        model = ScorePredictorSequential(feature_dim=dataset.feature_dim + 3).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

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
    print(f"detections: {len(dets)}   predicted {pl}-{pr}   true {len(gt_left)}-{len(gt_right)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {w}x{h} @ {fps:.1f}fps, {total} frames -> {out_path}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    left, right = 0, 0
    gl, gr = 0, 0
    di = 0                      
    li, ri = 0, 0               
    frame_no = 0

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

        coords = dataset.coords_dict.get(frame_no)
        if coords is not None:
            x_norm, y_norm, fresh = coords
            if fresh > 0.5:
                px, py = int(x_norm * w), int(y_norm * h)
                cv2.circle(frame, (px, py), BALL_RADIUS, BALL_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), BALL_RADIUS + 2, (0, 0, 0), 2, cv2.LINE_AA)

        if frame_no in per_frame_pred:
            cls, conf = per_frame_pred[frame_no]
            if cls != 0:
                label = "LEFT" if cls == 1 else "RIGHT"
                cv2.putText(frame, f"frame pred: {label} ({conf:.2f})",
                            (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (210, 210, 210), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_no += 1
        if frame_no % 5000 == 0:
            print(f"  {frame_no}/{total}")

    cap.release()
    writer.release()
    print(f"\ndone -> {out_path}")
    print(f"final: predicted {left}-{right}   true {gl}-{gr}")


if __name__ == "__main__":
    main()