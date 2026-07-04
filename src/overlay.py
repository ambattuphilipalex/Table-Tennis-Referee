import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ball_head import BallHead
from cache_dataset import CACHE_ROOT, DATA_ROOT, CachedBallDataset
from metrics import ORIG_WH, denormalize, pixel_errors

GT_COLOR = (0, 0, 255)
PRED_COLOR = (0, 255, 255)
LINE_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--game", default="game_5")
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("overlay.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--worst-k", type=int, default=24, help="save the k highest-error cached frames as PNGs")
    return p.parse_args()


def read_frame_bgr(video_path: Path, frame_no: int) -> np.ndarray:
    cur = cv2.VideoCapture(str(video_path))
    cur.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    flag, frame = cur.read()
    cur.release()
    if flag is False:
        raise ValueError(f"couldn't read frame {frame_no} from {video_path}")
    return frame


def annotate(frame: np.ndarray, gt_px: np.ndarray, pred_px: np.ndarray, frame_no: int, err_px: float) -> np.ndarray:
    g = (int(round(gt_px[0])), int(round(gt_px[1])))
    q = (int(round(pred_px[0])), int(round(pred_px[1])))
    cv2.line(frame, g, q, LINE_COLOR, 1, cv2.LINE_AA)
    cv2.circle(frame, g, 10, GT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(frame, q, 10, PRED_COLOR, 2, cv2.LINE_AA)
    text = f"f={frame_no}  err={err_px:.1f}px"
    cv2.putText(frame, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


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


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = args.cache_root / args.game / "cache.pt"
    if not cache_path.exists():
        raise SystemExit(f"No cache for {args.game!r} at {cache_path}.")
    ds = CachedBallDataset(cache_path)
    wh = ds.meta.get("orig_wh", ORIG_WH)
    split = ds.meta.get("split", "train" if args.game.startswith("game") else "test")
    video_path = DATA_ROOT / "OpenTT" / "videos" / split / f"{args.game}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Video not found at {video_path}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = BallHead().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"ckpt: {args.ckpt} (epoch {ckpt.get('epoch')})  device: {device}")

    pred_norm = predict(model, ds, device)
    gt_norm = ds.ball.float()
    errs = pixel_errors(pred_norm, gt_norm, wh)
    pred_px = denormalize(pred_norm, wh).numpy()
    gt_px = denormalize(gt_norm, wh).numpy()
    frames = ds.frames.numpy()
    print(f"{args.game}: {len(ds)} cached frames, "
          f"mean err {errs.mean():.1f}px over the full game")

    end = args.end_frame if args.end_frame is not None else int(frames.max())
    idx = np.where((frames >= args.start_frame) & (frames <= end))[0]
    idx = idx[np.argsort(frames[idx])]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, tuple(wh))
    for n, i in enumerate(idx, 1):
        frame = read_frame_bgr(video_path, frames[i])
        writer.write(annotate(frame, gt_px[i], pred_px[i], int(frames[i]),
                              float(errs[i])))
        if n % 50 == 0 or n == idx.size:
            print(f"  clip: {n}/{idx.size} frames")
    writer.release()
    clip = errs[torch.from_numpy(idx)]
    print(f"wrote {args.out}: {idx.size} labeled frames from source range "
          f"[{frames[idx[0]]}, {frames[idx[-1]]}], {idx.size / args.fps:.1f}s at "
          f"{args.fps} fps ({120 / args.fps:.0f}x slow motion vs the 120 fps "
          f"source), mean err {clip.mean():.1f}px")

    k = min(args.worst_k, len(ds))
    if k <= 0:
        return
    worst = torch.topk(errs, k).indices.numpy()
    for rank, i in enumerate(worst, 1):
        frame = read_frame_bgr(video_path, frames[i])
        annotate(frame, gt_px[i], pred_px[i], int(frames[i]), float(errs[i]))
        png = args.out.parent / (f"worst_{rank:02d}_{args.game}_f{frames[i]}"
                                 f"_err{errs[i]:.0f}px.png")
        cv2.imwrite(str(png), frame)
    print(f"wrote {k} worst-frame PNGs to {args.out.parent} "
          f"(errors {errs[worst[-1]]:.0f}..{errs[worst[0]]:.0f}px)")


if __name__ == "__main__":
    main()
