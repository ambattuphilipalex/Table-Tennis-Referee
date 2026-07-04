from typing import Sequence

import torch

ORIG_WH: tuple[int, int] = (1920, 1080)

DEFAULT_THRESHOLDS: tuple[float, ...] = (5, 15, 50)


def denormalize(xy_norm: torch.Tensor, wh: Sequence[float] = ORIG_WH) -> torch.Tensor:
    return xy_norm * xy_norm.new_tensor(tuple(wh))


def pixel_errors(pred_norm: torch.Tensor, gt_norm: torch.Tensor,
                 wh: Sequence[float] = ORIG_WH) -> torch.Tensor:
    diff = denormalize(pred_norm.float(), wh) - denormalize(gt_norm.float(), wh)
    return torch.linalg.vector_norm(diff, dim=-1)


def summarize(errors_px: torch.Tensor,
              thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> dict[str, float]:
    e = errors_px.detach().float().flatten()
    if e.numel() == 0:
        raise ValueError("summarize() got an empty error tensor")
    out: dict[str, float] = {
        "n": int(e.numel()),
        "mean_px": e.mean().item(),
        "median_px": e.median().item(),
        "p90_px": torch.quantile(e, 0.9).item(),
    }
    for r in thresholds:
        out[f"pce@{r:g}"] = (e <= r).float().mean().item()
    return out


def mean_predictor_baseline(train_ball: torch.Tensor, eval_ball: torch.Tensor,
                            wh: Sequence[float] = ORIG_WH,
                            thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                            ) -> dict[str, float]:
    mean_xy = train_ball.float().mean(dim=0, keepdim=True)
    pred = mean_xy.expand(eval_ball.shape[0], 2)
    return summarize(pixel_errors(pred, eval_ball.float(), wh), thresholds)


def format_summary(s: dict[str, float]) -> str:
    parts = [f"n={s['n']}", f"mean={s['mean_px']:.1f}px",
             f"median={s['median_px']:.1f}px", f"p90={s['p90_px']:.1f}px"]
    parts += [f"{k}={v:.3f}" for k, v in s.items() if k.startswith("pce@")]
    return "  ".join(parts)
