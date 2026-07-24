import torch
from torch.utils.data import DataLoader

CLASS_NAMES = ["None", "Left", "Right"]


@torch.no_grad()
def run_autoregressive_eval(model, datasets, device, verbose=True, use_zone_filter=True):
    model.eval()
    cm = torch.zeros(3, 3, dtype=torch.int64)  # rows = true, cols = pred

    for ds in datasets:
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        hidden, score = None, None

        for batch in loader:
            if bool(batch["is_run_start"].item()):
                hidden, score = None, None

            feats = batch["features"].to(device)
            event_type = batch["event_type"].to(device)
            frame_numbers = batch["frame_numbers"][0].tolist()
            score_in = score if score is not None else batch["score_state"].to(device)

            type_logits, hidden, score = model(feats, score_in, hidden=hidden)
            pred_cls = torch.argmax(type_logits, dim=-1)[0].cpu()
            true_cls = event_type[0].cpu()

            if use_zone_filter:
                keep = []
                for i, f in enumerate(frame_numbers):
                    x, y, _ = ds.coords_dict.get(f, [0.5, 0.0, 0.0])[:3]
                    keep.append(0.15 < x < 0.85 and 0.15 < y < 0.85)
                keep = torch.tensor(keep, dtype=torch.bool)
                pred_cls, true_cls = pred_cls[keep], true_cls[keep]

            idx = true_cls * 3 + pred_cls
            cm += torch.bincount(idx, minlength=9).reshape(3, 3)

    per_class = {}
    for c in range(3):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[CLASS_NAMES[c]] = {"precision": precision, "recall": recall, "f1": f1, "support": cm[c, :].sum().item()}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / 3

    if verbose:
        print("\nConfusion matrix (rows=true, cols=pred, zone-filtered):" if use_zone_filter
              else "\nConfusion matrix (rows=true, cols=pred):")
        header = "".join(f"{n:>10}" for n in CLASS_NAMES)
        print(f"{'':>10}{header}")
        for i, name in enumerate(CLASS_NAMES):
            row = "".join(f"{cm[i,j].item():>10}" for j in range(3))
            print(f"{name:>10}{row}")
        print(f"\n{'class':>8} {'precision':>10} {'recall':>8} {'f1':>7} {'support':>8}")
        for name, m in per_class.items():
            print(f"{name:>8} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>7.3f} {m['support']:>8}")
        print(f"Macro F1: {macro_f1:.3f}")

    return cm, per_class, macro_f1