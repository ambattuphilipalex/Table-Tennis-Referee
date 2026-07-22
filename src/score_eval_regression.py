import torch
from torch.utils.data import DataLoader

CLASS_NAMES = ["None", "Left", "Right"]


@torch.no_grad()
def print_frames_near_events(model, dataset, device, event_frames, window=60):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None
    all_rows = []  # (frame, true_cls, pred_cls, confidence)

    for batch in loader:
        if bool(batch["is_run_start"].item()):
            hidden, score = None, None

        feats = batch["features"].to(device)
        event_type = batch["event_type"].to(device)
        frame_numbers = batch["frame_numbers"][0].tolist()
        score_in = score if score is not None else batch["score_state"].to(device)

        _, type_logits, hidden, score = model(feats, score_in, hidden=hidden, teacher_event_type=None)
        probs = torch.softmax(type_logits, dim=-1)[0]
        pred_cls = torch.argmax(type_logits, dim=-1)[0].tolist()
        true_cls = event_type[0].tolist()

        for i, f in enumerate(frame_numbers):
            tc, pc = true_cls[i], pred_cls[i]
            conf = probs[i, pc].item()
            all_rows.append((f, tc, pc, conf))

    frame_to_row = {r[0]: r for r in all_rows}
    print(f"\n{'frame':>8} {'true':>6} {'pred':>6} {'conf':>6}   (near real events)")
    for ef in sorted(event_frames):
        print(f"--- ground-truth event near frame {ef} ---")
        for f in range(ef - window, ef + window):
            if f in frame_to_row:
                _, tc, pc, conf = frame_to_row[f]
                marker = " <-- EVENT" if f == ef else ""
                print(f"{f:>8} {CLASS_NAMES[tc]:>6} {CLASS_NAMES[pc]:>6} {conf:>6.3f}{marker}")



@torch.no_grad()
def print_frame_samples(model, dataset, device, num_frames=40):
    """Prints a per-timestamp table for the first `num_frames` of the eval set."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    hidden, score = None, None
    printed = 0

    print(f"\n{'frame':>8} {'true':>6} {'pred':>6} {'true_fu':>8} {'pred_fu':>8}")
    for batch in loader:
        if bool(batch["is_run_start"].item()):
            hidden, score = None, None

        feats = batch["features"].to(device)
        event_type = batch["event_type"].to(device)
        frames_until = batch["frames_until"].to(device)
        frame_numbers = batch["frame_numbers"][0].tolist()
        score_in = score if score is not None else batch["score_state"].to(device)

        frames_pred, type_logits, hidden, score = model(
            feats, score_in, hidden=hidden, teacher_event_type=None
        )
        pred_cls = torch.argmax(type_logits, dim=-1)[0].tolist()
        true_cls = event_type[0].tolist()
        pred_fu = frames_pred[0, :, 0].tolist()
        true_fu = frames_until[0].tolist()

        for f, tc, pc, tfu, pfu in zip(frame_numbers, true_cls, pred_cls, true_fu, pred_fu):
            print(f"{f:>8} {CLASS_NAMES[tc]:>6} {CLASS_NAMES[pc]:>6} {tfu:>8.3f} {pfu:>8.3f}")
            printed += 1
            if printed >= num_frames:
                return
    print("(reached end of eval set before num_frames)")


@torch.no_grad()
def run_autoregressive_eval(model, datasets, device, verbose=True):
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
            score_in = score if score is not None else batch["score_state"].to(device)

            _, type_logits, hidden, score = model(
                feats, score_in, hidden=hidden, teacher_event_type=None
            )
            pred_cls = torch.argmax(type_logits, dim=-1).reshape(-1).cpu()
            true_cls = event_type.reshape(-1).cpu()

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
        print("\nConfusion matrix (rows=true, cols=pred):")
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