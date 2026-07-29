import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MERGE_WINDOW = 90        # frames: cluster same-class raw firings closer than this
CROSS_SUPPRESS = 180     # frames: window in which a stronger opposite-class detection wins
MATCH_LO = -30           # a detection at f matches gt at g if MATCH_LO <= (g - f) <= MATCH_HI
MATCH_HI = 150
ZONE_LO, ZONE_HI = 0.15, 0.85
N_RANDOM_DRAWS = 20      # draws averaged for the random-detector baseline

CLASS_NAMES = ["None", "Left", "Right"]
RESULTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "results.jsonl"

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _confusion(per_frame, gt_frame_labels, coords_dict=None, zone_filter=False):
    """3x3 confusion matrix, rows=true, cols=pred."""
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for frame, pred, _conf in per_frame:
        if zone_filter and coords_dict is not None:
            xy = coords_dict.get(frame, [0.5, 0.0, 0.0])
            if not (ZONE_LO < xy[0] < ZONE_HI and ZONE_LO < xy[1] < ZONE_HI):
                continue
        true = gt_frame_labels.get(frame, 0)
        cm[true][pred] += 1
    return cm


def _prf_from_cm(cm):
    per_class, f1s = {}, []
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(3)) - tp
        fn = sum(cm[c]) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[CLASS_NAMES[c]] = {"precision": p, "recall": r, "f1": f, "support": sum(cm[c])}
        f1s.append(f)
    return per_class, sum(f1s) / 3


def cluster_detections(raw):
    """raw: list of (frame, cls, conf) with cls != 0. -> discrete detections."""
    detections = []
    for cls in (1, 2):
        cls_preds = sorted((r for r in raw if r[1] == cls), key=lambda r: r[0])
        if not cls_preds:
            continue
        cluster = [cls_preds[0]]
        for r in cls_preds[1:]:
            if r[0] - cluster[-1][0] <= MERGE_WINDOW:
                cluster.append(r)
            else:
                detections.append(max(cluster, key=lambda c: c[2]))
                cluster = [r]
        detections.append(max(cluster, key=lambda c: c[2]))

    detections.sort(key=lambda d: d[0])
    filtered, i = [], 0
    while i < len(detections):
        current = detections[i]
        j = i + 1
        while j < len(detections) and detections[j][0] - current[0] < CROSS_SUPPRESS:
            if detections[j][1] != current[1] and detections[j][2] > current[2]:
                current = detections[j]
            j += 1
        filtered.append(current)
        i = j
    return sorted(filtered, key=lambda d: d[0])


def match_events(detections, gt_left, gt_right):
    hits = 0
    matched = set()
    for frame, cls, _conf in detections:
        gt_list = gt_left if cls == 1 else gt_right
        for g in gt_list:
            if (g, cls) not in matched and MATCH_LO <= (g - frame) <= MATCH_HI:
                matched.add((g, cls))
                hits += 1
                break
    misses = len(detections) - hits
    total_gt = len(gt_left) + len(gt_right)
    recall = len(matched) / total_gt if total_gt else 0.0
    precision = hits / (hits + misses) if (hits + misses) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"hits": hits, "false_alarms": misses, "matched_gt": len(matched),
            "total_gt": total_gt, "precision": precision, "recall": recall, "f1": f1}


def _random_baseline(frame_pool, n_detect, n_left, n_right, gt_left, gt_right, seed=0):
    """Detector firing uniformly at random with the same total detection count."""
    if n_detect == 0 or not frame_pool:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "left_points": 0, "right_points": 0}
    rng = random.Random(seed)
    acc = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for _ in range(N_RANDOM_DRAWS):
        picks = rng.sample(frame_pool, min(n_detect, len(frame_pool)))
        dets = []
        for k, f in enumerate(sorted(picks)):
            cls = 1 if k < n_left else 2
            dets.append((f, cls, 1.0))
        m = match_events(sorted(dets), gt_left, gt_right)
        for k in acc:
            acc[k] += m[k]
    for k in acc:
        acc[k] /= N_RANDOM_DRAWS
    return {**acc, "left_points": n_left, "right_points": n_right}


def evaluate_stage2(per_frame, gt_left, gt_right, coords_dict=None,
                    min_confidence=0.0, use_zone_filter_at_predict=False,
                    max_frames_ahead=30, verbose=False):
    """Grade one Stage-2 run. Returns a flat metrics dict (JSON-serialisable)."""
    gt_left = sorted(gt_left)
    gt_right = sorted(gt_right)

    # Per-frame ground truth uses the SAME rectangle rule the datasets train on,
    # so per-frame F1 here is comparable to what the trainers report.
    all_events = sorted([(f, 1) for f in gt_left] + [(f, 2) for f in gt_right])
    gt_frame_labels = {}
    for frame, _p, _c in per_frame:
        lab = 0
        for ef, cls in all_events:
            diff = ef - frame
            if 0 <= diff <= max_frames_ahead:
                lab = cls
                break
            if diff > max_frames_ahead:
                break
        gt_frame_labels[frame] = lab

    # 1. per-frame metrics -- zone filter OFF is primary (see E2)
    cm = _confusion(per_frame, gt_frame_labels, coords_dict, zone_filter=False)
    per_class, macro_f1 = _prf_from_cm(cm)
    cm_zone = _confusion(per_frame, gt_frame_labels, coords_dict, zone_filter=True)
    _, macro_f1_zone = _prf_from_cm(cm_zone)

    # 2. event-level metrics
    raw = []
    for frame, cls, conf in per_frame:
        if cls == 0 or conf < min_confidence:
            continue
        if use_zone_filter_at_predict and coords_dict is not None:
            xy = coords_dict.get(frame, [0.5, 0.0, 0.0])
            if not (ZONE_LO < xy[0] < ZONE_HI and ZONE_LO < xy[1] < ZONE_HI):
                continue
        raw.append((frame, cls, conf))

    detections = cluster_detections(raw)
    event = match_events(detections, gt_left, gt_right)

    # 3. scoreline
    pred_left = sum(1 for d in detections if d[1] == 1)
    pred_right = sum(1 for d in detections if d[1] == 2)
    true_left, true_right = len(gt_left), len(gt_right)

    # 4. baselines
    maj_cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for frame, _p, _c in per_frame:
        maj_cm[gt_frame_labels[frame]][0] += 1
    _, maj_macro_f1 = _prf_from_cm(maj_cm)

    frame_pool = [f for f, _p, _c in per_frame]
    rand = _random_baseline(frame_pool, len(detections), pred_left, pred_right,
                            gt_left, gt_right)

    out = {
        "n_frames_scored": len(per_frame),
        "n_raw_firings": len(raw),
        "n_detections": len(detections),
        "frame_macro_f1": macro_f1,
        "frame_macro_f1_zone_filtered": macro_f1_zone,
        "frame_per_class": per_class,
        "confusion_matrix": cm,
        "event_precision": event["precision"],
        "event_recall": event["recall"],
        "event_f1": event["f1"],
        "event_hits": event["hits"],
        "event_false_alarms": event["false_alarms"],
        "total_gt_events": event["total_gt"],
        "pred_score": [pred_left, pred_right],
        "true_score": [true_left, true_right],
        "scoreline_abs_err_left": abs(pred_left - true_left),
        "scoreline_abs_err_right": abs(pred_right - true_right),
        "scoreline_abs_err_total": abs(pred_left - true_left) + abs(pred_right - true_right),
        "baseline_majority_frame_macro_f1": maj_macro_f1,
        "baseline_majority_event_f1": 0.0,
        "baseline_random_event_precision": rand["precision"],
        "baseline_random_event_recall": rand["recall"],
        "baseline_random_event_f1": rand["f1"],
        "min_confidence": min_confidence,
        "zone_filter_at_predict": use_zone_filter_at_predict,
    }

    if verbose:
        print_report(out)
    return out


def print_report(m):
    print("\n" + "=" * 62)
    print("STAGE-2 UNIFIED EVALUATION")
    print("=" * 62)
    print(f"frames scored: {m['n_frames_scored']}   raw firings: {m['n_raw_firings']}"
          f"   detections: {m['n_detections']}")
    print("\nConfusion matrix (rows=true, cols=pred, zone filter OFF):")
    print(f"{'':>10}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>10}" + "".join(f"{m['confusion_matrix'][i][j]:>10}" for j in range(3)))
    print(f"\n{'class':>8} {'precision':>10} {'recall':>8} {'f1':>7} {'support':>8}")
    for name, v in m["frame_per_class"].items():
        print(f"{name:>8} {v['precision']:>10.3f} {v['recall']:>8.3f} {v['f1']:>7.3f} {v['support']:>8}")
    print(f"\nper-frame macro F1 (zone OFF): {m['frame_macro_f1']:.4f}"
          f"   (zone ON: {m['frame_macro_f1_zone_filtered']:.4f})")
    print(f"event-level  P={m['event_precision']:.3f}  R={m['event_recall']:.3f}"
          f"  F1={m['event_f1']:.3f}   ({m['event_hits']} hits /"
          f" {m['event_false_alarms']} false alarms / {m['total_gt_events']} gt)")
    print(f"scoreline    pred {m['pred_score'][0]}-{m['pred_score'][1]}"
          f"   true {m['true_score'][0]}-{m['true_score'][1]}"
          f"   abs err {m['scoreline_abs_err_total']}")
    print("-" * 62)
    print(f"BASELINE majority (always none): frame macro F1"
          f" {m['baseline_majority_frame_macro_f1']:.4f} | event F1 0.000")
    print(f"BASELINE random  (same count)  : event P"
          f" {m['baseline_random_event_precision']:.3f}"
          f"  R {m['baseline_random_event_recall']:.3f}"
          f"  F1 {m['baseline_random_event_f1']:.3f}")
    print("=" * 62)


def append_result(exp_id, arm, seed, metrics, config, game=None):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "exp_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "arm": arm,
        "seed": seed,
        "game": game,
        "config": config,
        **metrics,
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row
