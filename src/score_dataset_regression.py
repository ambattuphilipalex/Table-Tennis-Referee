import json
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SequentialScoreDataset(Dataset):
    def __init__(self, cache_path, event_json_path, csv_path,
                 chunk_len=256, max_frame_gap=150, max_frames_ahead=30, rally_window=200, far_background_weight=0.2):
        cache_path = Path(cache_path)
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.frames_raw = blob["frames"]
        self.rally_window = rally_window
        self.far_background_weight = far_background_weight

        feat_path = cache_path.parent / "cls_token_features.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing {feat_path}. Run extract_features.py first.")
        feat_blob = torch.load(feat_path, weights_only=False)
        feature_frames = feat_blob["frames"].tolist()
        self.feature_dim = feat_blob["features"].shape[-1]
        self.feature_by_frame = {
            int(f): feat_blob["features"][i] for i, f in enumerate(feature_frames)
        }

        with open(event_json_path, "r") as f:
            self.events = json.load(f)

        self.chunk_len = chunk_len
        self.max_frames_ahead = max_frames_ahead

        self.left_scores = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
        self.right_scores = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]

        self.left_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.left_scores)
        )
        self.right_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.right_scores)
        )
        # (frame, class) pairs, sorted by frame, used for per-frame lookahead labeling
        self.all_events = sorted(
            [(f, 1) for f in self.left_event_frames] + [(f, 2) for f in self.right_event_frames]
        )

        self.coords_dict = self._load_coordinates(csv_path)

        valid_frame_nos = sorted(
            int(f) for f in self.frames_raw.tolist() if int(f) != -1 and int(f) in self.feature_by_frame
        )

        runs = []
        if valid_frame_nos:
            cur_run = [valid_frame_nos[0]]
            for prev_f, f in zip(valid_frame_nos, valid_frame_nos[1:]):
                if f - prev_f > max_frame_gap:
                    runs.append(cur_run)
                    cur_run = [f]
                else:
                    cur_run.append(f)
            runs.append(cur_run)

        self.chunks = []  # list of (frame_list, is_run_start)
        for run in runs:
            n_chunks = (len(run) + chunk_len - 1) // chunk_len
            for c in range(n_chunks):
                seg = run[c * chunk_len:(c + 1) * chunk_len]
                if len(seg) < 2:
                    continue
                self.chunks.append((seg, c == 0))

        n_runs = len(runs)
        print(f"  built {len(self.chunks)} ordered chunks from {n_runs} contiguous run(s) "
              f"({len(valid_frame_nos)} usable frames)")

    def _load_coordinates(self, csv_path):
        coords = {}
        last_valid_x, last_valid_y = 0.5, 0.0
        try:
            with open(csv_path, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        f_num = int(float(row["frame"]))
                        x = float(row.get("x_norm", -1))
                        y = float(row.get("y_norm", -1))
                        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                            last_valid_x, last_valid_y = x, y
                            fresh = 1.0
                        else:
                            x, y = last_valid_x, last_valid_y
                            fresh = 0.0
                        coords[f_num] = [x, y, fresh]
                    except (ValueError, KeyError):
                        coords[f_num] = [last_valid_x, last_valid_y, 0.0]
        except Exception as e:
            print(f"Warning: Tracking CSV read error ({e}).")
        return coords

    def _score_before(self, frame_no):
        """Ground-truth (left, right) score BEFORE this frame. Never assumes 0-0."""
        left = sum(1 for f in self.left_event_frames if f < frame_no)
        right = sum(1 for f in self.right_event_frames if f < frame_no)
        return left, right

    def _target_at(self, frame_no):
        """Per-frame lookahead target: nearest upcoming event within horizon."""
        best = None
        for ef, cls in self.all_events:
            diff = ef - frame_no
            if 0 <= diff <= self.max_frames_ahead:
                if best is None or diff < best[0]:
                    best = (diff, cls)
            elif ef - frame_no > self.max_frames_ahead:
                break 
        if best is None:
            return self.max_frames_ahead, 0
        return best

    def __len__(self):
        return len(self.chunks)

    def _distance_to_nearest_event(self, frame_no):
        if not self.all_events:
            return float("inf")
        return min(abs(ef - frame_no) for ef, _ in self.all_events)

    def __getitem__(self, idx):
        seg, is_run_start = self.chunks[idx]

        feats = torch.stack([self.feature_by_frame[f] for f in seg]).float()
        coords = torch.tensor(
            [self.coords_dict.get(f, [0.5, 0.0, 0.0]) for f in seg], dtype=torch.float32
        )
        fused = torch.cat([feats, coords], dim=-1)  # [L, feature_dim + 3]

        event_type = torch.zeros(len(seg), dtype=torch.long)
        frames_until = torch.zeros(len(seg), dtype=torch.float32)
        for i, f in enumerate(seg):
            diff, cls = self._target_at(f)
            event_type[i] = cls
            frames_until[i] = float(diff) / self.max_frames_ahead

        frame_weight = torch.zeros(len(seg))
        for i, f in enumerate(seg):
            dist = self._distance_to_nearest_event(f)
            frame_weight[i] = 1.0 if dist <= self.rally_window else self.far_background_weight

        left0, right0 = self._score_before(seg[0])

        return {
            "features": fused,                                   # [L, feature_dim+3]
            "event_type": event_type,                             # [L]
            "frames_until": frames_until,                         # [L]
            "score_state": torch.tensor([left0, right0], dtype=torch.float32),  # [2], ground truth, not 0-0
            "is_run_start": torch.tensor(is_run_start),
            "frame_numbers": torch.tensor(seg, dtype=torch.int64),
            "frame_weight": frame_weight,
        }