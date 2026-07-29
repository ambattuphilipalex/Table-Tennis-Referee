import json
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from score_constants import LEFT_SCORES, RIGHT_SCORES


class SequentialClsDataset(Dataset):
    def __init__(self, cache_path, event_json_path, csv_path,
                 chunk_len=256, max_frame_gap=150, max_frames_ahead=30,
                 left_zone_x=0.4, right_zone_x=0.6):
        cache_path = Path(cache_path)
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.frames_raw = blob["frames"]

        feat_path = cache_path.parent / "cls_token_features.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing {feat_path}. Run extract_features.py first.")
        feat_blob = torch.load(feat_path, weights_only=False)
        feature_frames = feat_blob["frames"].tolist()
        self.cls_feature_dim = feat_blob["features"].shape[-1]
        self.feature_dim = self.cls_feature_dim + 5              # + (x, y, fresh, is_left, is_right)
        self.feature_by_frame = {
            int(f): feat_blob["features"][i] for i, f in enumerate(feature_frames)
        }

        with open(event_json_path, "r") as f:
            self.events = json.load(f)

        self.chunk_len = chunk_len
        self.max_frames_ahead = max_frames_ahead
        self.left_zone_x = left_zone_x
        self.right_zone_x = right_zone_x

        self.left_scores = LEFT_SCORES

        self.right_scores = RIGHT_SCORES
        self.left_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.left_scores)
        )
        self.right_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.right_scores)
        )
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

        self.chunks = []  # (frame_list, is_run_start)
        for run in runs:
            n_chunks = (len(run) + chunk_len - 1) // chunk_len
            for c in range(n_chunks):
                seg = run[c * chunk_len:(c + 1) * chunk_len]
                if len(seg) < 2:
                    continue
                self.chunks.append((seg, c == 0))

        print(f"  built {len(self.chunks)} ordered chunks from {len(runs)} contiguous run(s) "
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
        left = sum(1 for f in self.left_event_frames if f < frame_no)
        right = sum(1 for f in self.right_event_frames if f < frame_no)
        return left, right

    def _label_at(self, frame_no):
        for ef, cls in self.all_events:
            diff = ef - frame_no
            if 0 <= diff <= self.max_frames_ahead:
                return cls
            if diff > self.max_frames_ahead:
                break 
        return 0

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seg, is_run_start = self.chunks[idx]

        feats = torch.stack([self.feature_by_frame[f] for f in seg]).float()

        coords_rows = []
        for f in seg:
            x, y, fresh = self.coords_dict.get(f, [0.5, 0.0, 0.0])
            is_left = 1.0 if x < self.left_zone_x else 0.0
            is_right = 1.0 if x > self.right_zone_x else 0.0
            coords_rows.append([x, y, fresh, is_left, is_right])
        position_features = torch.tensor(coords_rows, dtype=torch.float32)  # [L, 5]

        fused = torch.cat([feats, position_features], dim=-1)  # [L, feature_dim]

        event_type = torch.tensor([self._label_at(f) for f in seg], dtype=torch.long)
        left0, right0 = self._score_before(seg[0])

        return {
            "features": fused,
            "event_type": event_type,
            "score_state": torch.tensor([left0, right0], dtype=torch.float32),
            "is_run_start": torch.tensor(is_run_start),
            "frame_numbers": torch.tensor(seg, dtype=torch.int64),
        }