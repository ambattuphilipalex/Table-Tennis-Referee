import json
import bisect
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceScoreDataset(Dataset):
    def __init__(self, cache_path, event_json_path, csv_path, seq_len=64,
            event_match_tolerance=30, max_frame_gap=150):
        cache_path = Path(cache_path)
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.frames = blob["frames"]

        feat_path = cache_path.parent / "cls_token_features.pt"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Missing {feat_path}. Run extract_features.py first."
            )
        feat_blob = torch.load(feat_path, weights_only=False)
        feature_frames = feat_blob["frames"].tolist()
        self.feature_dim = feat_blob["features"].shape[-1]
        self.feature_by_frame = {
            int(f): feat_blob["features"][i] for i, f in enumerate(feature_frames)
        }

        with open(event_json_path, 'r') as f:
            self.events = json.load(f)

        self.seq_len = seq_len
        self.event_match_tolerance = event_match_tolerance

        self.valid_indices = [
            i for i, f in enumerate(self.frames.tolist())
            if f != -1 and int(f) in self.feature_by_frame
        ]

        self.left_scores = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
        self.right_scores = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]

        self.left_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.left_scores)
        )
        self.right_event_frames = sorted(
            int(fno) for fno, ev in self.events.items() if any(s in ev for s in self.right_scores)
        )

        print(f"Game events - Left: {len(self.left_event_frames)}, Right: {len(self.right_event_frames)}")
        
        self.coords_dict = self._load_coordinates(csv_path)
        self.window_starts = self._build_valid_windows(max_frame_gap)

    def _load_coordinates(self, csv_path):
        coords = {}
        last_valid_x = 0.5
        last_valid_y = 0.0

        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        f_num = int(float(row['frame']))
                        x = float(row.get('x_norm', -1))
                        y = float(row.get('y_norm', -1))
                        
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

    def _build_valid_windows(self, max_frame_gap):
        starts = []
        n = len(self.valid_indices) - self.seq_len + 1
        dropped = 0
        for start in range(max(0, n)):
            window = self.valid_indices[start:start + self.seq_len]
            frame_nos = [int(self.frames[i]) for i in window]
            gaps = [b - a for a, b in zip(frame_nos, frame_nos[1:])]
            if gaps and max(gaps) > max_frame_gap:
                dropped += 1
                continue
            starts.append(start)
        if n > 0:
            print(f"  window filter: kept {len(starts)}/{n} windows "
                  f"(dropped {dropped} that spanned a >{max_frame_gap}-frame gap)")
        return starts

    def __len__(self):
        return len(self.window_starts)

    def _label_for_window(self, window_indices):
        last_window_frame = int(self.frames[window_indices[-1]])
        
        for event_frame in self.left_event_frames:
            if 0 <= (event_frame - last_window_frame) <= 30:
                return 1
        for event_frame in self.right_event_frames:
            if 0 <= (event_frame - last_window_frame) <= 30:
                return 2
        
        return 0

    def label_only(self, idx):
        start = self.window_starts[idx]
        window_indices = self.valid_indices[start: start + self.seq_len]
        return self._label_for_window(window_indices)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        window_indices = self.valid_indices[start: start + self.seq_len]

        real_features = torch.stack([
            self.feature_by_frame[int(self.frames[i])] for i in window_indices
        ]).float()

        window_coords = []
        for index in window_indices:
            frame_no = int(self.frames[index])
            xy = self.coords_dict.get(frame_no, [0.5, 0.0, 0.0])
            window_coords.append(xy)

        label = self._label_for_window(window_indices)

        coords_tensor = torch.tensor(window_coords, dtype=torch.float32)
        fused_features = torch.cat([real_features, coords_tensor], dim=-1)

        return fused_features, torch.tensor(label, dtype=torch.long)