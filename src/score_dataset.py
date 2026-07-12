import json
import bisect
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceScoreDataset(Dataset):
    def __init__(self, cache_path, event_json_path, csv_path, seq_len=64,
            label_lookback=10, event_match_tolerance=30, max_frame_gap=150):
        cache_path = Path(cache_path)
        blob = torch.load(cache_path, mmap=True, weights_only=False)
        self.frames = blob["frames"]  # [M] -- used only to know which frame numbers exist

        feat_path = cache_path.parent / "cls_token_features.pt"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Missing {feat_path}. Run extract_features.py with your trained "
                f"BallHead checkpoint first, e.g.:\n"
                f"  python extract_features.py --ckpt runs/<run>/ckpt_best.pt "
                f"--games {cache_path.parent.name}"
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
        self.label_lookback = min(label_lookback, seq_len)
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

        self.coords_dict = self._load_coordinates(csv_path)
        self.window_starts = self._build_valid_windows(max_frame_gap)

    def _load_coordinates(self, csv_path):
        coords = {}
        last_valid_x = 0.5
        last_valid_y = 0.0

        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'frame' in reader.fieldnames:
                    for row in reader:
                        f_num = int(float(row['frame']))
                        try:
                            x = float(row['x_norm']) if row['x_norm'] else -1.0
                            y = float(row['y_norm']) if row['y_norm'] else -1.0
                        except ValueError:
                            x, y = -1.0, -1.0

                        if x < 0 or y < 0 or x > 2 or y > 2:
                            x, y, fresh = last_valid_x, last_valid_y, 0.0
                        else:
                            last_valid_x, last_valid_y = x, y
                            fresh = 1.0

                        coords[f_num] = [x, y, fresh]
                else:
                    with open(csv_path, mode='r') as f_retry:
                        fallback_reader = csv.reader(f_retry)
                        first_row = next(fallback_reader, None)
                        start_idx = 0
                        if first_row and not first_row[0].replace('.', '', 1).isdigit():
                            pass
                        else:
                            if first_row:
                                coords[0] = [float(first_row[0]), float(first_row[1]), 1.0]
                                start_idx = 1
                        for idx, row in enumerate(fallback_reader, start=start_idx):
                            if len(row) >= 2:
                                try:
                                    x = float(row[0]) if row[0] else -1.0
                                    y = float(row[1]) if row[1] else -1.0
                                except ValueError:
                                    x, y = -1.0, -1.0

                                if x < 0 or y < 0 or x > 2 or y > 2:
                                    x, y, fresh = last_valid_x, last_valid_y, 0.0
                                else:
                                    last_valid_x, last_valid_y = x, y
                                    fresh = 1.0
                                coords[idx] = [x, y, fresh]
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

    def _nearest_event_match(self, frame_no, event_frames):
        if not event_frames:
            return False
        pos = bisect.bisect_left(event_frames, frame_no)
        candidates = []
        if pos < len(event_frames):
            candidates.append(event_frames[pos])
        if pos > 0:
            candidates.append(event_frames[pos - 1])
        return any(abs(c - frame_no) <= self.event_match_tolerance for c in candidates)

    def __len__(self):
        return len(self.window_starts)

    def _label_for_window(self, window_indices):
        label = 0
        for index in window_indices[-self.label_lookback:]:
            frame_no = int(self.frames[index])
            if self._nearest_event_match(frame_no, self.left_event_frames):
                label = 1
                break
            elif self._nearest_event_match(frame_no, self.right_event_frames):
                label = 2
                break
        return label

    def label_only(self, idx):
        """Cheap label lookup with no feature access -- use for dataset
        stats/class distribution instead of dataset[idx]."""
        start = self.window_starts[idx]
        window_indices = self.valid_indices[start: start + self.seq_len]
        return self._label_for_window(window_indices)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        window_indices = self.valid_indices[start: start + self.seq_len]

        real_features = torch.stack([
            self.feature_by_frame[int(self.frames[i])] for i in window_indices
        ]).float()  # [seq_len, feature_dim] -- feature_dim=384, the trained CLS embedding only

        window_coords = []
        for index in window_indices:
            frame_no = int(self.frames[index])
            xy = self.coords_dict.get(frame_no, [0.5, 0.0, 0.0])
            window_coords.append(xy)

        label = self._label_for_window(window_indices)

        coords_tensor = torch.tensor(window_coords, dtype=torch.float32)  # [seq_len, 3]
        fused_features = torch.cat([real_features, coords_tensor], dim=-1)  # [seq_len, feature_dim+3]

        return fused_features, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    cache_path = "data/dino_cache/game_1/cache.pt"
    event_json = "data/OpenTT/annotations/train/game_1.json"
    mock_csv = "runs/20260704-1040_baseline/game_1_predictions.csv"

    dataset = SequenceScoreDataset(cache_path, event_json, mock_csv)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Dataset initialized with trained CLS + patch features.")
    labels_seen = {0: 0, 1: 0, 2: 0}
    for tokens, label in loader:
        labels_seen[int(label.item())] += 1
    print(f"Fused tokens shape (Batch, Seq, Dim): {tokens.shape}")
    print(f"Label distribution: {labels_seen}")