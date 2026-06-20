# Datasets

All datasets share the same CLI surface:

```bash
o3b dataset fetch -d <config>           # download raw data
o3b dataset index -d <config>           # build index / manifest
o3b dataset viz   -d <config>           # visualise samples
```

A *config* is either a short name resolved from
`third_party/o3b/src/configs/dataset/` (e.g. `opentt`) or a full path
to a YAML file.  Add `--platform slurm` (or any other platform name) to have
the platform's `path_datasets_raw` / `path_datasets_preprocess` variables
override the paths in the config before it is loaded.

---

## HouseCorr3D

| | |
|---|---|
| **Config names** | `hc3d`, `hc3d_object`, `hc3d_object_pair` |
| **Class** | `HouseCorr3D` |
| **Item types** | `object`, `object_pair` |
| **Content** | 3-D household objects with mesh, keypoints, and per-vertex part IDs |

### Paths (default)

```
{path_datasets_raw}/Omni6DPose/
{path_datasets_preprocess}/Omni6DPose_Preprocess/
```

### CLI

```bash
o3b dataset fetch -d hc3d
o3b dataset index -d hc3d
o3b dataset viz   -d hc3d --limit 20
o3b dataset viz   -d hc3d --render                   # interactive viser viewer
o3b dataset viz   -d hc3d --render-frames 6          # render 6 viewpoints (pyrender)
o3b dataset viz   -d hc3d --render-frames 6 --renderer nvdiffrast
```

---

## DenseMatcher

| | |
|---|---|
| **Config names** | `dm`, `dm_object`, `dm_object_pair` |
| **Class** | `DenseMatcher` |
| **Item types** | `object`, `object_pair` |
| **Content** | Diverse 3-D meshes organised by category for dense correspondence learning |

### Paths (default)

```
{path_datasets_raw}/DenseMatcher/
{path_datasets_preprocess}/DenseMatcher_Preprocess/
```

### CLI

```bash
o3b dataset fetch -d dm
o3b dataset index -d dm
o3b dataset viz   -d dm --limit 20
```

---

## OpenTT

| | |
|---|---|
| **Config name** | `opentt` |
| **Class** | `OpenTT` |
| **Item type** | `Scene` (sliding-window video clip) |
| **Content** | Table tennis matches with per-frame game-event annotations |
| **Videos** | 12 full-HD recordings at **120 fps** from a static side-view camera |
| **Splits** | `train` (5 games) · `test` (7 games) · `all` |
| **License** | CC BY-NC-SA 4.0 |
| **Paper** | [arXiv 2512.19327](https://arxiv.org/abs/2512.19327) |

### Sources

| Asset | Origin |
|---|---|
| Annotation JSONs | [github.com/moamal01/table_tennis_data](https://github.com/moamal01/table_tennis_data) |
| MP4 videos | `https://lab.osai.ai/datasets/openttgames/data/` |

### Paths (default)

```
{path_datasets_raw}/OpenTT/
    annotations/
        train/  game_1.json … game_5.json
        test/   test_1.json … test_7.json
    videos/
        train/  game_1.mp4 … game_5.mp4
        test/   test_1.mp4 … test_7.mp4

{path_datasets_preprocess}/OpenTT_Preprocess/
    manifest.json          # frame counts per video (written by index)
```

### Annotations

Each JSON file maps an integer frame number to an event label string.
Three event categories exist:

**Ball events** — simple string label:

| Label | Meaning |
|---|---|
| `bounce` | Ball bounces on the table |
| `net` | Ball hits the net |
| `empty_event` | Explicitly annotated as no event |

**Point endings** — prefixed with the side that caused the ending:

| Label | Meaning |
|---|---|
| `left_net` / `right_net` | Player hits net or own-side table |
| `left_winner` / `right_winner` | Unreachable shot |
| `left_double_bounce` / `right_double_bounce` | Ball bounces twice |
| `left_out` / `right_out` | Ball misses the table |
| `left_miss_on_own_side` / `right_miss_on_own_side` | Ball dips below own side |
| `left_not_hitting_ball` / `right_not_hitting_ball` | Player swings and misses |

**Stroke events** — space-separated triple `<technique> <lean> <feet>`:

```
left_forehand_loop  right_leaning  both_feet_planted
```

*Technique* prefix: `left_` or `right_` + `forehand_` or `backhand_` + one of:
`serve`, `loop`, `block`, `push`, `flick`, `lob`, `smash`

*Lean*: `neutral` · `back_heavy` · `front_heavy` · `right_leaning` · `left_leaning` · `unknown`

*Feet*: `both_feet_planted` · `both_feet_lifted` · `right_foot_lifted` · `left_foot_lifted` · `unknown`

Frames without any annotation receive `event = None`.

### Returned data type

`OpenTT.__getitem__` returns a `Scene`:

```python
@dataclass
class Scene:
    scene_id:    str              # e.g. "game_1_0002048"
    rgbs:        Tensor           # (T, H, W, 3)  float32 in [0, 1]
    events:      list[str|None]   # T entries — event label or None
    scoreboards: list[dict|None]  # T entries — {"bbox":[x1,y1,x2,y2],
                                  #   "score_left": int, "score_right": int}
                                  #   or None when not available
```

`scoreboards` is populated automatically when `{path_preprocess}/scoreboards.db`
exists (written by `o3b dataset preprocess -d opentt`).  Each entry holds the
nearest stored scoreboard result within 45 frames of the clip frame.

Multiple `Scene` instances can be batched with `collate_scenes()`:

```python
from o3b.data.datatypes import collate_scenes
batch = collate_scenes([scene_a, scene_b])
# batch.rgbs         (B, T_max, H, W, 3)
# batch.rgbs_mask    (B, T_max)  bool — True for valid (non-padded) frames
# batch.events       list[list[str|None]]  shape (B, T_max)
# batch.scoreboards  list[list[dict|None]] shape (B, T_max)
```

### Config knobs

| Key | Default | Meaning |
|---|---|---|
| `split` | `train` | `train`, `test`, or `all` |
| `scene_length` | `16` | Number of frames T per returned clip |
| `extra.frame_stride` | `1` | Step between consecutive frame indices within a clip |
| `extra.clip_stride` | `scene_length × frame_stride` | Step between clip start frames |
| `filter_count_max` | `null` | Cap total number of clips |

At 120 fps with `frame_stride: 4` each clip spans `scene_length × 4 / 120 ≈ 0.5 s` of real time.

### CLI

```bash
# Download annotations (~KB) and videos (~several GB total)
o3b dataset fetch -d opentt

# Scan videos, print clip statistics, write manifest.json
o3b dataset index -d opentt

# Show 4 clip strips in matplotlib (frames left→right, event label as title,
# gold scoreboard bbox + score text overlaid when preprocess DB is available)
o3b dataset viz -d opentt --limit 4

# Restrict to a single video
o3b dataset viz -d opentt --limit 4 --object-id game_1

# Print strip tensor shape + events alongside each window
o3b dataset viz -d opentt --limit 4 --debug
```

### Preprocess — scoreboard detection

The `preprocess` command runs **Qwen2.5-VL** on every Nth frame of each video,
asks it to locate the scoreboard and read the score, and stores the results in a
SQLite database.  The run is fully resumable: already-stored frames are skipped.

```bash
# Run on all train videos (default: every 30th frame ≈ 4 fps at 120 fps)
o3b dataset preprocess -d opentt

# Run on a single video only
o3b dataset preprocess -d opentt --video game_1

# Use a smaller or larger Qwen3.5 variant
o3b dataset preprocess -d opentt --model Qwen/Qwen3.5-0.8B   # smallest
o3b dataset preprocess -d opentt --model Qwen/Qwen3.5-2B
o3b dataset preprocess -d opentt --model Qwen/Qwen3.5-4B     # default
o3b dataset preprocess -d opentt --frame-stride 60 --batch-size 8

# Write the database to a custom location
o3b dataset preprocess -d opentt --db /tmp/scores.db
```

**Flags**

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-4B` | Hugging Face VLM model ID (`Qwen3.5-0.8B` / `2B` / `4B`) |
| `--frame-stride` | `30` | Process every Nth frame |
| `--batch-size` | `4` | Frames per model forward pass |
| `--video` | *(all)* | Restrict to one video name |
| `--db` | `{path_preprocess}/scoreboards.db` | SQLite output path |

**Output SQLite schema** (`scoreboards` table)

| Column | Type | Description |
|---|---|---|
| `video_name` | TEXT | e.g. `game_1`, `test_3` |
| `frame_idx` | INTEGER | 0-based frame number in the source mp4 |
| `bbox_x1/y1/x2/y2` | REAL | Scoreboard bounding box in pixels; `NULL` if not detected |
| `score_left` | INTEGER | Left player's score in the current game; `NULL` if unreadable |
| `score_right` | INTEGER | Right player's score |
| `score_raw` | TEXT | Raw model output (for debugging / re-parsing) |

**Python API**

```python
import sqlite3
from o3b.dataset.opentt import OpenTT

OpenTT.preprocess(cfg, frame_stride=30, batch_size=4)

# Query results
con = sqlite3.connect("data/OpenTT_Preprocess/scoreboards.db")
rows = con.execute(
    "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2, score_left, score_right "
    "FROM scoreboards WHERE video_name = 'game_1' AND score_left IS NOT NULL "
    "ORDER BY frame_idx"
).fetchall()
con.close()
```

### Python API

```python
from pathlib import Path
from o3b.dataset.dataset import DatasetConfig, build_dataset
from o3b.dataset.opentt import OpenTT
from o3b.data.datatypes import collate_scenes

cfg = DatasetConfig(
    class_name  = "OpenTT",
    path_raw    = Path("data/OpenTT"),
    split       = "train",
    scene_length= 16,
    extra       = {"frame_stride": 4, "clip_stride": 64},
)

# one-time download
OpenTT.fetch(cfg)
OpenTT.index(cfg)

# iterate
ds    = build_dataset(cfg)        # or OpenTT(cfg)
scene = ds[0]
print(scene.scene_id)             # "game_1_0000000"
print(scene.rgbs.shape)           # torch.Size([16, 1080, 1920, 3])
print(scene.events)               # [None, 'bounce', None, …]

# visualise one clip
scene.viz()

# build a DataLoader
loader = ds.build_loader(batch_size=4, collate_fn=ds.collate_fn)
for batch in loader:
    # batch.rgbs  (4, 16, H, W, 3)
    pass
```
