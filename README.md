# Table Tennis Referee

An automated referee for table tennis broadcast video: given a full-HD 120 fps side-view
recording, the system locates the ball on every annotated frame, detects when each rally ends,
attributes the point to the left or right player and accumulates a final scoreline.

The score is **counted from detected rally endings**, never read off the on-screen scoreboard -
the scoreboard is blacked out before the video ever reaches the backbone.

---

## Pipeline

![Pipeline architecture](figures/architecture.png)

---

### Stage 1 : ball localisation

A frozen DINOv2 ViT-S/14 produces 37×37 = 1369 patch tokens per frame. A trainable `BallHead`
(~7.1M params: a learned CLS token, 4 ViT blocks, LayerNorm, `Linear(384->2)`, sigmoid) attends over
those tokens and regresses normalised `(x, y)`.

The backbone is frozen and its output is cached once. The DINOv2 forward pass is by far the most
expensive part of the pipeline, so caching turns Stage-1 training from hours into minutes and lets
Stage 2 reuse exactly the same features.

Loss is smooth-L1 with **`beta = 0.05`**, coordinates live in `[0, 1]`.

### Stage 2 : rally-ending detection and scoring

Per frame, the input is the 384-d Stage-1 CLS embedding concatenated with the predicted ball
coordinates. A frame is labelled `left`/`right` if a rally ending of that class falls within the
next `max_frames_ahead = 30` frames (0.25 s), else `none`. Frames are grouped into contiguous runs
(gaps > 150 frames start a new run) and chunked into sequences of 256.

Two formulations are maintained:

| | classification arm (`cls`) | regression arm (`reg`) |
|---|---|---|
| Model | `ScorePredictorClsSequential` | `ScorePredictorSequential` |
| Temporal core | **bidirectional** `nn.GRU` | unidirectional `nn.GRUCell` loop |
| Extra features | + `(x, y, fresh, is_left, is_right)` -> 389-d | + `(x, y, fresh)` -> 387-d, plus running score and previous-step probs fed back |
| Extra head | -- | `fc_frames`: countdown to the event, used to shift each firing forward |
| Operating threshold | `min_conf = 0.65` | `min_conf = 0.65` |

**The bidirectional GRU is the single most important result.** A rally ending is barely visible at
the moment it happens; what identifies it is the aftermath: the ball goes dead, the players reset,
a serve follows. A causal model has to guess from the run-up.

---

## Results

Stage 2, evaluated on `game_3`, trained on `game_1, 2, 4, 5` and tested on 7 test videos.

| | Validation on `game_3` | Test Videos |
|---|---|---|
| Detection recall | 0.925 | 0.379 $\pm$ 0.017 |
| Detection precision | 0.729 | 0.893 $\pm$ 0.018 |
| Attribution Accuracy | 0.954 | 0.775 $\pm$ 0.072 |
| Event F1 | 0.777 | 0.452 $\pm$ 0.018 |
| Winner identified correctly | 3/3 seeds | 4 / 7 matches |

---

## Repository layout

```
fetch_data.py              download OpenTT videos + annotations into ./data
DATASETS.md                dataset schema: label taxonomy, paths, o3b CLI

src/
  # Stage 1 - ball localisation
  frame_reader.py          build the DINOv2 token cache (run once, slow)
  cache_dataset.py         CachedBallDataset, build_split, bad-label filtering
  ball_head.py             BallHead regressor
  metrics.py               pixel error, PCE@{5,15,50}, mean-predictor baseline
  train.py                 Stage-1 training loop
  evaluate.py              per-game metrics + <game>_predictions.csv
  extract_features.py      dump 384-d CLS embeddings -> cls_token_features.pt
  overlay.py               annotated video + worst-k error frames as PNGs
  bad_labels.py            flag implausible GT jumps (>150 px on both neighbours)

  # Stage 2 - rally-ending detection and scoring
  score_constants.py       LEFT_SCORES / RIGHT_SCORES  (source of truth)
  score_dataset.py         SequentialClsDataset       (389-d)
  score_head.py            ScorePredictorClsSequential (bidirectional GRU)
  score_dataset_regression.py  SequentialScoreDataset (387-d, + frames_until, frame_weight)
  score_head_regression.py     ScorePredictorSequential (GRUCell + countdown head)
  eval_stage2.py           The grading harness - clustering, matching, all metrics
  run_experiment.py        multi-seed train/eval driver for both arms
  score_train.py           pinned baseline config, cls arm
  score_train_regression.py  pinned baseline config, reg arm
  score_predict.py         standalone inference + grading, cls arm
  score_predict_regression.py  standalone inference + grading, reg arm
  score_eval_utils.py      GT event loading, scoreboard-change verification
  overlay_scores.py        video overlay: predicted vs true scoreline and ball

data/
  OpenTT/annotations/{train,test}/  game_N.json (events), game_N_ball.json (coordinates)
  OpenTT/videos/{train,test}/       game_1..5.mp4, test_1..7.mp4
  OpenTT_Preprocess/video_bboxes.json   scoreboard + per-digit boxes, per video
  dino_cache/<game>/cache.pt, cls_token_features.pt
  bad_label_candidates/<game>.json      frames excluded from Stage-1 training

```

---

## Setup and reproduction

Requires PyTorch (CUDA), `timm`, `opencv-python`, `matplotlib`, `numpy`. Run everything from
the repository root.

```bash
# 0. data (~several GB of video)
python fetch_data.py

# 1. build the DINOv2 token cache - slow (takes many hours) and only needs doing once.
python src/frame_reader.py

# 2. Stage 1: train the ball regressor (game-level split)
python src/train.py --train-games game_1 game_2 game_3 game_4 --val-games game_5

# 3. Stage 1: metrics + the coordinate CSVs that Stage 2 consumes
python src/evaluate.py --ckpt runs/<run>/ckpt_best.pt --games game_1 game_2 game_3 game_4 game_5

# 4. Stage 1: dump the CLS embeddings Stage 2 consumes
python src/extract_features.py --ckpt runs/<run>/ckpt_best.pt --games game_1 game_2 game_3 game_4 game_5

# 5. Stage 2: the shipped configuration, 3 seeds
python src/run_experiment.py --exp-id gru_bidir_cls --arm cls --seeds 0 1 2
python src/run_experiment.py --exp-id gru_reg --arm reg --seeds 0 1 2

# 6. qualitative check
python src/overlay_scores.py --game game_3 --arm cls --min-conf 0.65 --max-frames 3000
python src/overlay.py --ckpt runs/<run>/ckpt_best.pt --game game_5 --worst-k 24
```

---

## Data

[Extended OpenTTGames](https://arxiv.org/abs/2512.19327) - 12 full-HD 120 fps recordings from a
fixed side-view camera, 5 train / 7 test, CC BY-NC-SA 4.0. Annotations from
[moamal01/table_tennis_data](https://github.com/moamal01/table_tennis_data); videos from
`lab.osai.ai/datasets/openttgames/data/`. Label schema is documented in `DATASETS.md`.

Rally-ending labels are prefixed with the player who *caused* the ending, which is not always the
player who *wins* the point, `left_out` awards the point to the right player. `score_constants.py`
encodes this mapping; the training annotations contain 223 rally endings and the test annotations 58.