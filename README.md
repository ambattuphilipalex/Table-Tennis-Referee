# Table Tennis Referee

An automated referee for table tennis broadcast video: given a full-HD 120 fps side-view
recording, the system locates the ball on every annotated frame, detects when each rally ends,
attributes the point to the left or right player and accumulates a final scoreline.

The score is **counted from detected rally endings**, never read off the on-screen scoreboard.
The scoreboard is blacked out before the video ever reaches the backbone.

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

Loss is smooth-L1 with **`beta = 0.05`**. Coordinates live in `[0, 1]`, so the PyTorch default of
`beta = 1.0` would put the quadratic-to-linear crossover above every observed error and the loss
would behave as plain MSE.

**Encoder comparison**, held-out `game_5`:

| encoder | resolution | patch | grid | tokens | mean px | median px |
|---|---|---|---|---|---|---|
| DINOv2 | 518 | 14 | 37×37 | 1369 | 30.6 | 18.6 |
| DINOv2 | 784 | 14 | 56×56 | 3136 | 30.6 | **17.5** |
| V-JEPA 2.1 | 384 | 16 | 24×24 | 576 | 31.3 | 26.8 |
| V-JEPA 2.1 | 512 | 16 | 32×32 | 1024 | **27.8** | 22.8 |

Mean-predictor baseline: 216 px median.

Localisation precision is bounded by patch-grid resolution rather than encoder type. V-JEPA's video
pretraining gives the best mean (fewer gross errors, since motion helps it avoid losing the ball)
but its coarser 16 px patches cannot match DINOv2's median. Raising resolution improved both
encoders, with sharply diminishing returns past 518.

### Stage 2 : rally-ending detection and scoring

Per frame, the input is the 384-d Stage-1 CLS embedding concatenated with position features derived
from the predicted ball coordinates. A frame is labelled `left`/`right` if a rally ending of that
class falls within the next `max_frames_ahead = 30` frames (0.25 s), else `none`. Frames are grouped
into contiguous runs (gaps > 150 frames start a new run) and chunked into sequences of 256.

Two formulations are maintained:

| | classification arm (`cls`) | regression arm (`reg`) |
|---|---|---|
| Model | `ScorePredictorClsSequential` | `ScorePredictorSequential` |
| Temporal core | **bidirectional** `nn.GRU` | unidirectional `nn.GRUCell` loop |
| Extra features | `(x, y, fresh, is_left, is_right)` -> 389-d | `(x, y, fresh)` -> 387-d, plus running score and previous-step probs fed back |
| Extra head | none | `fc_frames`: countdown to the event, used to shift each prediction forward |
| Shipped threshold | `min_conf = 0.6` | `min_conf = 0.9` |

**The bidirectional GRU is the single most important result.** A rally ending is barely visible at
the moment it happens; what identifies it is the aftermath: the ball goes dead, the players reset,
a serve follows. A causal model has to guess from the run-up.

The regression arm cannot be made bidirectional. Each step consumes its own previous prediction, so
running the sequence backwards would require predictions from the future.

---

## Results

Trained on `game_1, 2, 4, 5`, validated on `game_3`, tested on 7 held-out videos. 3 seeds.
Classification arm: 25 epochs, `min_conf = 0.6`. Regression arm: 60 epochs, `min_conf = 0.9`.

| held-out test | classification | random baseline | regression | random baseline |
|---|---|---|---|---|
| Detection recall | **0.379 ± 0.017** | 0.197 | 0.224 ± 0.017 | 0.174 |
| Detection precision | **0.893 ± 0.018** | 0.463 | 0.633 ± 0.090 | 0.488 |
| Attribution accuracy | **0.775 ± 0.072** | 0.500 | 0.517 ± 0.080 | 0.500 |
| Event F1 | **0.452 ± 0.018** | 0.150 | 0.186 ± 0.014 | 0.136 |
| Winner identified correctly | **4 / 7 matches** | 0.500 | 3 / 7 matches | 0.500 |

The random baseline places the same number of detections the model produced, with the same
left/right split, at random frames, and grades them identically over 20 draws. Matching the count
isolates skill from firing rate, which is why each arm has its own baseline.

The regression arm's attribution accuracy sits at chance: it locates rally endings slightly above
random but assigns the winner by coin flip.

**Metric definitions.** *Detection recall* is the fraction of real rally endings located, matching
on time alone. *Attribution accuracy* is the fraction of those located that were credited to the
correct player. Splitting them matters because the two failures need different remedies.

**Ablation: are the ball coordinates needed?** Zeroing the five position features while keeping the
input dimension fixed leaves performance unchanged (test event F1 0.464 ± 0.012 without them versus
0.452 ± 0.018 with), which suggests the CLS embedding already encodes ball position.

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
  eval_stage2.py           the grading harness - clustering, matching, all metrics
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

Requires PyTorch, `timm`, `opencv-python`, `matplotlib`, `numpy`. Run everything from the
repository root.

**Hardware.** Step 1 needs a CUDA GPU and takes several hours; the resulting token caches total
roughly 25 GB at 518 px across the 12 videos, and game_2 alone is ~18 GB. The caches are opened
with `mmap=True`, so RAM is not the constraint provided nothing indexes the whole tensor at once,
but the disk space is required. Everything after step 1 runs comfortably on CPU: Stage-2 training
is a small GRU over cached features, and a 3-seed run takes minutes.

```bash
# 0. data (~several GB of video)
python fetch_data.py

# 1. build the DINOv2 token cache - slow (many hours) and only needs doing once
python src/frame_reader.py

# 2. Stage 1: train the ball regressor (game-level split)
python src/train.py --train-games game_1 game_2 game_3 game_4 --val-games game_5

# 3. Stage 1: metrics + the coordinate CSVs that Stage 2 consumes
python src/evaluate.py --ckpt runs/<run>/ckpt_best.pt \
    --games game_1 game_2 game_3 game_4 game_5 test_1 test_2 test_3 test_4 test_5 test_6 test_7

# 4. Stage 1: dump the CLS embeddings Stage 2 consumes
python src/extract_features.py --ckpt runs/<run>/ckpt_best.pt \
    --games game_1 game_2 game_3 game_4 game_5 test_1 test_2 test_3 test_4 test_5 test_6 test_7

# 5. Stage 2: the shipped configuration, 3 seeds
python src/run_experiment.py --exp-id gru_bidir_cls --arm cls --seeds 0 1 2 \
    --epochs 25 --min-conf 0.6 \
    --test-games test_1 test_2 test_3 test_4 test_5 test_6 test_7

python src/run_experiment.py --exp-id gru_reg --arm reg --seeds 0 1 2 \
    --epochs 60 --min-conf 0.9 \
    --test-games test_1 test_2 test_3 test_4 test_5 test_6 test_7

# 6. per-match winner accuracy (the pooled test scoreline sums seven separate
#    matches, so the winner is only meaningful per video)
for g in test_1 test_2 test_3 test_4 test_5 test_6 test_7; do
  python src/run_experiment.py --exp-id w_$g --arm cls --seeds 0 \
      --epochs 25 --min-conf 0.6 --test-games $g
done

for g in test_1 test_2 test_3 test_4 test_5 test_6 test_7; do
  python src/run_experiment.py --exp-id regw_$g --arm reg --seeds 0 \
      --epochs 60 --min-conf 0.9 --test-games $g
done

# 7. qualitative check
python src/overlay_scores.py --game test_6 --arm cls \
    --model runs_stage2/<run>/seed0/ckpt_best.pt --min-conf 0.6
python src/overlay.py --ckpt runs/<run>/ckpt_best.pt --game game_5 --worst-k 24
```

Before step 5, set `BASELINE_RUN` at the top of `run_experiment.py` to the Stage-1 run folder from
step 3. It must contain `<game>_predictions.csv` for **every** game, test videos included. If a CSV
is missing the coordinate loader returns an empty dict, every frame falls back to a default whose
`y = 0.0` fails the zone filter, and the affected split silently produces zero detections with no
error.

Steps 5 and 6 write to `runs_stage2/<timestamp>_<exp-id>/`, one folder per seed with
`metrics.jsonl`, `curves.png`, `eval_metrics.json`, `test_metrics.json` and `ckpt_best.pt`.

---

## Data

[Extended OpenTTGames](https://arxiv.org/abs/2512.19327) - 12 full-HD 120 fps recordings from a
fixed side-view camera, 5 train / 7 test, CC BY-NC-SA 4.0. Annotations from
[moamal01/table_tennis_data](https://github.com/moamal01/table_tennis_data); videos from
`lab.osai.ai/datasets/openttgames/data/`. Label schema is documented in `DATASETS.md`.

Rally-ending labels are prefixed with the player who *caused* the ending, which is not always the
player who *wins* the point: `left_out` awards the point to the right player. `score_constants.py`
encodes this mapping.

**Rally endings per video:**

| | | | |
|---|---|---|---|
| game_1 | 24 | test_1 | 7 |
| game_2 | 85 | test_2 | 2 |
| game_3 | 31 | test_3 | 5 |
| game_4 | 30 | test_4 | 21 |
| game_5 | 49 | test_5 | 7 |
| | | test_6 | 8 |
| | | test_7 | 8 |
| **train** | **219** | **test** | **58** |

277 in total. The Stage-2 training set (`game_1, 2, 4, 5`) contains 188 rally endings, which is the
binding constraint on the whole system: Stage 1 learns from ~38,000 labelled ball positions, Stage 2
from 188 events.

**Ground truth is derived, not observed.** The scoreline is reconstructed by counting annotated
events rather than read from the scoreboard, and coverage is partial. In one clip the on-screen
scoreboard read 9-8 where the annotations gave 5-4, so the model can be penalised for correctly
detecting points that were never labelled. Reported figures are a lower bound.
