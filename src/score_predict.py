import torch
from score_head import ScorePredictor
from score_dataset import SequenceScoreDataset
from score_eval_utils import evaluate_model, load_gt_events
from score_train import EVAL_GAME_ID


def predict_and_evaluate(cache_path, event_json, csv_path, model_path="score_predictor.pth",
                          min_confidence=0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_left_frames, gt_right_frames = load_gt_events(event_json)
    true_left, true_right = len(gt_left_frames), len(gt_right_frames)
    print(f"Ground Truth Loaded: {true_left} Left Points, {true_right} Right Points in this video.")
    print(f"  -> Real Left Points at frames: {gt_left_frames}")
    print(f"  -> Real Right Points at frames: {gt_right_frames}")

    dataset = SequenceScoreDataset(cache_path, event_json, csv_path)
    inferred_input_dim = dataset[0][0].shape[-1]

    model = ScorePredictor(input_dim=inferred_input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\nStarting Auto-Grader...")
    metrics = evaluate_model(model, dataset, gt_left_frames, gt_right_frames, device,
                             verbose=True, min_confidence=min_confidence)

    print("\n" + "=" * 30)
    print("FINAL EVALUATION REPORT")
    print("=" * 30)
    print(f"True Final Score:      {true_left} - {true_right}")
    print(f"Predicted Final Score: {metrics['left_points']} - {metrics['right_points']}")
    print(f"Final score matches:   {(true_left, true_right) == (metrics['left_points'], metrics['right_points'])}")
    print(f"True Positives (Hits): {metrics['hits']}")
    print(f"False Positives:       {metrics['misses']}")
    print(f"Ground-truth points found (recall): {metrics['matched_gt']}/{metrics['total_gt']} "
          f"({metrics['recall']*100:.1f}%)")
    print(f"Model Precision: {metrics['precision']*100:.1f}%")
    print(f"F1: {metrics['f1']:.3f}")


if __name__ == "__main__":
    predict_and_evaluate(
        f"data/dino_cache/game_{EVAL_GAME_ID}/cache.pt",
        f"data/OpenTT/annotations/train/game_{EVAL_GAME_ID}.json",
        f"runs/20260704-1040_baseline/game_{EVAL_GAME_ID}_predictions.csv",
        min_confidence=0.65,
    )