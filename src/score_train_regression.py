import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, ConcatDataset
from score_head_regression import ScorePredictorRegression
from score_dataset_regression import SequenceScoreDatasetRegression

TRAIN_GAME_IDS = ["1", "2", "4", "5"]
EVAL_GAME_ID = "3"
VAL_CHECK_EVERY = 5


def build_dataset(gid):
    cache_path = f"data/dino_cache/game_{gid}/cache.pt"
    event_json = f"data/OpenTT/annotations/train/game_{gid}.json"
    csv_path = f"runs/20260704-1040_baseline/game_{gid}_predictions.csv"
    return SequenceScoreDatasetRegression(cache_path, event_json, csv_path)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Train games: {TRAIN_GAME_IDS} | Eval game: {EVAL_GAME_ID}")

    dataset_list = []
    for gid in TRAIN_GAME_IDS:
        ds = build_dataset(gid)
        dataset_list.append(ds)
        print(f"Loaded game_{gid}: {len(ds)} windows")

    full_dataset = ConcatDataset(dataset_list)
    
    type_counts = Counter()
    for i in range(len(full_dataset)):
        _, target = full_dataset[i]
        type_counts[int(target[1].item())] += 1
    
    print(f"Event type distribution: {dict(type_counts)}")

    total = sum(type_counts.values())
    w0 = total / (3 * type_counts.get(0, 1))
    w1 = total / (3 * type_counts.get(1, 1))
    w2 = total / (3 * type_counts.get(2, 1))
    class_weights = torch.tensor([w0, w1, w2]).to(device)
    print(f"Class weights: [BG={w0:.1f}, Left={w1:.1f}, Right={w2:.1f}]")
    
    loader = DataLoader(full_dataset, batch_size=16, shuffle=True,
                        num_workers=2, pin_memory=(device == "cuda"))

    sample_features, _ = full_dataset[0]
    input_dim = sample_features.shape[-1]
    print(f"Input dim: {input_dim}")
    
    model = ScorePredictorRegression(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights) 

    print("\nStarting regression training...")
    best_val_loss = float('inf')
    
    for epoch in range(60):
        model.train()
        total_loss = 0.0
        
        for tokens, targets in loader:
            tokens = tokens.to(device).float()
            frames_target = targets[:, 0:1].to(device)
            type_target = targets[:, 1].long().to(device)
            
            optimizer.zero_grad()
            frames_pred, type_logits = model(tokens)
            
            loss_type = ce_loss(type_logits, type_target)
            
            pos_mask = type_target != 0
            if pos_mask.any():
                loss_frames = mse_loss(frames_pred[pos_mask], frames_target[pos_mask])
            else:
                loss_frames = torch.tensor(0.0, device=device)
            
            loss = loss_frames + loss_type
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(loader)
        
        if (epoch + 1) % VAL_CHECK_EVERY == 0 or epoch == 59:
            model.eval()
            val_loss = 0.0
            eval_ds = build_dataset(EVAL_GAME_ID)
            eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False,
                                     num_workers=2, pin_memory=(device == "cuda"))
            with torch.no_grad():
                for tokens, targets in eval_loader:
                    tokens = tokens.to(device).float()
                    frames_target = targets[:, 0:1].to(device)
                    type_target = targets[:, 1].long().to(device)
                    
                    frames_pred, type_logits = model(tokens)
                    
                    loss_type = ce_loss(type_logits, type_target)
                    pos_mask = type_target != 0
                    if pos_mask.any():
                        loss_frames = mse_loss(frames_pred[pos_mask], frames_target[pos_mask])
                    else:
                        loss_frames = torch.tensor(0.0, device=device)
                    
                    val_loss += (loss_frames + loss_type).item()
            
            avg_val_loss = val_loss / len(eval_loader)
            print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "score_predictor_regression.pth")
                print(f"  -> new best, saved")
        else:
            print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f}")

    print(f"\nBest val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()