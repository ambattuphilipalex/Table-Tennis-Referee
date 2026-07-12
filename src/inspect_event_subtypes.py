import json
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
EVENT_ROOT = DATA_ROOT / "OpenTT" / "annotations" / "train"

LEFT_SCORES = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
RIGHT_SCORES = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]


def subtype_counts(events, score_list):
    counts = Counter()
    for ev in events.values():
        for s in score_list:
            if s in ev:
                counts[s] += 1
                break
    return counts


def main():
    games = ["1", "2", "3", "4", "5"]
    all_left, all_right = {}, {}

    for gid in games:
        path = EVENT_ROOT / f"game_{gid}.json"
        if not path.exists():
            print(f"game_{gid}: no event json, skip")
            continue
        with open(path) as f:
            events = json.load(f)
        all_left[gid] = subtype_counts(events, LEFT_SCORES)
        all_right[gid] = subtype_counts(events, RIGHT_SCORES)

    def print_table(label, score_list, data):
        print(f"\n=== {label} subtype breakdown ===")
        header = f"{'game':>6}" + "".join(f"{s:>18}" for s in score_list) + f"{'total':>8}"
        print(header)
        for gid in games:
            counts = data.get(gid, {})
            row = f"{gid:>6}" + "".join(f"{counts.get(s, 0):>18}" for s in score_list)
            row += f"{sum(counts.values()):>8}"
            print(row)

    print_table("LEFT (label=1)", LEFT_SCORES, all_left)
    print_table("RIGHT (label=2)", RIGHT_SCORES, all_right)

    print("\nLook at game_1's row vs games 2-5's rows for LEFT specifically:")
    print("if game_1 is dominated by a subtype that's rare/absent in 2-5 (or vice versa),")
    print("that's a plausible reason the model's Left signal doesn't transfer to game_1.")


if __name__ == "__main__":
    main()