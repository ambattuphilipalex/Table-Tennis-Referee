LEFT_SCORES = ["left_winner", "right_out", "right_net", "right_miss", "right_not_hitting"]
RIGHT_SCORES = ["right_winner", "left_out", "left_net", "left_miss", "left_not_hitting"]

LEFT_SCORES_FULL = LEFT_SCORES + ["right_double_bounce"]
RIGHT_SCORES_FULL = RIGHT_SCORES + ["left_double_bounce"]


def side_of(event_str, full=False):
    """-> 1 (left scores), 2 (right scores) or 0 (not a rally ending)."""
    left = LEFT_SCORES_FULL if full else LEFT_SCORES
    right = RIGHT_SCORES_FULL if full else RIGHT_SCORES
    if any(s in event_str for s in left):
        return 1
    if any(s in event_str for s in right):
        return 2
    return 0
