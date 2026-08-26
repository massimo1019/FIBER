import numpy as np
import pandas as pd

from fiber.fairness_metrics import calculate_group_fairness, measure_final_score


def test_zero_disparity_when_predictions_match_across_groups():
    df = pd.DataFrame({"group": [0, 0, 1, 1]})
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    assert calculate_group_fairness(df, y_pred, y_true, "group", "SPD") == 0.0
    assert calculate_group_fairness(df, y_pred, y_true, "group", "eod") == 0.0
    assert calculate_group_fairness(df, y_pred, y_true, "group", "aod") == 0.0


def test_spd_detects_group_selection_rate_difference():
    df = pd.DataFrame({"group": [0, 0, 1, 1]})
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])

    assert calculate_group_fairness(df, y_pred, y_true, "group", "SPD") > 0.9


def test_public_entry_point_matches_direct_calculation():
    df = pd.DataFrame({"group": [0, 0, 1, 1]})
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])

    direct = calculate_group_fairness(df, y_pred, y_true, "group", "SPD")
    wrapped = measure_final_score(
        df, y_pred, None, None, None, None, y_true, "group", "SPD"
    )
    assert wrapped == direct
