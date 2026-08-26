"""Fairness metrics used by the FIBER experiments.

For a candidate audit attribute with two or more categories, each category is
compared with the rest of the data (One-vs-Rest), and the category-level
fairness scores are averaged.

All returned metrics are disparities for which smaller values are better:
AOD, EOD, and SPD are absolute differences; DI is represented as
``abs(1 - selection_rate_group / selection_rate_rest)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_group_split(
    test_df: pd.DataFrame,
    attribute: str,
    group_value: object,
) -> pd.Series:
    """Return 1 for rows in ``group_value`` and 0 for all other rows."""

    return (test_df[attribute] == group_value).astype(int)


def get_confusion_elements(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return true positives, true negatives, false positives, and false negatives."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def calculate_group_fairness(
    test_df: pd.DataFrame,
    y_pred,
    y_true,
    attribute: str,
    metric: str,
) -> float:
    """Compute a One-vs-Rest fairness disparity and average across groups.

    Parameters
    ----------
    test_df:
        Original test rows containing the true audit-attribute values.
    y_pred:
        Binary target predictions.
    y_true:
        Binary true target labels.
    attribute:
        Column used to define the groups being audited.
    metric:
        One of ``"aod"``, ``"eod"``, ``"SPD"``, or ``"DI"``.

    Returns
    -------
    float
        Mean absolute group-vs-rest disparity, rounded to four decimals.
    """

    if attribute not in test_df.columns:
        raise ValueError(f"Audit attribute {attribute!r} is not in the test data.")

    supported_metrics = {"aod", "eod", "SPD", "DI"}
    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported fairness metric: {metric!r}. "
            f"Choose from {sorted(supported_metrics)}."
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(test_df) != len(y_true) or len(test_df) != len(y_pred):
        raise ValueError("test_df, y_true, and y_pred must have the same length.")

    groups = test_df[attribute].unique()
    scores: list[float] = []
    epsilon = 1e-6

    for group_value in groups:
        group_mask = binary_group_split(test_df, attribute, group_value).to_numpy(bool)
        rest_mask = ~group_mask

        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]
        tp_g, tn_g, fp_g, fn_g = get_confusion_elements(y_true_group, y_pred_group)

        y_true_rest = y_true[rest_mask]
        y_pred_rest = y_pred[rest_mask]
        tp_r, tn_r, fp_r, fn_r = get_confusion_elements(y_true_rest, y_pred_rest)

        tpr_g = tp_g / (tp_g + fn_g + epsilon)
        fpr_g = fp_g / (fp_g + tn_g + epsilon)
        tpr_r = tp_r / (tp_r + fn_r + epsilon)
        fpr_r = fp_r / (fp_r + tn_r + epsilon)

        if metric == "aod":
            score = 0.5 * (abs(tpr_g - tpr_r) + abs(fpr_g - fpr_r))
        elif metric == "eod":
            score = abs(tpr_g - tpr_r)
        else:
            rate_g = (tp_g + fp_g) / (tp_g + fp_g + tn_g + fn_g + epsilon)
            rate_r = (tp_r + fp_r) / (tp_r + fp_r + tn_r + fn_r + epsilon)
            if metric == "SPD":
                score = abs(rate_g - rate_r)
            else:  # DI
                score = abs(1 - rate_g / (rate_r + epsilon))

        scores.append(float(score))

    return round(float(np.mean(scores)), 4)


def measure_final_score(
    test_df: pd.DataFrame,
    y_pred,
    cm,
    X_train,
    y_train,
    X_test,
    y_test,
    biased_col: str,
    metric: str,
) -> float:
    """Compatibility entry point used by the original experiment scripts.

    Parameters kept for compatibility with the uploaded implementation include
    ``cm``, ``X_train``, ``y_train``, and ``X_test``. The One-vs-Rest metric
    calculation itself uses the original test dataframe, predictions, true
    test labels, the audit column, and the requested metric.
    """

    _ = (cm, X_train, y_train, X_test)  # Explicitly mark compatibility args unused.
    return calculate_group_fairness(test_df, y_pred, y_test, biased_col, metric)
