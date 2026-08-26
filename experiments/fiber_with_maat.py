"""Compare Base, Manual MAAT, and FIBER-selected MAAT on DataLaw.
"""

import argparse
import copy
import sys
import time
from pathlib import Path
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Repository paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_20.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maat"

from fiber.fairness_metrics import measure_final_score


# ============================================================
# Experiment configuration
# ============================================================

TARGET_COLUMN = "pass_bar"


MANUAL_ATTRIBUTE_REQUESTS = [
    "full time",
    "sex",
]


AUDIT_ATTRIBUTE_REQUESTS = [
    "full time",
    "sex",
]

REPS = 20
TEST_RATIO = 0.20
RANDOM_STATE = 42

# FIBER settings
TOP_N = 2
MIN_SCORE = 0.5
ALPHA = 1.0
CANDIDATE_MAX_CARDINALITY = 10

W_BIAS = 0.14
W_EVEN = 0.51
W_REL = 0.35


PERFORMANCE_AVERAGE = "binary"


DI_OUTPUT_MODE = "distance"


MULTICLASS_PROTECTED_MODE = "highest_rate_vs_rest"


PRIVILEGED_VALUE_OVERRIDES = {}


# ============================================================
# General helpers
# ============================================================


def canonical_column_name(name: str) -> str:
    """
    Normalize a column name for case-, space-, underscore-, and
    punctuation-insensitive matching.
    """

    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def resolve_column_name(columns, requested_name: str) -> str:
    """
    Resolve a requested name such as "full time" to the actual CSV column,
    for example "fulltime" or "full_time".
    """

    requested_key = canonical_column_name(requested_name)

    matches = [
        column
        for column in columns
        if canonical_column_name(column) == requested_key
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous request {requested_name!r}; matches: {matches}"
        )

    raise ValueError(
        f"Could not find a column matching {requested_name!r}. "
        f"Available columns: {list(columns)}"
    )


def ensure_binary_target(df: pd.DataFrame, target_col: str) -> None:
    """Require the target to be encoded as binary 0/1."""

    unique_values = set(
        df[target_col].dropna().astype(int).unique().tolist()
    )

    if not unique_values.issubset({0, 1}) or len(unique_values) != 2:
        raise ValueError(
            f"{target_col!r} must contain exactly binary values 0 and 1; "
            f"found {sorted(unique_values)}."
        )

    df[target_col] = df[target_col].astype(int)


def mean_value(values) -> float:
    """Mean across repetitions, with no standard-deviation display."""

    return float(np.nanmean(np.asarray(values, dtype=float)))


def append_metrics(storage: dict, current: dict) -> None:
    """Append a dictionary of current metric values to list storage."""

    for metric_name in storage:
        storage[metric_name].append(current[metric_name])


# ============================================================
# FIBER attribute detection
# ============================================================


def ssp_from_counts(counts) -> float:
    """Sum of squared proportions; larger values indicate more unevenness."""

    counts = np.asarray(counts, dtype=float)
    total = counts.sum()

    if total <= 0:
        return 0.0

    proportions = counts / total
    return float(np.sum(proportions ** 2))


def detect_fiber_attributes(
    train_df: pd.DataFrame,
    target_col: str,
    detector,
    top_n: int = TOP_N,
    min_score: float = MIN_SCORE,
    alpha: float = ALPHA,
    candidate_max_cardinality: int = CANDIDATE_MAX_CARDINALITY,
):
    """
    Select exactly top_n FIBER attributes using training data only.

    This avoids using test data for attribute selection and keeps the number
    of selected attributes equal to the manual two-attribute baseline.
    """

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int)

    fitted_detector = copy.deepcopy(detector)
    fitted_detector.fit(X, y)

    y_pred = fitted_detector.predict(X)
    cm = confusion_matrix(y, y_pred)

    importances = getattr(
        fitted_detector,
        "feature_importances_",
        np.zeros(X.shape[1], dtype=float),
    )

    rows = []

    for column in X.columns:
        cardinality = X[column].nunique(dropna=True)

        if cardinality > candidate_max_cardinality:
            continue

        feature_ssp = ssp_from_counts(
            X[column].value_counts(dropna=False).values
        )

        grouped = (
            train_df.groupby(column, dropna=False)[target_col]
            .value_counts()
            .unstack(fill_value=0)
        )

        target_ssp_values = [
            ssp_from_counts(grouped.loc[group].values)
            for group in grouped.index
        ]

        target_ssp = (
            float(np.mean(target_ssp_values))
            if target_ssp_values
            else 0.0
        )

        ssp_raw = 0.5 * (feature_ssp + target_ssp)

        di_raw = float(
            measure_final_score(
                train_df,
                y_pred,
                cm,
                X,
                y,
                X,
                y,
                column,
                "DI",
            )
        )

        spd_raw = abs(
            float(
                measure_final_score(
                    train_df,
                    y_pred,
                    cm,
                    X,
                    y,
                    X,
                    y,
                    column,
                    "SPD",
                )
            )
        )

        feature_index = list(X.columns).index(column)
        importance_raw = (
            float(importances[feature_index])
            if len(importances) == X.shape[1]
            else 0.0
        )

        rows.append(
            {
                "Attribute": column,
                "DI Raw": di_raw,
                "SPD Raw": spd_raw,
                "SSP Raw": ssp_raw,
                "Importance Raw": importance_raw,
            }
        )

    if not rows:
        raise ValueError(
            "FIBER found no candidate attributes. Check "
            "CANDIDATE_MAX_CARDINALITY and the dataset columns."
        )

    scores = pd.DataFrame(rows)

    epsilon = 1e-12
    median_di = max(
        float(np.median(scores["DI Raw"].values)),
        epsilon,
    )
    median_spd = max(
        float(np.median(scores["SPD Raw"].values)),
        epsilon,
    )

    scores["DI Calibrated"] = (
        scores["DI Raw"]
        / (scores["DI Raw"] + alpha * median_di + epsilon)
    )

    scores["SPD Calibrated"] = (
        scores["SPD Raw"]
        / (scores["SPD Raw"] + alpha * median_spd + epsilon)
    )

    scores["Bias Score"] = 0.5 * (
        scores["DI Calibrated"]
        + scores["SPD Calibrated"]
    )

    scores["Total Score"] = (
        W_BIAS * scores["Bias Score"]
        + W_EVEN * scores["SSP Raw"]
        + W_REL * scores["Importance Raw"]
    )

    scores = (
        scores.sort_values("Total Score", ascending=False)
        .reset_index(drop=True)
    )

    selected = scores.loc[
        scores["Total Score"] >= min_score,
        "Attribute",
    ].tolist()[:top_n]

    # Fall back to the ranking so every run selects exactly top_n attributes.
    if len(selected) < top_n:
        for attribute in scores["Attribute"].tolist():
            if attribute not in selected:
                selected.append(attribute)

            if len(selected) == top_n:
                break

    if len(selected) != top_n:
        raise ValueError(
            f"FIBER could select only {len(selected)} attributes, "
            f"but TOP_N={top_n}."
        )

    score_display = [
        (row["Attribute"], round(float(row["Total Score"]), 4))
        for _, row in scores.iterrows()
    ]

    print(f"FIBER selected attributes: {selected}")
    print(f"FIBER scores: {score_display}")

    return selected, scores


# ============================================================
# MAAT WAE data debugging
# ============================================================


def choose_privileged_value(
    train_df: pd.DataFrame,
    protected_attribute: str,
    target_col: str,
):
    """Choose the group treated as privileged for MAAT's WAE adjustment.

    Unless explicitly overridden, the category with the highest favorable
    target rate in the current training split is used. For multi-category
    attributes, that category is compared with all remaining categories.
    """

    values = train_df[protected_attribute].dropna().unique().tolist()

    if len(values) < 2:
        raise ValueError(
            f"Protected attribute {protected_attribute!r} has fewer than "
            "two observed values in this training split."
        )

    if protected_attribute in PRIVILEGED_VALUE_OVERRIDES:
        privileged_value = PRIVILEGED_VALUE_OVERRIDES[protected_attribute]

        if privileged_value not in values:
            raise ValueError(
                f"Configured privileged value {privileged_value!r} is not "
                f"present for {protected_attribute!r}."
            )

        return privileged_value

    favorable_rates = (
        train_df.groupby(protected_attribute)[target_col]
        .mean()
        .sort_values(ascending=False)
    )

    privileged_value = favorable_rates.index[0]

    if len(values) > 2:
        if MULTICLASS_PROTECTED_MODE != "highest_rate_vs_rest":
            raise ValueError(
                "Only MULTICLASS_PROTECTED_MODE='highest_rate_vs_rest' "
                "is currently supported."
            )

        warnings.warn(
            f"{protected_attribute!r} has {len(values)} values. Original "
            "MAAT assumes a binary protected attribute. This run uses the "
            f"highest-favorable-rate category {privileged_value!r} as the "
            "privileged group and all other categories as unprivileged.",
            RuntimeWarning,
        )

    return privileged_value


def _best_integer_removals(
    pf: int,
    pu: int,
    uf: int,
    uu: int,
):
    """Find integer PF/UU removals that best equalize favorable rates.

    The search also penalizes changes to the original privileged-to-
    unprivileged group-size ratio and prefers fewer removals when tied.
    """

    privileged_total = pf + pu
    unprivileged_total = uf + uu

    if privileged_total == 0 or unprivileged_total == 0:
        return 0, 0

    a_real = pf - (privileged_total * uf / unprivileged_total)
    b_real = (
        unprivileged_total * pf / privileged_total
        - uf
    )

    a_real = float(np.clip(a_real, 0.0, float(pf)))
    b_real = float(np.clip(b_real, 0.0, float(uu)))

    candidate_a = {
        int(np.floor(a_real)),
        int(np.ceil(a_real)),
        int(np.round(a_real)),
    }
    candidate_b = {
        int(np.floor(b_real)),
        int(np.ceil(b_real)),
        int(np.round(b_real)),
    }


    candidate_a = {
        int(np.clip(value + delta, 0, pf))
        for value in candidate_a
        for delta in range(-3, 4)
    }
    candidate_b = {
        int(np.clip(value + delta, 0, uu))
        for value in candidate_b
        for delta in range(-3, 4)
    }

    original_group_ratio = (
        privileged_total / unprivileged_total
    )

    best = (0, 0)
    best_score = float("inf")

    for a_remove in candidate_a:
        for b_remove in candidate_b:
            new_pf = pf - a_remove
            new_uu = uu - b_remove

            new_privileged_total = new_pf + pu
            new_unprivileged_total = uf + new_uu

            if (
                new_privileged_total <= 0
                or new_unprivileged_total <= 0
            ):
                continue

            privileged_rate = new_pf / new_privileged_total
            unprivileged_rate = uf / new_unprivileged_total
            new_group_ratio = (
                new_privileged_total / new_unprivileged_total
            )

            rate_residual = abs(
                privileged_rate - unprivileged_rate
            )
            ratio_residual = abs(
                new_group_ratio - original_group_ratio
            )

            score = rate_residual + ratio_residual

            # Prefer fewer removals when residuals tie.
            tie_break = a_remove + b_remove
            current_key = (score, tie_break)
            best_key = (best_score, best[0] + best[1])

            if current_key < best_key:
                best_score = score
                best = (a_remove, b_remove)

    return best


def debug_training_data_wae(
    train_df: pd.DataFrame,
    protected_attribute: str,
    target_col: str,
    random_state: int,
):
    """Apply the MAAT WAE-style row-removal adjustment for one attribute.

    Rows are removed only from privileged/favorable (PF) and
    unprivileged/unfavorable (UU) cells using deterministic random sampling.
    The returned diagnostic record reports the chosen privileged value and
    row counts before and after the adjustment.
    """

    privileged_value = choose_privileged_value(
        train_df,
        protected_attribute,
        target_col,
    )

    temp_group = (
        train_df[protected_attribute] == privileged_value
    ).astype(int)

    privileged_favorable_mask = (
        (temp_group == 1)
        & (train_df[target_col] == 1)
    )
    privileged_unfavorable_mask = (
        (temp_group == 1)
        & (train_df[target_col] == 0)
    )
    unprivileged_favorable_mask = (
        (temp_group == 0)
        & (train_df[target_col] == 1)
    )
    unprivileged_unfavorable_mask = (
        (temp_group == 0)
        & (train_df[target_col] == 0)
    )

    pf = int(privileged_favorable_mask.sum())
    pu = int(privileged_unfavorable_mask.sum())
    uf = int(unprivileged_favorable_mask.sum())
    uu = int(unprivileged_unfavorable_mask.sum())

    if min(pf + pu, uf + uu) == 0:
        warnings.warn(
            f"Cannot WAE-debug {protected_attribute!r}: one binary group "
            "is empty. The original training set is used.",
            RuntimeWarning,
        )
        return train_df.copy(), {
            "Attribute": protected_attribute,
            "Privileged Value": privileged_value,
            "PF Removed": 0,
            "UU Removed": 0,
            "Rows Before": len(train_df),
            "Rows After": len(train_df),
        }

    privileged_rate = pf / max(pf + pu, 1)
    unprivileged_rate = uf / max(uf + uu, 1)

    # By construction, the privileged value should have the higher favorable
    # rate. An override may violate that assumption, so warn and keep data.
    if privileged_rate < unprivileged_rate:
        warnings.warn(
            f"For {protected_attribute!r}, the configured privileged group "
            "has a lower favorable rate than the unprivileged group. MAAT's "
            "PF/UU removal direction is not applicable; original data used.",
            RuntimeWarning,
        )
        return train_df.copy(), {
            "Attribute": protected_attribute,
            "Privileged Value": privileged_value,
            "PF Removed": 0,
            "UU Removed": 0,
            "Rows Before": len(train_df),
            "Rows After": len(train_df),
        }

    a_remove, b_remove = _best_integer_removals(
        pf=pf,
        pu=pu,
        uf=uf,
        uu=uu,
    )

    rng = np.random.default_rng(random_state)

    pf_indices = train_df.index[
        privileged_favorable_mask
    ].to_numpy()
    uu_indices = train_df.index[
        unprivileged_unfavorable_mask
    ].to_numpy()

    remove_pf_indices = (
        rng.choice(pf_indices, size=a_remove, replace=False)
        if a_remove > 0
        else np.array([], dtype=pf_indices.dtype)
    )

    remove_uu_indices = (
        rng.choice(uu_indices, size=b_remove, replace=False)
        if b_remove > 0
        else np.array([], dtype=uu_indices.dtype)
    )

    remove_indices = np.concatenate(
        [remove_pf_indices, remove_uu_indices]
    )

    debugged_train_df = (
        train_df.drop(index=remove_indices)
        .reset_index(drop=True)
    )

    debug_info = {
        "Attribute": protected_attribute,
        "Privileged Value": privileged_value,
        "PF Removed": int(a_remove),
        "UU Removed": int(b_remove),
        "Rows Before": int(len(train_df)),
        "Rows After": int(len(debugged_train_df)),
    }

    return debugged_train_df, debug_info


# ============================================================
# MAAT model construction
# ============================================================


def train_performance_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_template,
):
    """Train MAAT's performance model on the original training data."""

    feature_columns = [
        column
        for column in train_df.columns
        if column != target_col
    ]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(
        train_df[feature_columns]
    )
    X_test = scaler.transform(
        test_df[feature_columns]
    )

    y_train = train_df[target_col].astype(int).to_numpy()

    model = copy.deepcopy(model_template)
    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)

    return model, scaler, probability


def train_fairness_model_probability(
    original_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protected_attribute: str,
    target_col: str,
    model_template,
    common_scaler: MinMaxScaler,
    random_state: int,
):
    """
    Train one separate MAAT fairness model for one selected attribute.

    The common scaler is fitted on the original training data and reused for
    every fairness model so all models receive the same feature representation.
    """

    debugged_train_df, debug_info = debug_training_data_wae(
        train_df=original_train_df,
        protected_attribute=protected_attribute,
        target_col=target_col,
        random_state=random_state,
    )

    feature_columns = [
        column
        for column in original_train_df.columns
        if column != target_col
    ]

    X_train_fair = common_scaler.transform(
        debugged_train_df[feature_columns]
    )
    X_test = common_scaler.transform(
        test_df[feature_columns]
    )

    y_train_fair = (
        debugged_train_df[target_col]
        .astype(int)
        .to_numpy()
    )

    fairness_model = copy.deepcopy(model_template)
    fairness_model.fit(X_train_fair, y_train_fair)

    probability = fairness_model.predict_proba(X_test)

    return probability, debug_info


def maat_ensemble_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    selected_attributes: list,
    performance_model_template,
    fairness_model_template,
    performance_probability: np.ndarray,
    common_scaler: MinMaxScaler,
    random_state: int,
):
    """
    Build one fairness model per selected attribute and average all vectors.

    With two selected attributes, the final probability is:
        (performance probability + fairness probability 1
         + fairness probability 2) / 3
    """

    probability_vectors = [performance_probability]
    debug_information = []

    for attribute_index, protected_attribute in enumerate(
        selected_attributes
    ):
        fairness_probability, debug_info = (
            train_fairness_model_probability(
                original_train_df=train_df,
                test_df=test_df,
                protected_attribute=protected_attribute,
                target_col=target_col,
                model_template=fairness_model_template,
                common_scaler=common_scaler,
                random_state=(
                    random_state * 100
                    + attribute_index
                    + 1
                ),
            )
        )

        probability_vectors.append(fairness_probability)
        debug_information.append(debug_info)

    final_probability = np.mean(
        np.stack(probability_vectors, axis=0),
        axis=0,
    )

    final_prediction = np.argmax(
        final_probability,
        axis=1,
    ).astype(int)

    return final_prediction, final_probability, debug_information


# ============================================================
# Evaluation
# ============================================================


def calculate_performance(y_true, y_pred) -> dict:
    """Calculate target-prediction performance."""

    metric_kwargs = {"zero_division": 0}

    if PERFORMANCE_AVERAGE == "macro":
        metric_kwargs["average"] = "macro"
    elif PERFORMANCE_AVERAGE == "binary":
        metric_kwargs["average"] = "binary"
    else:
        raise ValueError(
            "PERFORMANCE_AVERAGE must be 'binary' or 'macro'."
        )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            **metric_kwargs,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            **metric_kwargs,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            **metric_kwargs,
        ),
    }


def di_to_zero_ideal(di_value: float) -> float:
    """Convert DI to a disparity where zero is the ideal value."""

    di_value = float(di_value)

    if not np.isfinite(di_value):
        return np.nan

    if DI_OUTPUT_MODE == "distance":
        return abs(di_value)

    if DI_OUTPUT_MODE == "ratio":
        if di_value <= 0:
            return np.nan

        return 1.0 - min(
            di_value,
            1.0 / di_value,
        )

    raise ValueError(
        "DI_OUTPUT_MODE must be 'distance' or 'ratio'."
    )


def calculate_attribute_fairness(
    train_df_original: pd.DataFrame,
    test_df_original: pd.DataFrame,
    target_col: str,
    y_pred,
    audit_attribute: str,
) -> dict:
    """
    Audit pass_bar predictions by one fixed true attribute.

    Original, unmodified group values are used for every method.
    """

    feature_columns = [
        column
        for column in train_df_original.columns
        if column != target_col
    ]

    X_train_original = train_df_original[feature_columns]
    X_test_original = test_df_original[feature_columns]

    y_train = train_df_original[target_col].astype(int)
    y_test = test_df_original[target_col].astype(int)

    cm = confusion_matrix(y_test, y_pred)

    aod = float(
        measure_final_score(
            test_df_original,
            y_pred,
            cm,
            X_train_original,
            y_train,
            X_test_original,
            y_test,
            audit_attribute,
            "aod",
        )
    )

    eod = float(
        measure_final_score(
            test_df_original,
            y_pred,
            cm,
            X_train_original,
            y_train,
            X_test_original,
            y_test,
            audit_attribute,
            "eod",
        )
    )

    spd = float(
        measure_final_score(
            test_df_original,
            y_pred,
            cm,
            X_train_original,
            y_train,
            X_test_original,
            y_test,
            audit_attribute,
            "SPD",
        )
    )

    di = float(
        measure_final_score(
            test_df_original,
            y_pred,
            cm,
            X_train_original,
            y_train,
            X_test_original,
            y_test,
            audit_attribute,
            "DI",
        )
    )

    return {
        "AOD": abs(aod),
        "EOD": abs(eod),
        "SPD": abs(spd),
        "DI": di_to_zero_ideal(di),
    }


def calculate_macro_fairness(
    train_df_original: pd.DataFrame,
    test_df_original: pd.DataFrame,
    target_col: str,
    y_pred,
    audit_attributes: list,
):
    """Equal-weight macro-average over the fixed audit attributes."""

    fairness_by_attribute = {
        attribute: calculate_attribute_fairness(
            train_df_original=train_df_original,
            test_df_original=test_df_original,
            target_col=target_col,
            y_pred=y_pred,
            audit_attribute=attribute,
        )
        for attribute in audit_attributes
    }

    macro = {
        "Macro AOD": float(
            np.nanmean(
                [
                    fairness_by_attribute[attr]["AOD"]
                    for attr in audit_attributes
                ]
            )
        ),
        "Macro EOD": float(
            np.nanmean(
                [
                    fairness_by_attribute[attr]["EOD"]
                    for attr in audit_attributes
                ]
            )
        ),
        "Macro SPD": float(
            np.nanmean(
                [
                    fairness_by_attribute[attr]["SPD"]
                    for attr in audit_attributes
                ]
            )
        ),
        "Macro DI": float(
            np.nanmean(
                [
                    fairness_by_attribute[attr]["DI"]
                    for attr in audit_attributes
                ]
            )
        ),
    }

    return macro, fairness_by_attribute


# ============================================================
# Main direct-comparison experiment
# ============================================================

def parse_args():
    """Parse command-line options for a reproducible experiment run."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Base, Manual MAAT, and FIBER MAAT on a binary "
            "tabular dataset."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV dataset path (default: data/test_20.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for result CSV files.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=REPS,
        help=f"Number of repeated train/test splits (default: {REPS}).",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be at least 1.")

    data_path = args.data.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    print(f"Loaded DataLaw dataset with shape: {df.shape}")
    print(df.head())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset is missing target column {TARGET_COLUMN!r}. "
            f"Available columns: {df.columns.tolist()}"
        )

    manual_attributes = [
        resolve_column_name(df.columns, requested_name)
        for requested_name in MANUAL_ATTRIBUTE_REQUESTS
    ]

    audit_attributes = [
        resolve_column_name(df.columns, requested_name)
        for requested_name in AUDIT_ATTRIBUTE_REQUESTS
    ]

    if len(set(manual_attributes)) != 2:
        raise ValueError(
            f"Manual requests resolved to duplicate columns: "
            f"{manual_attributes}"
        )

    if len(set(audit_attributes)) != 2:
        raise ValueError(
            f"Audit requests resolved to duplicate columns: "
            f"{audit_attributes}"
        )

    print(f"Manual MAAT attributes: {manual_attributes}")
    print(f"Fixed fairness-audit attributes: {audit_attributes}")

    data_clean = df.dropna().reset_index(drop=True)
    ensure_binary_target(data_clean, TARGET_COLUMN)

    performance_model_template = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        bootstrap=False,
    )

    fairness_model_template = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        bootstrap=False,
    )

    detector_template = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        bootstrap=False,
    )

    methods = [
        "Base Model",
        "Manual-MAAT",
        "FIBER-MAAT",
    ]

    performance_storage = {
        method: {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1": [],
        }
        for method in methods
    }

    macro_storage = {
        method: {
            "Macro AOD": [],
            "Macro EOD": [],
            "Macro SPD": [],
            "Macro DI": [],
        }
        for method in methods
    }

    # Detailed results are retained for diagnostics, although the main table
    # reports only the macro-average.
    detailed_storage = {
        method: {
            attribute: {
                "AOD": [],
                "EOD": [],
                "SPD": [],
                "DI": [],
            }
            for attribute in audit_attributes
        }
        for method in methods
    }

    selection_counter = Counter()
    selection_pairs = Counter()
    debugging_records = []

    for seed in range(args.reps):
        start_time = time.time()

        train_df, test_df = train_test_split(
            data_clean,
            test_size=TEST_RATIO,
            random_state=seed,
            stratify=data_clean[TARGET_COLUMN],
            shuffle=True,
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

        # ----------------------------------------------------
        # FIBER selects exactly two fairness-model attributes
        # from the training split only.
        # ----------------------------------------------------

        fiber_attributes, _ = detect_fiber_attributes(
            train_df=train_df,
            target_col=TARGET_COLUMN,
            detector=detector_template,
            top_n=TOP_N,
            min_score=MIN_SCORE,
            alpha=ALPHA,
            candidate_max_cardinality=(
                CANDIDATE_MAX_CARDINALITY
            ),
        )

        selection_counter.update(fiber_attributes)
        selection_pairs.update([tuple(fiber_attributes)])

        # ----------------------------------------------------
        # Train the single shared performance model.
        # ----------------------------------------------------

        (
            performance_model,
            common_scaler,
            performance_probability,
        ) = train_performance_model(
            train_df=train_df,
            test_df=test_df,
            target_col=TARGET_COLUMN,
            model_template=performance_model_template,
        )

        y_pred_base = np.argmax(
            performance_probability,
            axis=1,
        ).astype(int)

        # ----------------------------------------------------
        # Manual MAAT: one fairness model for each manually
        # selected attribute, then average three probability
        # vectors (one performance + two fairness).
        # ----------------------------------------------------

        (
            y_pred_manual,
            _,
            manual_debug_info,
        ) = maat_ensemble_prediction(
            train_df=train_df,
            test_df=test_df,
            target_col=TARGET_COLUMN,
            selected_attributes=manual_attributes,
            performance_model_template=(
                performance_model_template
            ),
            fairness_model_template=(
                fairness_model_template
            ),
            performance_probability=(
                performance_probability
            ),
            common_scaler=common_scaler,
            random_state=seed,
        )

        # ----------------------------------------------------
        # FIBER MAAT: same MAAT mechanism, but fairness-model
        # attributes come from FIBER.
        # ----------------------------------------------------

        (
            y_pred_fiber,
            _,
            fiber_debug_info,
        ) = maat_ensemble_prediction(
            train_df=train_df,
            test_df=test_df,
            target_col=TARGET_COLUMN,
            selected_attributes=fiber_attributes,
            performance_model_template=(
                performance_model_template
            ),
            fairness_model_template=(
                fairness_model_template
            ),
            performance_probability=(
                performance_probability
            ),
            common_scaler=common_scaler,
            random_state=seed + 1000,
        )

        prediction_by_method = {
            "Base Model": y_pred_base,
            "Manual-MAAT": y_pred_manual,
            "FIBER-MAAT": y_pred_fiber,
        }

        # ----------------------------------------------------
        # Identical evaluation for all methods.
        # ----------------------------------------------------

        for method_name, y_pred in prediction_by_method.items():
            current_performance = calculate_performance(
                y_test,
                y_pred,
            )

            current_macro, current_detailed = (
                calculate_macro_fairness(
                    train_df_original=train_df,
                    test_df_original=test_df,
                    target_col=TARGET_COLUMN,
                    y_pred=y_pred,
                    audit_attributes=audit_attributes,
                )
            )

            append_metrics(
                performance_storage[method_name],
                current_performance,
            )

            append_metrics(
                macro_storage[method_name],
                current_macro,
            )

            for audit_attribute in audit_attributes:
                for metric_name in detailed_storage[
                    method_name
                ][audit_attribute]:
                    detailed_storage[
                        method_name
                    ][audit_attribute][metric_name].append(
                        current_detailed[
                            audit_attribute
                        ][metric_name]
                    )

        for item in manual_debug_info:
            debugging_records.append(
                {
                    "Run": seed + 1,
                    "Method": "Manual-MAAT",
                    **item,
                }
            )

        for item in fiber_debug_info:
            debugging_records.append(
                {
                    "Run": seed + 1,
                    "Method": "FIBER-MAAT",
                    **item,
                }
            )

        print(
            f"Round {seed + 1:02d}/{args.reps} finished in "
            f"{time.time() - start_time:.2f} seconds."
        )

    # ========================================================
    # Main table: direct macro-average comparison
    # ========================================================

    rows = []

    masking_descriptions = {
        "Base Model": "None",
        "Manual-MAAT": "; ".join(manual_attributes),
        "FIBER-MAAT": "FIBER Top-2 per split",
    }

    for method_name in methods:
        rows.append(
            {
                "Method": method_name,
                "Fairness-Model Attributes": (
                    masking_descriptions[method_name]
                ),
                "Accuracy": mean_value(
                    performance_storage[
                        method_name
                    ]["Accuracy"]
                ),
                "Precision": mean_value(
                    performance_storage[
                        method_name
                    ]["Precision"]
                ),
                "Recall": mean_value(
                    performance_storage[
                        method_name
                    ]["Recall"]
                ),
                "F1": mean_value(
                    performance_storage[
                        method_name
                    ]["F1"]
                ),
                "Macro AOD": mean_value(
                    macro_storage[
                        method_name
                    ]["Macro AOD"]
                ),
                "Macro EOD": mean_value(
                    macro_storage[
                        method_name
                    ]["Macro EOD"]
                ),
                "Macro SPD": mean_value(
                    macro_storage[
                        method_name
                    ]["Macro SPD"]
                ),
                "Macro DI": mean_value(
                    macro_storage[
                        method_name
                    ]["Macro DI"]
                ),
            }
        )

    results_table = pd.DataFrame(rows)

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Macro AOD",
        "Macro EOD",
        "Macro SPD",
        "Macro DI",
    ]

    results_table[numeric_columns] = (
        results_table[numeric_columns].round(4)
    )

    print("\n" + "=" * 140)
    print("DATALAW: MANUAL-MAAT VS FIBER-MAAT")
    print(
        "Fixed audit attributes: "
        + " and ".join(audit_attributes)
    )
    print(
        "Lower is better for Macro AOD, Macro EOD, "
        "Macro SPD, and Macro DI."
    )
    print("=" * 140)
    print(results_table.to_string(index=False))

    # ========================================================
    # FIBER selection frequency
    # ========================================================

    selection_table = pd.DataFrame(
        [
            {
                "Attribute": attribute,
                "Selected Runs": count,
                "Selection Rate": count / args.reps,
            }
            for attribute, count
            in selection_counter.most_common()
        ]
    )

    if not selection_table.empty:
        selection_table["Selection Rate"] = (
            selection_table["Selection Rate"].round(4)
        )

        print("\n" + "=" * 90)
        print("FIBER ATTRIBUTE-SELECTION FREQUENCY")
        print("=" * 90)
        print(selection_table.to_string(index=False))

    pair_table = pd.DataFrame(
        [
            {
                "Selected Pair": ", ".join(pair),
                "Runs": count,
                "Rate": count / args.reps,
            }
            for pair, count in selection_pairs.most_common()
        ]
    )

    if not pair_table.empty:
        pair_table["Rate"] = pair_table["Rate"].round(4)

        print("\n" + "=" * 90)
        print("FIBER SELECTED-PAIR FREQUENCY")
        print("=" * 90)
        print(pair_table.to_string(index=False))

    # ========================================================
    # Optional detailed audit table
    # ========================================================

    detailed_rows = []

    for method_name in methods:
        for audit_attribute in audit_attributes:
            detailed_rows.append(
                {
                    "Method": method_name,
                    "Audit Attribute": audit_attribute,
                    "AOD": mean_value(
                        detailed_storage[
                            method_name
                        ][audit_attribute]["AOD"]
                    ),
                    "EOD": mean_value(
                        detailed_storage[
                            method_name
                        ][audit_attribute]["EOD"]
                    ),
                    "SPD": mean_value(
                        detailed_storage[
                            method_name
                        ][audit_attribute]["SPD"]
                    ),
                    "DI": mean_value(
                        detailed_storage[
                            method_name
                        ][audit_attribute]["DI"]
                    ),
                }
            )

    detailed_table = pd.DataFrame(detailed_rows)
    detailed_table[["AOD", "EOD", "SPD", "DI"]] = (
        detailed_table[["AOD", "EOD", "SPD", "DI"]]
        .round(4)
    )

    print("\n" + "=" * 110)
    print("DETAILED FIXED-ATTRIBUTE FAIRNESS AUDIT")
    print("=" * 110)
    print(detailed_table.to_string(index=False))

    # Save main results and diagnostics.
    results_table.to_csv(
        output_directory / "DataLaw_MAAT_macro_comparison.csv",
        index=False,
    )
    detailed_table.to_csv(
        output_directory / "DataLaw_MAAT_detailed_audit.csv",
        index=False,
    )
    selection_table.to_csv(
        output_directory / "DataLaw_FIBER_selection_frequency.csv",
        index=False,
    )
    pair_table.to_csv(
        output_directory / "DataLaw_FIBER_selected_pair_frequency.csv",
        index=False,
    )

    if debugging_records:
        pd.DataFrame(debugging_records).to_csv(
            output_directory / "DataLaw_MAAT_WAE_debugging_log.csv",
            index=False,
        )

    print(
        "\nSaved MAAT result files to: "
        f"{output_directory}"
    )
