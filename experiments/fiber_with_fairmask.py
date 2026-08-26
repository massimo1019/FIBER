"""Compare Base, Manual FairMask, and FIBER-selected FairMask on DataLaw.

This script is a repository-ready version of the supplied Colab experiment.
The modeling and fairness-evaluation logic is preserved, while file paths are
made portable and command-line options are added for the dataset, output
directory, and number of repeated train/test splits.
"""

import argparse
import time
import copy
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Repository paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "test_20.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairmask"

from fiber.fairness_metrics import measure_final_score


# ============================================================
# Experiment configuration
# ============================================================

TARGET_COLUMN = "pass_bar"

MANUAL_ATTRIBUTE_REQUESTS = [
    "full time",
    "sex",
]

FIXED_AUDIT_REQUESTS = [
    "full time",
    "sex",
]

# FIBER settings
TOP_N = 2
MIN_SCORE = 0.5
CANDIDATE_MAX_CARDINALITY = 10
ALPHA = 1.0

W_BIAS = 0.14
W_EVEN = 0.51
W_REL = 0.35

# Repeated evaluation settings
REPS = 20
TEST_RATIO = 0.20
RANDOM_STATE = 42
USE_SMOTE_FOR_ATTRIBUTE = True

# Measurenew returns
# DI as distance from 1, where 0 is best.

DI_OUTPUT_MODE = "distance"


# ============================================================
# Column-name
# ============================================================

def canonical_name(name):
    """
    Normalize a column name so aliases such as:
      full time, Full_Time, and fulltime
    can be matched safely.
    """

    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def resolve_column(columns, requested_name):
    """
    Resolve a requested human-readable name to an actual CSV
    column using case-, space-, and underscore-insensitive matching.
    """

    requested_key = canonical_name(requested_name)

    matches = [
        column
        for column in columns
        if canonical_name(column) == requested_key
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous column request '{requested_name}'. "
            f"Matches: {matches}"
        )

    raise ValueError(
        f"Could not find a column matching '{requested_name}'. "
        f"Available columns: {list(columns)}"
    )


# ============================================================
# FIBER scoring
# ============================================================

def ssp_from_counts(counts):
    """Sum of squared proportions; higher means more uneven."""

    counts = np.asarray(counts, dtype=float)
    total = counts.sum()

    if total <= 0:
        return 0.0

    proportions = counts / total
    return float(np.sum(proportions ** 2))


def detect_fiber_attributes(
    train_df,
    target_col,
    classifier,
    top_n=TOP_N,
    min_score=MIN_SCORE,
    alpha=ALPHA,
    candidate_max_cardinality=CANDIDATE_MAX_CARDINALITY,
):
    """
    Run FIBER on the training split only and return exactly top_n
    automatically selected fairness-sensitive masking attributes.
    """

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int)

    detector = copy.deepcopy(classifier)
    detector.fit(X, y)

    y_pred = detector.predict(X)
    cm = confusion_matrix(y, y_pred)

    importances = getattr(
        detector,
        "feature_importances_",
        np.zeros(X.shape[1], dtype=float),
    )

    rows = []

    for column in X.columns:
        if (
            X[column].nunique(dropna=True)
            > candidate_max_cardinality
        ):
            continue

        feature_counts = (
            X[column]
            .value_counts(dropna=False)
            .values
        )
        ssp_feature = ssp_from_counts(feature_counts)

        grouped = (
            train_df.groupby(
                column,
                dropna=False,
            )[target_col]
            .value_counts()
            .unstack(fill_value=0)
        )

        target_ssp_values = [
            ssp_from_counts(grouped.loc[group].values)
            for group in grouped.index
        ]

        ssp_target = (
            float(np.mean(target_ssp_values))
            if target_ssp_values
            else 0.0
        )

        ssp_raw = 0.5 * (
            ssp_feature + ssp_target
        )

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

        column_index = list(X.columns).index(column)
        importance_raw = (
            float(importances[column_index])
            if len(importances) == X.shape[1]
            else 0.0
        )

        rows.append(
            {
                "column": column,
                "di_raw": di_raw,
                "spd_raw": spd_raw,
                "ssp_raw": ssp_raw,
                "importance_raw": importance_raw,
            }
        )

    if not rows:
        raise ValueError(
            "FIBER found no candidate fairness-sensitive attributes. "
            "Check CANDIDATE_MAX_CARDINALITY."
        )

    scores = pd.DataFrame(rows)

    epsilon = 1e-12

    median_di = max(
        float(np.median(scores["di_raw"])),
        epsilon,
    )

    median_spd = max(
        float(np.median(scores["spd_raw"])),
        epsilon,
    )

    scores["di_calibrated"] = (
        scores["di_raw"]
        / (
            scores["di_raw"]
            + alpha * median_di
            + epsilon
        )
    )

    scores["spd_calibrated"] = (
        scores["spd_raw"]
        / (
            scores["spd_raw"]
            + alpha * median_spd
            + epsilon
        )
    )

    scores["bias_score"] = 0.5 * (
        scores["di_calibrated"]
        + scores["spd_calibrated"]
    )

    scores["total_score"] = (
        W_BIAS * scores["bias_score"]
        + W_EVEN * scores["ssp_raw"]
        + W_REL * scores["importance_raw"]
    )

    scores = (
        scores.sort_values(
            "total_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    above_threshold = scores.loc[
        scores["total_score"] >= min_score,
        "column",
    ].tolist()

    selected = above_threshold[:top_n]

    # Always return exactly TOP_N attributes so the automatic
    # method masks the same number as the manual baseline.
    if len(selected) < top_n:
        for column in scores["column"].tolist():
            if column not in selected:
                selected.append(column)

            if len(selected) == top_n:
                break

    selected = selected[:top_n]

    score_display = [
        (
            row["column"],
            round(float(row["total_score"]), 4),
        )
        for _, row in scores.iterrows()
    ]

    print(
        f"FIBER selected masking attributes: "
        f"{selected}"
    )
    print(f"FIBER scores: {score_display}")

    return selected, scores


# ============================================================
# FairMask reconstruction
# ============================================================

def mask_multiple_attributes(
    X_train,
    X_test_original,
    mask_attributes,
    attribute_model,
    use_smote=True,
    random_state=42,
    verbose=False,
):
    """Reconstruct selected attributes from the remaining features.

    A separate attribute model is fitted for each selected column. The true
    test-set values of those columns are then replaced by model predictions,
    which is the FairMask-style masking step used in this experiment.
    """

    missing = [
        attribute
        for attribute in mask_attributes
        if attribute not in X_train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing masking attributes: {missing}"
        )

    non_mask_columns = [
        column
        for column in X_train.columns
        if column not in mask_attributes
    ]

    X_attribute_train = (
        X_train[non_mask_columns]
        .reset_index(drop=True)
    )

    X_attribute_test = (
        X_test_original[non_mask_columns]
        .reset_index(drop=True)
    )

    predicted_attributes = {}

    for attribute_index, attribute in enumerate(
        mask_attributes
    ):
        y_attribute = (
            X_train[attribute]
            .astype(str)
            .reset_index(drop=True)
        )

        model = copy.deepcopy(attribute_model)

        X_fit = X_attribute_train
        y_fit = y_attribute

        minimum_class_count = (
            y_attribute.value_counts().min()
        )

        if (
            use_smote
            and minimum_class_count is not None
            and minimum_class_count >= 6
        ):
            try:
                smote = SMOTE(
                    random_state=(
                        random_state + attribute_index
                    )
                )

                X_fit, y_fit = smote.fit_resample(
                    X_attribute_train,
                    y_attribute,
                )

            except Exception as error:
                if verbose:
                    print(
                        f"SMOTE failed for {attribute}: "
                        f"{error}. Original data used."
                    )

                X_fit = X_attribute_train
                y_fit = y_attribute

        elif verbose and use_smote:
            print(
                f"SMOTE not used for {attribute}; "
                f"minimum class count is "
                f"{minimum_class_count}."
            )

        model.fit(X_fit, y_fit)

        predicted = model.predict(
            X_attribute_test
        )

        predicted = pd.to_numeric(
            pd.Series(predicted),
            errors="raise",
        ).to_numpy()

        predicted_attributes[attribute] = predicted

    X_test_masked = X_test_original.copy()

    for attribute in mask_attributes:
        X_test_masked.loc[:, attribute] = (
            predicted_attributes[attribute]
        )

    return X_test_masked


# ============================================================
# Performance and fairness evaluation
# ============================================================

def calculate_performance(y_true, y_pred):
    """Performance of pass_bar predictions."""

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def di_to_zero_ideal(di_value):
    """
    Convert DI into a disparity where 0 is ideal.
    """

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
    test_df_original,
    y_pred,
    cm,
    X_train,
    y_train,
    X_test_original,
    y_test,
    audit_attribute,
):
    """
    Audit pass_bar predictions using the true, original values
    of one fixed audit attribute.
    """

    aod = float(
        measure_final_score(
            test_df_original,
            y_pred,
            cm,
            X_train,
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
            X_train,
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
            X_train,
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
            X_train,
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
    fairness_by_attribute,
    audit_attributes,
):
    """
    Equal-weight macro-average across the fixed audit set.

    Lower is better for all returned metrics.
    """

    return {
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


def evaluate_predictions(
    test_df_original,
    y_pred,
    X_train,
    y_train,
    X_test_original,
    y_test,
    audit_attributes,
):
    """
    Return performance, attribute-level fairness, and fixed-set
    macro-average fairness for one prediction vector.
    """

    cm = confusion_matrix(y_test, y_pred)

    performance = calculate_performance(
        y_test,
        y_pred,
    )

    fairness_by_attribute = {}

    for audit_attribute in audit_attributes:
        fairness_by_attribute[
            audit_attribute
        ] = calculate_attribute_fairness(
            test_df_original=test_df_original,
            y_pred=y_pred,
            cm=cm,
            X_train=X_train,
            y_train=y_train,
            X_test_original=X_test_original,
            y_test=y_test,
            audit_attribute=audit_attribute,
        )

    macro_fairness = calculate_macro_fairness(
        fairness_by_attribute,
        audit_attributes,
    )

    return (
        performance,
        fairness_by_attribute,
        macro_fairness,
    )


def create_storage():
    """Create list-based storage for one method."""

    return {
        "performance": {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1": [],
        },
        "macro": {
            "Macro AOD": [],
            "Macro EOD": [],
            "Macro SPD": [],
            "Macro DI": [],
        },
        "details": {},
    }


def initialize_detail_storage(
    method_storage,
    audit_attributes,
):
    """Initialize per-attribute fairness metric storage for one method."""

    method_storage["details"] = {
        attribute: {
            "AOD": [],
            "EOD": [],
            "SPD": [],
            "DI": [],
        }
        for attribute in audit_attributes
    }


def store_evaluation(
    storage,
    performance,
    fairness_by_attribute,
    macro_fairness,
):
    """Append one repetition's performance and fairness results."""

    for metric, value in performance.items():
        storage["performance"][metric].append(value)

    for metric, value in macro_fairness.items():
        storage["macro"][metric].append(value)

    for attribute in fairness_by_attribute:
        for metric, value in (
            fairness_by_attribute[attribute].items()
        ):
            storage["details"][attribute][metric].append(
                value
            )


def mean_value(values):
    """Mean across runs, without ± standard deviation."""

    return float(
        np.nanmean(
            np.asarray(values, dtype=float)
        )
    )


# ============================================================
# Main experiment
# ============================================================

def parse_args():
    """Parse command-line options for a reproducible experiment run."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Base, Manual FairMask, and FIBER FairMask on a "
            "binary tabular dataset."
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "DataLaw_FairMask_manual_vs_FIBER_macro.csv"
    selection_path = output_dir / "DataLaw_FIBER_selection_frequency.csv"
    pair_path = output_dir / "DataLaw_FIBER_selected_pair_frequency.csv"
    detail_path = output_dir / "DataLaw_FairMask_attribute_level_audit.csv"

    df = pd.read_csv(data_path)

    print(
        f"Loaded dataset with shape: {df.shape}"
    )
    print(f"Columns: {df.columns.tolist()}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    manual_attributes = [
        resolve_column(
            df.columns,
            requested,
        )
        for requested in MANUAL_ATTRIBUTE_REQUESTS
    ]

    audit_attributes = [
        resolve_column(
            df.columns,
            requested,
        )
        for requested in FIXED_AUDIT_REQUESTS
    ]

    if len(set(manual_attributes)) != 2:
        raise ValueError(
            "The two manual attributes resolved to the same "
            f"column: {manual_attributes}"
        )

    if len(set(audit_attributes)) != 2:
        raise ValueError(
            "The two audit attributes resolved to the same "
            f"column: {audit_attributes}"
        )

    print(
        "Manual masking attributes:",
        manual_attributes,
    )
    print(
        "Fixed fairness-audit attributes:",
        audit_attributes,
    )

    data_clean = (
        df.dropna()
        .reset_index(drop=True)
    )

    target_values = set(
        data_clean[TARGET_COLUMN]
        .astype(int)
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "pass_bar must be encoded as binary 0/1."
        )

    task_classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        bootstrap=False,
    )

    detector_classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        bootstrap=False,
    )

    attribute_model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    method_names = [
        "Base model",
        "Manual FairMask",
        "FIBER FairMask",
    ]

    results = {
        method: create_storage()
        for method in method_names
    }

    for method in method_names:
        initialize_detail_storage(
            results[method],
            audit_attributes,
        )

    selection_counter = Counter()
    selection_pair_counter = Counter()

    for seed in range(args.reps):
        start_time = time.time()

        train_df, test_df = train_test_split(
            data_clean,
            test_size=TEST_RATIO,
            random_state=seed,
            stratify=data_clean[TARGET_COLUMN],
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # FIBER detection uses training data only.
        fiber_attributes, _ = (
            detect_fiber_attributes(
                train_df=train_df,
                target_col=TARGET_COLUMN,
                classifier=detector_classifier,
                top_n=TOP_N,
                min_score=MIN_SCORE,
                alpha=ALPHA,
                candidate_max_cardinality=(
                    CANDIDATE_MAX_CARDINALITY
                ),
            )
        )

        selection_counter.update(fiber_attributes)
        selection_pair_counter.update(
            [tuple(fiber_attributes)]
        )

        X_train = train_df.drop(
            columns=[TARGET_COLUMN]
        )

        y_train = (
            train_df[TARGET_COLUMN]
            .astype(int)
            .reset_index(drop=True)
        )

        X_test_original = test_df.drop(
            columns=[TARGET_COLUMN]
        )

        y_test = (
            test_df[TARGET_COLUMN]
            .astype(int)
            .reset_index(drop=True)
        )

        # Train the pass_bar model once, so all three conditions
        # use the same learned task model.
        classifier = copy.deepcopy(
            task_classifier
        )
        classifier.fit(X_train, y_train)

        # ====================================================
        # 1. Base model: no masking
        # ====================================================

        y_pred_base = classifier.predict(
            X_test_original
        )

        base_evaluation = evaluate_predictions(
            test_df_original=test_df,
            y_pred=y_pred_base,
            X_train=X_train,
            y_train=y_train,
            X_test_original=X_test_original,
            y_test=y_test,
            audit_attributes=audit_attributes,
        )

        store_evaluation(
            results["Base model"],
            *base_evaluation,
        )

        # ====================================================
        # 2. Manual FairMask:
        #    mask full time and sex
        # ====================================================

        X_test_manual = mask_multiple_attributes(
            X_train=X_train,
            X_test_original=X_test_original,
            mask_attributes=manual_attributes,
            attribute_model=attribute_model,
            use_smote=USE_SMOTE_FOR_ATTRIBUTE,
            random_state=seed,
            verbose=False,
        )

        y_pred_manual = classifier.predict(
            X_test_manual
        )

        manual_evaluation = evaluate_predictions(
            test_df_original=test_df,
            y_pred=y_pred_manual,
            X_train=X_train,
            y_train=y_train,
            X_test_original=X_test_original,
            y_test=y_test,
            audit_attributes=audit_attributes,
        )

        store_evaluation(
            results["Manual FairMask"],
            *manual_evaluation,
        )

        # ====================================================
        # 3. FIBER FairMask:
        #    mask FIBER's automatically selected Top-2
        # ====================================================

        X_test_fiber = mask_multiple_attributes(
            X_train=X_train,
            X_test_original=X_test_original,
            mask_attributes=fiber_attributes,
            attribute_model=attribute_model,
            use_smote=USE_SMOTE_FOR_ATTRIBUTE,
            random_state=seed,
            verbose=False,
        )

        y_pred_fiber = classifier.predict(
            X_test_fiber
        )

        fiber_evaluation = evaluate_predictions(
            test_df_original=test_df,
            y_pred=y_pred_fiber,
            X_train=X_train,
            y_train=y_train,
            X_test_original=X_test_original,
            y_test=y_test,
            audit_attributes=audit_attributes,
        )

        store_evaluation(
            results["FIBER FairMask"],
            *fiber_evaluation,
        )

        print(
            f"Round {seed + 1:02d}/{args.reps} finished in "
            f"{time.time() - start_time:.2f} seconds."
        )

    # ========================================================
    # Main comparison table: macro-average only
    # ========================================================

    masking_descriptions = {
        "Base model": "None",
        "Manual FairMask": ", ".join(
            manual_attributes
        ),
        "FIBER FairMask": "FIBER Top-2 per split",
    }

    summary_rows = []

    for method in method_names:
        row = {
            "Method": method,
            "Masking Attributes": (
                masking_descriptions[method]
            ),
        }

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]:
            row[metric] = mean_value(
                results[method][
                    "performance"
                ][metric]
            )

        for metric in [
            "Macro AOD",
            "Macro EOD",
            "Macro SPD",
            "Macro DI",
        ]:
            row[metric] = mean_value(
                results[method]["macro"][metric]
            )

        summary_rows.append(row)

    summary_table = pd.DataFrame(
        summary_rows
    )

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

    summary_table[numeric_columns] = (
        summary_table[numeric_columns]
        .round(4)
    )

    print("\n" + "=" * 130)
    print(
        "DATALAW: MANUAL FAIRMask VS FIBER FAIRMask"
    )
    print(
        "Fixed fairness-audit attributes: "
        + ", ".join(audit_attributes)
    )
    print(
        "Lower is better for Macro AOD, Macro EOD, "
        "Macro SPD, and Macro DI."
    )
    print("=" * 130)

    print(
        summary_table.to_string(index=False)
    )

    summary_table.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # FIBER selection frequency
    # ========================================================

    selection_rows = [
        {
            "Attribute": attribute,
            "Selected Runs": count,
            "Selection Rate": count / args.reps,
        }
        for attribute, count
        in selection_counter.most_common()
    ]

    selection_table = pd.DataFrame(
        selection_rows
    )

    if not selection_table.empty:
        selection_table[
            "Selection Rate"
        ] = selection_table[
            "Selection Rate"
        ].round(4)

        print("\n" + "=" * 90)
        print("FIBER ATTRIBUTE-SELECTION FREQUENCY")
        print("=" * 90)
        print(
            selection_table.to_string(index=False)
        )

        selection_table.to_csv(
            selection_path,
            index=False,
        )

    pair_rows = [
        {
            "Selected Pair": ", ".join(pair),
            "Runs": count,
            "Rate": count / args.reps,
        }
        for pair, count
        in selection_pair_counter.most_common()
    ]

    pair_table = pd.DataFrame(pair_rows)

    if not pair_table.empty:
        pair_table["Rate"] = (
            pair_table["Rate"].round(4)
        )

        print("\n" + "=" * 90)
        print("FIBER SELECTED-PAIR FREQUENCY")
        print("=" * 90)
        print(
            pair_table.to_string(index=False)
        )
        pair_table.to_csv(pair_path, index=False)

    # ========================================================
    # Save attribute-level audit details for transparency.
    # These are not required in the main macro-average table.
    # ========================================================

    detail_rows = []

    for method in method_names:
        for attribute in audit_attributes:
            detail_rows.append(
                {
                    "Method": method,
                    "Audit Attribute": attribute,
                    "AOD": mean_value(
                        results[method]["details"][
                            attribute
                        ]["AOD"]
                    ),
                    "EOD": mean_value(
                        results[method]["details"][
                            attribute
                        ]["EOD"]
                    ),
                    "SPD": mean_value(
                        results[method]["details"][
                            attribute
                        ]["SPD"]
                    ),
                    "DI": mean_value(
                        results[method]["details"][
                            attribute
                        ]["DI"]
                    ),
                }
            )

    detail_table = pd.DataFrame(
        detail_rows
    )

    detail_table[
        ["AOD", "EOD", "SPD", "DI"]
    ] = detail_table[
        ["AOD", "EOD", "SPD", "DI"]
    ].round(4)

    detail_table.to_csv(
        detail_path,
        index=False,
    )

    print("\nSaved main comparison to:")
    print(results_path)

    print("\nSaved FIBER selection frequency to:")
    print(selection_path)

    print("\nSaved selected-pair frequency to:")
    print(pair_path)

    print("\nSaved detailed fixed-attribute audit to:")
    print(detail_path)