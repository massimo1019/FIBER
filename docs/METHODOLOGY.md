# Methodology implemented in this repository



## 1. FIBER attribute scoring

For each candidate feature whose cardinality does not exceed the configured maximum, FIBER computes three components:

1. **Bias component**: uses DI and SPD disparities measured for that attribute. The raw values are median-calibrated and averaged.
2. **Distribution component**: uses the sum of squared proportions (SSP) from the feature distribution and the target distribution within feature groups.
3. **Relevance component**: uses random-forest feature importance.

The current supplied experiment code combines them as:

```text
Total Score = 0.14 * Bias Score
            + 0.51 * SSP
            + 0.35 * Feature Importance
```

Attributes are ranked by total score. The experiment requests two attributes. If fewer than two meet the score threshold of 0.5, the ranking is used to fill the remaining positions so that FIBER and the manual baseline mitigate the same number of attributes. Attribute selection is performed on the training split only.

## 2. FairMask experiment

The task model is a random forest trained on the original training features and `pass_bar` target.

- **Base model**: predicts using the original test features.
- **Manual FairMask**: reconstructs `fulltime` and `sex` from the other features using separate decision trees, then predicts with the same task model.
- **FIBER FairMask**: reconstructs FIBER's two selected attributes in the same way, then predicts with the same task model.

When enabled, SMOTE is used while training each attribute-reconstruction model if the smallest class has at least six samples. If SMOTE fails, the original attribute-training data are used.

## 3. MAAT experiment

The MAAT code trains one performance model on the original training data. For each selected fairness attribute, it creates a fairness-adjusted training set through the supplied WAE-style row-removal procedure and trains a separate fairness model.

With two selected attributes, final class probabilities are the arithmetic mean of:

```text
performance-model probability
fairness-model probability for attribute 1
fairness-model probability for attribute 2
```

The manual condition uses `fulltime` and `sex`; the FIBER condition uses the two FIBER-selected attributes.

For a multi-category attribute in MAAT, the supplied implementation treats the category with the highest favorable target rate as privileged and combines all remaining categories as the unprivileged group. This behavior can be overridden in the script with `PRIVILEGED_VALUE_OVERRIDES`.

## 4. Fairness evaluation

Every method is audited using the **true, original values** of the fixed audit attributes `fulltime` and `sex`. For each audit attribute, the fairness helper performs category-vs-rest comparisons and averages across categories.

The main experiment tables then report equal-weight macro-averages across the two fixed audit attributes for AOD, EOD, SPD, and DI disparity.
