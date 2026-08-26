# FIBER: Automatic Identification of Fairness-Sensitive Attributes

## Overview

Many machine-learning fairness methods assume that the sensitive attributes are already known. In practice, however, deciding which attributes should receive fairness attention can be difficult. Important fairness-sensitive information may appear in demographic variables, proxy variables, or task-related variables whose relationship with model outcomes is not obvious before analysis.

**FIBER (Fairness Identification via Bias, Evenness, and Relevance)** is designed to support this attribute-identification step. FIBER evaluates candidate features using three complementary signals, ranks them, and selects the attributes that may deserve fairness attention. The selected attributes can then be supplied to an existing fairness-mitigation method.

This repository evaluates FIBER with two mitigation approaches:

- **FairMask**, which replaces selected attribute values at prediction time with reconstructed values inferred from the remaining features.
- **MAAT**, which creates fairness-oriented models for selected attributes and ensembles them with a performance model.

The main contribution demonstrated by this repository is therefore not a new classifier. It is a framework for **automatically identifying fairness-sensitive or proxy attributes and connecting that identification step to downstream fairness mitigation**.

```text
Dataset
   │
   ▼
FIBER attribute scoring
Bias + Evenness + Relevance
   │
   ▼
Top-ranked fairness-sensitive attributes
   │
   ├───────────────────────┐
   ▼                       ▼
FairMask                  MAAT
Attribute                 Fairness-model
reconstruction            ensemble
   │                       │
   └───────────┬───────────┘
               ▼
    Performance and fairness
           evaluation
```

---

## 1. What is FIBER?

FIBER ranks candidate attributes using **Bias**, **Evenness**, and **Relevance**.

### Bias

The bias component asks whether model outcomes differ across the groups of a candidate attribute. In the current implementation, the bias signal uses:

- **DI**: Disparate Impact disparity
- **SPD**: Statistical Parity Difference

The raw DI and SPD values are calibrated relative to the candidate set and then averaged to form the bias score. A stronger disparity across an attribute's groups produces a stronger bias signal.

### Evenness

The evenness component describes the distributional structure of a candidate attribute and the target labels within its groups. The implementation uses the **sum of squared proportions (SSP)**. Larger SSP values correspond to a more concentrated or uneven distribution.

This component allows FIBER to consider group structure rather than relying only on prediction disparity.

### Relevance

The relevance component measures how important an attribute is to the prediction task. In these experiments, relevance is represented by **Random Forest feature importance**.

This is useful because a feature can be associated with group disparity but have little relationship with the prediction task, or it can be highly predictive but show little evidence of disparity. FIBER combines these different perspectives instead of relying on only one criterion.

### Combined FIBER score

For the DataLaw experiments in this repository, the implemented score is:

```text
FIBER Score = 0.14 × Bias
            + 0.51 × Evenness/SSP
            + 0.35 × Relevance
```

Candidate attributes are ranked by the total score. In the experiments below, FIBER selects the **top two attributes** from each training split. The manual baseline also uses two attributes, which keeps the number of mitigated attributes comparable.

FIBER performs attribute selection using the **training split only**. Test data are not used to determine which attributes are selected.

### What does a FIBER-selected attribute mean?

FIBER should be interpreted as an **attribute-identification and decision-support framework**, not as a legal definition of a protected attribute. A highly ranked feature may be a conventional sensitive attribute, a proxy attribute, or a legitimate task-related variable that nevertheless deserves fairness examination.

The final decision about whether and how an attribute should be protected or mitigated still depends on the application context and domain judgment.

---

## 2. What is FairMask?

FairMask is used here as a fairness-mitigation approach that reduces direct reliance on the true values of selected attributes during prediction.

For every selected attribute, the implementation trains a separate reconstruction model using the remaining features. At test time, the original selected attribute value is replaced with the reconstructed value. The task classifier then predicts using this modified feature vector.

```text
Original test data
      │
      ▼
Remove selected attribute values
      │
      ▼
Predict those attributes from the remaining features
      │
      ▼
Replace the original values with reconstructed values
      │
      ▼
Use the task classifier for the final prediction
```

The task classifier itself is kept the same across the three FairMask conditions. This allows the experiment to focus on the effect of attribute masking.

The comparison contains:

1. **Base model**: no masking.
2. **Manual FairMask**: manually masks `fulltime` and `sex`.
3. **FIBER FairMask**: masks the two attributes automatically selected by FIBER in each split.

The purpose of this experiment is to test whether FIBER can identify attributes that are more useful for fairness mitigation than a fixed manually selected pair.

---

## 3. What is MAAT?

MAAT provides a second and structurally different mitigation setting for evaluating FIBER.

In the implementation used here, MAAT first trains a **performance model** on the original training data. For each selected fairness attribute, it then creates a fairness-adjusted training set and trains a separate **fairness model**.

The fairness-adjusted data are created using the WAE-style procedure in the MAAT implementation. The procedure removes selected privileged/favorable and unprivileged/unfavorable examples to reduce the outcome-rate difference while attempting to preserve the original group-size relationship.

When two fairness attributes are used, MAAT produces three probability vectors:

```text
1. Performance-model probability
2. Fairness-model probability for attribute 1
3. Fairness-model probability for attribute 2
```

The final probability is their arithmetic mean:

```text
Final probability =
    (performance probability
     + fairness probability 1
     + fairness probability 2) / 3
```

The MAAT comparison contains:

1. **Base Model**: the original performance model.
2. **Manual-MAAT**: fairness models for `fulltime` and `sex`.
3. **FIBER-MAAT**: fairness models for the two attributes selected by FIBER in each split.

Using both FairMask and MAAT allows us to test whether FIBER is useful beyond a single mitigation mechanism.

---

## 4. DataLaw experiment

The experiments in this repository use the **DataLaw** dataset. The prediction target is:

- `pass_bar`: whether the individual passed the bar examination.

The supplied data contain the following features:

- `sex`
- `race`
- `lsat`
- `ugpa`
- `fulltime`
- `tier`
- `fam_inc`

The supplied data notes identify `sex` as gender, `race` as race, `lsat` as the standardized LSAT score, `ugpa` as undergraduate GPA, `tier` as law-school ranking tier, and `fam_inc` as family-income level. The repository does not invent numerical mappings that were not included in the supplied documentation.

### Manual mitigation and fixed fairness audit

The manual baseline uses:

```text
fulltime, sex
```

The final predictions from **all methods** are audited using the same two original attributes:

```text
fulltime, sex
```

This distinction is important. FIBER is allowed to select different attributes for mitigation, but the fairness evaluation remains fixed. Therefore, the base model, manual mitigation, and FIBER mitigation are compared against the same fairness criteria.

The reported Macro AOD, Macro EOD, Macro SPD, and Macro DI values are equal-weight averages across the fixed `fulltime` and `sex` audits.

---

## 5. Evaluation metrics

### Predictive performance

The experiments report:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**

Higher values indicate better predictive performance.

### Fairness metrics

The experiments report:

- **AOD**: Average Odds Difference
- **EOD**: Equal Opportunity Difference
- **SPD**: Statistical Parity Difference
- **DI**: Disparate Impact disparity

For a categorical audit attribute, the implementation compares each group with the rest of the data using a One-vs-Rest procedure and averages the group disparities. The experiment then macro-averages the resulting fairness scores across `fulltime` and `sex`.

The code reports absolute disparity values, and DI is represented as distance from the ideal DI ratio of 1. Therefore:

> **Lower Macro AOD, Macro EOD, Macro SPD, and Macro DI values indicate better fairness. Values closer to 0 indicate less disparity.**

---

## 6. Results: FIBER with FairMask

| Method | Masking Attributes | Accuracy | Precision | Recall | F1 | Macro AOD ↓ | Macro EOD ↓ | Macro SPD ↓ | Macro DI ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base model | None | 0.9244 | 0.9465 | 0.9746 | 0.9603 | 0.0623 | 0.0346 | 0.0365 | 0.0389 |
| Manual FairMask | `fulltime`, `sex` | 0.9130 | 0.9446 | 0.9640 | 0.9542 | 0.0601 | 0.0311 | 0.0288 | 0.0309 |
| **FIBER FairMask** | **FIBER Top-2 per split** | **0.9163** | **0.9461** | **0.9660** | **0.9559** | **0.0523** | **0.0203** | **0.0206** | **0.0218** |

### Result interpretation

The FairMask experiment shows a clear advantage for FIBER-based attribute selection on the reported fairness measures.

Compared with **Manual FairMask**, FIBER FairMask reduces all four macro disparities:

- Macro AOD: **0.0601 → 0.0523**, approximately **13.0% lower**.
- Macro EOD: **0.0311 → 0.0203**, approximately **34.7% lower**.
- Macro SPD: **0.0288 → 0.0206**, approximately **28.5% lower**.
- Macro DI: **0.0309 → 0.0218**, approximately **29.4% lower**.

FIBER FairMask also performs slightly better than Manual FairMask on the main predictive metrics: accuracy increases from **0.9130 to 0.9163**, and F1 increases from **0.9542 to 0.9559**.

Compared with the **Base model**, FIBER FairMask also substantially reduces disparity. For example, Macro SPD falls from **0.0365 to 0.0206**, and Macro DI falls from **0.0389 to 0.0218**. This fairness improvement comes with a modest decrease in predictive performance relative to the unmitigated model: accuracy changes from **0.9244 to 0.9163**, and F1 changes from **0.9603 to 0.9559**.

Overall, the FairMask results indicate that automatically selecting attributes with FIBER can produce **better fairness than the manual `fulltime` + `sex` selection while maintaining similar predictive performance**.

---

## 7. Results: FIBER with MAAT

| Method | Fairness-Model Attributes | Accuracy | Precision | Recall | F1 | Macro AOD ↓ | Macro EOD ↓ | Macro SPD ↓ | Macro DI ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Model | None | 0.9239 | 0.9463 | 0.9744 | 0.9601 | 0.0658 | 0.0383 | 0.0400 | 0.0428 |
| Manual-MAAT | `fulltime`; `sex` | 0.9260 | **0.9460** | 0.9771 | 0.9613 | **0.0581** | 0.0224 | 0.0229 | 0.0240 |
| **FIBER-MAAT** | **FIBER Top-2 per split** | **0.9280** | 0.9447 | **0.9807** | **0.9624** | 0.0598 | **0.0177** | **0.0183** | **0.0190** |

### Result interpretation

The MAAT experiment provides a second test of FIBER with a different mitigation mechanism.

Compared with **Manual-MAAT**, FIBER-MAAT improves three of the four macro fairness measures:

- Macro EOD: **0.0224 → 0.0177**, approximately **21.0% lower**.
- Macro SPD: **0.0229 → 0.0183**, approximately **20.1% lower**.
- Macro DI: **0.0240 → 0.0190**, approximately **20.8% lower**.

Macro AOD is the exception. It changes from **0.0581** for Manual-MAAT to **0.0598** for FIBER-MAAT. Because lower is better, Manual-MAAT is slightly better on AOD in this comparison.

FIBER-MAAT nevertheless achieves the highest **accuracy (0.9280)**, **recall (0.9807)**, and **F1 (0.9624)** among the three MAAT conditions. Its precision of **0.9447** is slightly below the Base Model and Manual-MAAT.

Compared with the **Base Model**, FIBER-MAAT reduces Macro EOD from **0.0383 to 0.0177**, Macro SPD from **0.0400 to 0.0183**, and Macro DI from **0.0428 to 0.0190**, while accuracy increases from **0.9239 to 0.9280**.

The MAAT results therefore should not be described as a uniform improvement on every metric. Instead, they show that FIBER-selected attributes provide **stronger EOD, SPD, and DI fairness results than the manual attribute set while preserving or slightly improving predictive performance**, with a small tradeoff in AOD and precision.

---

## 8. Main findings

The experiments address the following research question:

> **Can fairness-sensitive attributes be identified automatically, and can those automatically selected attributes improve downstream fairness mitigation compared with a fixed manual attribute set?**

The DataLaw experiments provide encouraging evidence.

### 1. FIBER is not tied to one mitigation algorithm

The same FIBER selection mechanism is used with two different approaches. FairMask modifies selected feature values at prediction time, whereas MAAT creates and ensembles fairness-oriented models. Improvements across both settings suggest that FIBER can function as an attribute-selection layer rather than as a method designed for only one mitigation algorithm.

### 2. Automatic selection can outperform the manual attribute set

With FairMask, FIBER improves all four reported macro fairness measures compared with manually selecting `fulltime` and `sex`.

With MAAT, FIBER improves Macro EOD, Macro SPD, and Macro DI compared with the manual pair, while Manual-MAAT remains slightly better on Macro AOD.

### 3. The performance cost is small in these experiments

For FairMask, FIBER FairMask has lower accuracy than the unmitigated Base model but performs slightly better than Manual FairMask.

For MAAT, FIBER-MAAT obtains the highest accuracy, recall, and F1 of the three compared conditions while also producing the lowest EOD, SPD, and DI disparities.

### 4. FIBER complements, rather than replaces, human judgment

FIBER provides statistical evidence about which attributes may be fairness-sensitive. It does not determine whether an attribute is legally protected or whether mitigation is appropriate in every application. The selected attributes should still be examined case by case using domain knowledge, policy requirements, and the intended use of the model.

---

## 9. Repository organization

```text
FIBER/
├── data/
│   ├── train_80.csv
│   ├── test_20.csv
│   └── README.md
├── docs/
│   ├── METHODOLOGY.md
│   ├── REPRODUCIBILITY.md
│   ├── VALIDATION.md
│   └── CHANGES_FROM_UPLOADS.md
├── experiments/
│   ├── fiber_with_fairmask.py
│   └── fiber_with_maat.py
├── src/fiber/
│   ├── __init__.py
│   └── fairness_metrics.py
├── tests/
│   └── test_fairness_metrics.py
├── pyproject.toml
└── requirements.txt
```

The main experiment scripts are in `experiments/`. The reusable fairness calculations are in `src/fiber/fairness_metrics.py`, and additional implementation/reproducibility notes are in `docs/`.

---

## 10. Experimental configuration

The reported DataLaw experiments use:

- Target: `pass_bar`
- Manual mitigation attributes: `fulltime`, `sex`
- Fixed fairness-audit attributes: `fulltime`, `sex`
- FIBER-selected attributes: top 2 per training split
- Candidate maximum cardinality: 10
- FIBER minimum score threshold: 0.5
- FIBER weights: Bias = 0.14, Evenness/SSP = 0.51, Relevance = 0.35
- Repeated evaluations: 20
- Random Forest estimators: 200

Because FIBER performs selection independently on each training split, the selected pair can vary across repetitions.

---

## Citation

The final FIBER paper citation should be added here when the publication information is finalized.

```bibtex
@article{FIBER,
  title   = {FIBER: ...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## License

Add the intended software license and applicable dataset-use information before public release.
