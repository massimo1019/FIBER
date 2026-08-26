# FIBER

FIBER is an experimental framework for identifying **fairness-sensitive or proxy attributes** in tabular classification and using the selected attributes with downstream fairness-mitigation methods. This repository contains the supplied DataLaw experiments for **FairMask** and **MAAT**, cleaned for local, reproducible use outside Google Colab.

## Repository contents

```text
FIBER/
├── data/
│   ├── train_80.csv
│   ├── test_20.csv
│   └── README.md
├── docs/
│   ├── CHANGES_FROM_UPLOADS.md
│   ├── METHODOLOGY.md
│   └── REPRODUCIBILITY.md
├── experiments/
│   ├── fiber_with_fairmask.py
│   └── fiber_with_maat.py
├── src/fiber/
│   ├── __init__.py
│   └── fairness_metrics.py
├── tests/
│   └── test_fairness_metrics.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Quick start

### 1. Clone the repository and create an environment

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd FIBER
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

### 2. Install dependencies

Recommended:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

You may also install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

### 3. Run the FairMask experiment

```bash
python experiments/fiber_with_fairmask.py
```

A faster smoke test is:

```bash
python experiments/fiber_with_fairmask.py --reps 1
```

Results are written to `outputs/fairmask/`.

### 4. Run the MAAT experiment

```bash
python experiments/fiber_with_maat.py
```

For a one-repetition smoke test:

```bash
python experiments/fiber_with_maat.py --reps 1
```

Results are written to `outputs/maat/`.

## Command-line options

Both experiment scripts accept the same basic options:

```text
--data PATH         CSV file to analyze
--output-dir PATH   Directory for generated result files
--reps N            Number of repeated train/test splits
```

Example:

```bash
python experiments/fiber_with_fairmask.py \
  --data data/train_80.csv \
  --output-dir outputs/fairmask_train80 \
  --reps 5
```

## Default experimental configuration

The repository preserves the values in the supplied experiment code:

- Target: `pass_bar`
- Manual mitigation attributes: `fulltime` and `sex`
- Fixed fairness-audit attributes: `fulltime` and `sex`
- FIBER selects: top 2 candidate attributes
- Candidate maximum cardinality: 10
- FIBER minimum score: 0.5
- Calibration parameter `alpha`: 1.0
- FIBER weights: bias = 0.14, distribution/SSP = 0.51, relevance = 0.35
- Repetitions: 20 by default
- Test ratio inside each repetition: 0.20
- Random forest trees: 200

> **Reproducibility note:** the uploaded scripts used `test_20.csv` as their input dataset and then performed repeated 80/20 splits inside that file. The cleaned scripts retain that behavior as the default so that the repository does not silently change the supplied experiment. `train_80.csv` is included and can be selected explicitly with `--data`.

## Fairness metrics

The experiments report:

- **AOD**: Average Odds Difference disparity
- **EOD**: Equal Opportunity Difference disparity
- **SPD**: Statistical Parity Difference disparity
- **DI**: Disparate Impact represented as distance from the ideal ratio of 1

For a multi-category audit attribute, the helper performs a **One-vs-Rest** comparison for each category and averages the resulting disparities. The experiment then macro-averages across the fixed audit attributes. In the generated summary tables, lower values are better for all four reported fairness disparities.

## Data documentation

See [`data/README.md`](data/README.md). The repository records only mappings supported by the supplied documentation. It intentionally does not invent numeric meanings for fields whose mappings were not provided.


## Tests

```bash
pytest
```

The included tests check the fairness-metric helper against simple cases and validate the public compatibility entry point.
