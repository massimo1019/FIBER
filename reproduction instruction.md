

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


