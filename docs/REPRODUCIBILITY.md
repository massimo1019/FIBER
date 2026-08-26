# Reproducibility guide

## Environment

Use Python 3.10 or newer. Install with:

```bash
pip install -e ".[dev]"
```

## FairMask

Full configured run:

```bash
python experiments/fiber_with_fairmask.py --reps 20
```

Default output directory: `outputs/fairmask/`

Generated files:

- `DataLaw_FairMask_manual_vs_FIBER_macro.csv`: main performance and macro-fairness comparison
- `DataLaw_FIBER_selection_frequency.csv`: frequency of each FIBER-selected attribute
- `DataLaw_FIBER_selected_pair_frequency.csv`: frequency of each selected pair
- `DataLaw_FairMask_attribute_level_audit.csv`: fairness metrics for each fixed audit attribute

## MAAT

Full configured run:

```bash
python experiments/fiber_with_maat.py --reps 20
```

Default output directory: `outputs/maat/`

Generated files:

- `DataLaw_MAAT_macro_comparison.csv`: main performance and macro-fairness comparison
- `DataLaw_MAAT_detailed_audit.csv`: fairness metrics for each fixed audit attribute
- `DataLaw_FIBER_selection_frequency.csv`: frequency of each FIBER-selected attribute
- `DataLaw_FIBER_selected_pair_frequency.csv`: frequency of each selected pair
- `DataLaw_MAAT_WAE_debugging_log.csv`: WAE row-removal diagnostics, when records are produced

## Randomness

The supplied code uses deterministic seeds for repeated train/test splits and model templates. Each repetition uses its loop index as the split seed, while model constructors retain the configured random state. Attribute reconstruction and WAE row removal also receive deterministic seeds.

## Data protocol retained from the uploaded code

Both original scripts use `test_20.csv` as the dataset loaded before the repeated split loop. This repository retains that exact default. Changing the protocol to train on `train_80.csv` and test on `test_20.csv` would affect the experiment design and possibly the reported results, so it should be treated as a separate methodological revision.

## Recommended pre-release checks

Before attaching this repository to a paper submission or publication:

1. Run both 20-repetition experiments from a fresh environment.
2. Compare the generated tables with the final manuscript tables.
3. Confirm the final FIBER weights and threshold match the manuscript.
4. Confirm the intended treatment of DI (distance from 1 in this code).
5. Confirm the exact dataset source, race-code mapping, `fulltime` mapping, and redistribution license.
6. Add the paper citation and repository license.
