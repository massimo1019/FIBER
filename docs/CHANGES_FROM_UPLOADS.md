# Repository cleanup changes

The goal of this cleanup is to make the supplied experiment code suitable for a public GitHub repository while preserving the implemented experimental logic.

## Changes made

- Removed Google Colab and Google Drive mounting requirements.
- Replaced hard-coded `/content/drive/...` paths with repository-relative defaults.
- Corrected the repository filename spelling from `fiber_with_fiarmask.py` to `fiber_with_fairmask.py`.
- Converted `Measurenew.py` into the documented `fiber.fairness_metrics` module while preserving its One-vs-Rest metric behavior.
- Added English docstrings to functions that previously had none.
- Added command-line arguments for dataset path, output directory, and repetition count.
- Added automatic output-directory creation.
- Added CSV output for FIBER selected-pair frequency to both experiment pipelines.
- Added installation metadata, dependency files, a data dictionary, methodology notes, reproducibility instructions, and unit tests.
- Preserved the uploaded experiment defaults, including use of `test_20.csv` as the default input followed by repeated internal 80/20 splits.

## Intentionally not inferred

The supplied materials do not specify the full numeric mapping for `race`, the numeric mapping for `fulltime`, the dataset redistribution license/source citation, the final paper citation, or the desired software license. These items are flagged for completion before public release rather than guessed.
