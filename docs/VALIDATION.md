# Validation performed during repository cleanup

The cleaned repository was checked in the following ways:

- All Python source files compile successfully.
- Every top-level experiment/helper function has an English docstring.
- `pytest` passes all included fairness-metric unit tests.
- The cleaned fairness metric implementation was compared with the supplied `Measurenew.py` implementation across 25 deterministic randomized trials for AOD, EOD, SPD, and DI; the rounded results matched.
- A one-repetition FairMask experiment completed successfully on the supplied `test_20.csv` and generated all expected result files.
- A one-repetition MAAT experiment completed successfully on the supplied `test_20.csv` and generated all expected result files.

The full 20-repetition experiments were not bundled as generated outputs. They can be reproduced with the commands in `REPRODUCIBILITY.md`.
