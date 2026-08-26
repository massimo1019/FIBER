# DataLaw files

This directory contains the two CSV files supplied with the FIBER repository materials.

| File | Rows | Columns | Role in the supplied materials |
|---|---:|---:|---|
| `train_80.csv` | 17,588 | 8 | Provided 80% split; not used by the original experiment scripts by default |
| `test_20.csv` | 4,397 | 8 | Default input used by both supplied experiment scripts |

Both files contain the same columns:

| Column | Documentation supported by the supplied materials |
|---|---|
| `sex` | Gender coding: `1 = female`, `2 = male` |
| `race` | Race category used for fairness/bias analysis; numeric category mapping was not supplied |
| `lsat` | LSAT standardized test score |
| `ugpa` | Undergraduate GPA, documented on a 0.0-4.0 scale |
| `fulltime` | Binary-coded field used as a manual mitigation and audit attribute; numeric value mapping was not supplied |
| `pass_bar` | Binary target used by the experiment code (`0/1`) |
| `tier` | Law-school ranking tier; `1` is top tier and larger values indicate lower tiers |
| `fam_inc` | Ordinal family-income category |

## Important usage note

The supplied FairMask and MAAT scripts both read `test_20.csv` and then create repeated stratified 80/20 train/test splits internally. This repository preserves that default behavior. If the intended final experimental protocol should instead train on `train_80.csv` and evaluate once on `test_20.csv`, that is a methodological change and should be implemented and reported explicitly rather than silently changed during repository cleanup.
