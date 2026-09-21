# Changelog

## 0.3.0

Fidelity to C4.5/C5.0:

- Pessimistic pruning now uses C4.5's `AddErrs` (binomial upper limit) and the original confidence-factor
  table; pure leaves are penalised, so tiny leaves get pruned. Comparisons are made in error counts, with the
  0.1 tolerance of the original. `global_pruning` defaults to `False` (the old top-down pass compared a node
  against its immediate children only and could collapse a tree into one leaf on imbalanced data).
- Gain ratio is applied only among features with at least average gain; unknown cases enter the split
  information as an extra branch; continuous attributes pay the MDL penalty (`log2(thresholds) / n_known`).
  Both are switchable: `gain_ratio_avg_gain`, `mdl_penalty`.
- Boosting: a perfect tree gets a finite weight and stops the ensemble (was an arbitrary alpha of 10).

Performance and usability:

- Numeric thresholds are evaluated exhaustively and vectorised (`numeric_threshold_strategy="all"` is the
  default; `"quantile"` still available).
- Object columns detect missing values with `_isnan_scalar` in split search (a `float('nan')` inside an
  object column was previously treated as a category).
- pandas DataFrames accepted in `fit`/`predict`; column names become feature names; `infer_categorical`
  honoured by `C5Classifier`.
- Rule and printing helpers default to the feature names seen in `fit`.
- `__version__` now matches the package version.

Tests: `test_boosting_weights` disables pruning and the MDL penalty because with four cases C4.5 correctly
prunes a stump to a single leaf.
