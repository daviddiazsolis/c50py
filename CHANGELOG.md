# Changelog

## 0.4.0

Tree drawings identical to scikit-learn:

- New `plot_tree`, `export_graphviz` and `export_text`, as module functions (`c50py.plot_tree(model, ...)`)
  and as estimator methods (`model.plot_tree(...)`). Same parameters, defaults, palette, node text,
  Reingold-Tilford layout and DOT output as `sklearn.tree.plot_tree`, `sklearn.tree.export_graphviz` and
  `sklearn.tree.export_text` (the drawing code is adapted from scikit-learn, BSD-3-Clause, and vendored so the
  figures do not change with the installed scikit-learn version). Tests check, node by node, that a tree with
  the same structure produces the same annotations, positions, font sizes, colours and DOT text as scikit-learn.
- Inside the boxes: `entropy` for classifiers (the criterion C5.0 uses), `squared_error` for regressors,
  categorical splits written `feature in {a, b}` (the `True` branch holds the listed categories), and
  `samples` shown with decimals only when missing values were split fractionally.
- New `tree_index` argument to draw any tree of a boosted model (`trials > 1`); previously boosted models
  could not be drawn at all.
- `export_graphviz` writes the DOT text itself: the `graphviz` Python package is only needed to render images.
  Backwards compatible: `out_file=None` returns the DOT text (as before with no filename); passing `format`
  keeps the 0.3 behaviour (basename + format, `"dot"` writes `<basename>.dot`, other formats are rendered with
  Graphviz). Without `format`, `out_file` is a path or file handle, as in scikit-learn.
- Default style follows scikit-learn (`filled=False`); the old light-blue ellipses are gone. Use
  `filled=True, rounded=True` for coloured boxes.
- The `True`/`False` labels on the root's arrows follow the installed scikit-learn (they exist since 1.5), so the
  figures match scikit-learn's in the same environment. Tested against scikit-learn 1.3 to 1.8.
- `C5Regressor.fit` takes column names from a pandas DataFrame, as `C5Classifier` already did.

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
