# Changelog

## 0.5.1

Regression and documentation.

- `C5Regressor`: the split search is vectorised with NumPy (about 20 times faster on the Ames house
  prices data: 5 seconds instead of 107 for 1,000 houses and 79 columns).
- `C5Regressor`: only splits that leave at least `min_samples_leaf` on each side are candidates.
  Before, the best split was chosen first and, if it left too few cases on one side, the node
  became a leaf, so the tree could stop growing too early.
- `C5Regressor`: the grouping "one category against all the others" was never evaluated for the
  first category in the exhaustive search; it is now. Above `max_categories_exhaustive`
  categories, cuts of the categories ordered by mean target are evaluated, as before.
- New notebook `06_regression_house_prices.ipynb`: `C5Regressor` on the Ames data against
  scikit-learn's regression tree and HistGradientBoosting, the rule behind each price, and a small
  example where grouping categories matters.
- CI: the pinned versions of the scikit-learn 1.3 job are quoted again.

Regression trees fitted with 0.5.1 can differ from those of 0.5.0.

## 0.5.0

The complete C5.0: rulesets, winnowing and C5.0's boosting join the tree, and the tree is closer to
C4.5/C5.0. Everything is validated against the original C5.0 (R package `C50`), CART and
HistGradientBoosting on 23 OpenML datasets (`benchmarks/`). With its defaults the c50py tree has the
same accuracy as C5.0's (mean difference -0.2 points, Wilcoxon p = 0.31; trees went from 1.55x to
1.24x C5.0's leaves), rulesets are within 0.5 points of C5.0's (p = 0.07) and boosting matches C5.0's
boosting (+0.1 points) and HistGradientBoosting (+0.1 points).

Rules and rulesets:

- New `C5RulesClassifier`: C5.0's rulesets (`rules = TRUE` in R). Candidate rules from every node of
  the pruned tree are simplified with Laplace error rates and description length, a subset is selected
  by minimum description length, overlapping rules vote with their confidence, and a default class
  covers the rest. A scikit-learn classifier of its own (`fit`, `predict`, `predict_proba`, pipelines,
  grid search).
- `export_ruleset(as_frame=False)` lists the rules with cases, errors, confidence and lift;
  `apply_ruleset(X)` gives, for each row, the rule behind its prediction, how many rules it satisfies
  and the prediction (indexed like `X`).
- `C5Classifier.build_ruleset(X, y)` builds the ruleset from an already fitted tree.
- Rules of the tree: new `apply(X)` (leaf number in `export_rules()` order, as scikit-learn's `apply`)
  and `apply_rules(X)` (rule number, text and prediction for each row, indexed like `X`).
  `predict_rule(X)` now validates its input.

Winnowing:

- `winnow=True`: C5.0's feature selection. A trial tree grown on half of the data screens out the
  columns it does not use and those whose removal lowers its errors on the other half. The dropped
  columns are in `winnowed_features_`.

Boosting:

- `trials > 1` now follows C5.0's boosting instead of AdaBoost.M1/SAMME: misclassified cases gain
  weight additively, trees vote with the confidence of the leaf a case reaches, and boosting stops
  early when a tree is too accurate or too inaccurate. `alphas_` is kept (all ones) for backward
  compatibility; `estimator_errors_` holds the weighted error of each tree.

Documentation:

- Five notebooks in `examples/notebooks/`, all runnable in Colab: getting started on mixed data, c50py
  vs CART, c50py vs gradient boosting, rules and rulesets for churn campaigns, missing values and
  winnowing.
- The benchmark now includes the ruleset, winnowing and boosting variants, HistGradientBoosting, and
  C5.0's rules and boosting, with results by type of data.

Split selection:

- Within a feature the best split is the one with the highest **gain**; gain ratio is only used to compare
  features, as in C4.5 (`contin.c`). Before, categorical subsets were chosen by gain ratio, which favours
  very unbalanced splits and peeled off rare categories one at a time (`purpose NOT IN {a}`, then
  `purpose NOT IN {b}`, ...).
- Categorical groupings pay an MDL cost, `log2(number of groupings evaluated) / known cases`, the same
  principle C4.5 Release 8 applies to numeric thresholds: picking the best of thousands of groupings
  otherwise overstates the gain. Part of `mdl_penalty` (on by default).
- Numeric cuts must leave at least `max(min_samples_leaf, min(25, 10% of the known cases per class))` on
  each side, as in C4.5/C5.0 (`numeric_min_split=True`).

Pruning:

- Subtree raising, as in C4.5/C5.0: a subtree can be replaced by its largest branch
  (`subtree_raising=True`).
- New global pruning, as in C5.0: cost-complexity pruning within one standard error of the training
  errors. On by default (`global_pruning=True`), like C5.0; the old global pass was much weaker.

Defaults and speed:

- `min_samples_leaf=2` by default, C5.0's `minCases`.
- All binary groupings of a categorical feature are evaluated at once with NumPy: 12 to 23 times faster
  on features with many categories.

Trees fitted with 0.5.0 differ from those of 0.4.x. To get the 0.4.x behaviour back as far as possible:
`C5Classifier(min_samples_leaf=1, global_pruning=False, subtree_raising=False, numeric_min_split=False)`.

## 0.4.2

Works anywhere a scikit-learn estimator works:

- `C5Classifier` and `C5Regressor` pass scikit-learn's `check_estimator` (tested on scikit-learn 1.9).
  `get_params`, `set_params` and `clone` work, so `GridSearchCV`, `cross_val_score` and `Pipeline` work
  (before, every one of them failed with `AttributeError: 'C5Classifier' object has no attribute
  'feature_names'`). The constructors now store their parameters as given; fitted attributes are only
  created in `fit`.
- Input validation as in scikit-learn: `n_features_in_` and `feature_names_in_`, a clear error when
  `predict` gets a different number of columns, `NotFittedError` before `fit`, sparse input rejected with
  a message, invalid targets and negative or all-zero `sample_weight` rejected.
- Cases with zero weight are ignored, so weighting a case by `k` is the same as repeating it `k` times.
  The regressor breaks ties between equally good splits deterministically.

Categorical columns:

- `C5Classifier(infer_categorical=True)` is now the default (as in `C5Regressor`): pandas `category`,
  `object`, `string` and `bool` columns, and object columns holding strings, are categorical without
  any configuration. Before, a DataFrame with a text column failed with `could not convert string to
  float` unless `categorical_features` was given.
- With `infer_categorical=False`, a text column that is not in `categorical_features` raises a clear
  error naming the column. Unknown names in `categorical_features` also raise a clear error.
- `pd.NA` (nullable `string`, `Int64`, `boolean` columns) is treated as missing.
- `C5Regressor` accepts `categorical_features` by name with a DataFrame (before it needed the
  `feature_names` parameter).

Fixes:

- `export_rules(class_names=...)` and `print_tree(class_names=...)` failed with string labels
  (`TypeError: list indices must be integers`); `class_names` follow the order of `classes_`.
- Docstrings: `cf` (smaller values prune more), `global_pruning` (off by default in the classifier),
  `infer_categorical`, `numeric_threshold_strategy` (`"all"` by default) and `mdl_penalty` were wrong or
  missing.

Project:

- License: MIT, as already declared on PyPI (the repository's `LICENSE` file said GPL-3.0). The
  scikit-learn drawing code keeps its BSD-3-Clause notice.
- New README with a CART vs c50py comparison (`examples/make_readme_figure.py`) and quickstarts that
  run as written; tests for scikit-learn compatibility; GitHub Actions for tests and for publishing to
  PyPI.

## 0.4.1

- Colours by default (`filled=True`) with c50py's own palette (teal, violet, gold, rose, sky, lime; checked for
  colour-blind separation), so a C5.0 tree is recognisable next to scikit-learn's orange/blue CART trees.
  `palette="sklearn"` gives scikit-learn's exact colours; a list of colours sets one per class.
- Colour codes are always written with two hex digits per channel (custom palettes with dark colours
  produced invalid codes such as `#80 0 0`).

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
- The old light-blue ellipses are gone: same box layout as scikit-learn.
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
