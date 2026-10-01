# c50py — C5.0‑style Decision Trees for Python (clean 0.2.0)

`c50py` provides transparent, easily inspectable decision trees modelled on
Quinlan’s C5.0 algorithm.  Both classification and regression trees are
supported and expose a scikit‑learn‑like API.  The implementation is written
from scratch in pure Python/Numpy and includes support for numeric and
categorical variables, missing values, pre‑ and post‑pruning, boosting,
rule tracing/export and tree drawings identical to scikit-learn's
(`plot_tree`, `export_graphviz`, `export_text`).

## What's new in 0.4.0

Trees are drawn **exactly like scikit-learn**. `c50py` now has `plot_tree`, `export_graphviz` and
`export_text` with the same parameters, defaults, colours, box text and layout as
`sklearn.tree.plot_tree`, `sklearn.tree.export_graphviz` and `sklearn.tree.export_text`:

```python
import matplotlib.pyplot as plt
from c50py import C5Classifier, plot_tree

clf = C5Classifier(categorical_features=["region"]).fit(X, y)

fig, ax = plt.subplots(figsize=(14, 6))
clf.plot_tree(class_names=["stays", "leaves"], filled=True, rounded=True, ax=ax)   # method form
plot_tree(clf, class_names=["stays", "leaves"], filled=True, rounded=True)         # function form, as in sklearn
```

All of scikit-learn's options work: `max_depth`, `feature_names`, `class_names`, `label`, `filled`,
`impurity`, `node_ids`, `proportion`, `rounded`, `precision`, `ax`, `fontsize` (and, for
`export_graphviz`, `out_file`, `leaves_parallel`, `rotate`, `special_characters`, `fontname`).
The only extra one is `tree_index`, to draw any tree of a boosted model (`trials > 1`).

What is specific to C5.0 inside the boxes: the impurity is the **entropy** (C5.0's criterion; regression
trees show `squared_error` as in scikit-learn), categorical splits read `feature in {a, b}`, and
`samples` can be fractional when there are missing values (C5.0 sends those cases down both branches).

`export_graphviz` no longer needs the `graphviz` package to produce the DOT text, and old calls such as
`export_graphviz("tree", format="png")` keep working.

## What's new in 0.3.0

Version 0.3.0 brings the classifier much closer to Quinlan's C4.5/C5.0 and makes it faster:

- **Pessimistic pruning as in C4.5.** Leaves are penalised with the binomial upper limit (`AddErrs` from `prune.c`) and the original confidence-factor table (cf = 0.25 gives a deviate of about 0.69). Pure leaves now receive a positive penalty, so tiny leaves that only memorised the data are pruned. The previous normal approximation gave pure leaves zero penalty, so trees with pure leaves were never pruned, and the old "global" pass could collapse a useful tree into a single leaf on imbalanced data. `global_pruning` is now off by default.
- **Split selection as in C4.5.** Gain ratio competes only among features whose gain is at least the average gain (`gain_ratio_avg_gain=True`); unknown cases count as an extra branch in the split information; and continuous attributes pay the MDL penalty of C4.5 Release 8 (`mdl_penalty=True`).
- **Exhaustive, vectorised numeric thresholds.** `numeric_threshold_strategy="all"` is now the default and is evaluated with cumulative sums, so it is faster than the old 32-quantile subsample.
- **pandas DataFrames are accepted directly**, column names become feature names, and `infer_categorical=True` is honoured by the classifier (object, category, bool and string columns). `print_tree`, `export_rules`, `predict_rule` and `export_graphviz` default to the names seen in `fit`.
- **Boosting.** A perfect first tree no longer gets an arbitrary weight of 10; it gets a finite AdaBoost weight and boosting stops, as in C5.0. Combined with the new pruning, `trials > 1` now works from `min_samples_leaf=1`.

## Features

- **Scikit-learn API:** `fit(X, y)`, `predict(X)`, `score(X, y)`.
- **Categorical support:** Pass `categorical_features=[0, 2]` to handle categories natively.
- **Sample weights:** Supports `sample_weight` in `fit` for weighted splitting and pruning.
- **Missing values:** Handles missing values using C5.0's fractional propagation strategy.
- **Boosting:** Set `trials=10` to train a boosted ensemble.
- **Rule export:** call `export_rules()` to get a list of human-readable rules.
- **Tree drawing like scikit-learn:** `plot_tree()` (matplotlib), `export_graphviz()` (DOT) and
  `export_text()`, with the same parameters and look as `sklearn.tree`.
- **Pretty printing:** call `print_tree` to display the learned splits in a
  readable nested `if`/`else` format (single trees only).

## Documentation

For a comprehensive guide on how to use `c50py`, including advanced features and examples, please see the [Usage Guide](USAGE_GUIDE.md).

## Installation (development mode)

Install the package into your environment in editable mode:

```bash
pip install -e .
```

## Quickstart (Classification)

```python
import pandas as pd
from time import perf_counter
from c50py import C5Classifier

df = pd.read_csv("titanic.csv")
t0 = perf_counter(); clf.fit(X, y); print(f"fit: {perf_counter()-t0:.3f}s")

# Inspect the tree
clf.print_tree(feature_names=features, class_names=["No", "Yes"])

# Extract rules for each sample
rules = clf.predict_rule(X, feature_names=features)
print(rules[:5])

# Draw it exactly like sklearn.tree.plot_tree
clf.plot_tree(feature_names=features, class_names=["No", "Yes"], filled=True)

# Export as Graphviz
path = clf.export_graphviz(
    "titanic_tree",
    feature_names=features,
    class_names=["No", "Yes"],
    format="dot"  # save a .dot file directly
)
print(f"DOT file written to {path}")
```

## Quickstart – Regression (Diabetes)

Fit a regression tree to the diabetes dataset and obtain a visualisation:

```python
import pandas as pd
from time import perf_counter
from c50py import C5Regressor

df = pd.read_csv("diabetes.csv")
y = df["target"].values
X_df = df.drop(columns=["target"])
X = X_df.values.astype(object)
features = list(X_df.columns)

reg = C5Regressor(
    min_samples_split=30,
    min_samples_leaf=10,
    pruning=True, cf=0.25, global_pruning=True,
    feature_names=features,
    random_state=42,
    infer_categorical=False, int_as_categorical=False,
    numeric_threshold_strategy="quantile", max_numeric_thresholds=64
)

start = perf_counter(); reg.fit(X, y); print(f"fit: {perf_counter()-start:.3f}s")

# Export to DOT (Graphviz installed optional)
dot_path = reg.export_graphviz("diabetes_tree", feature_names=features, format="dot")
print(f"Tree saved to {dot_path}")

# Export human‑readable rules (single trees only)
rules = reg.export_rules(feature_names=features)
print(rules[:3])
```

## Performance tuning

Several hyperparameters influence model complexity and performance:

- **`numeric_threshold_strategy`** (`'quantile'` | `'all'`): subsample candidate numeric
  thresholds.  With `'quantile'` the number of splits considered is limited to
  `max_numeric_thresholds` per feature per node.  `'all'` evaluates every unique
  midpoint (slower on large datasets).
- **`max_numeric_thresholds`**: number of candidate thresholds when using
  `'quantile'` (typically 32–64).
- **`categorical_features`**: list of names or indices marking categorical columns.
- **`max_categories_exhaustive`**: maximum cardinality for exhaustive subset search on
  categorical features; beyond this a simpler one‑vs‑rest strategy is used.
- **`infer_categorical`/`int_as_categorical`**: enable automatic detection of
  categorical/boolean/integer columns when dtype information is not explicit.
- **`max_depth`**: optional depth limit for extremely noisy or deep trees.

When boosting (`trials > 1`) the same hyperparameters apply to each base tree.
