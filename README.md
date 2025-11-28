# c50py — C5.0‑style Decision Trees for Python (clean 0.2.0)

`c50py` provides transparent, easily inspectable decision trees modelled on
Quinlan’s C5.0 algorithm.  Both classification and regression trees are
supported and expose a scikit‑learn‑like API.  The implementation is written
from scratch in pure Python/Numpy and includes support for numeric and
categorical variables, missing values, pre‑ and post‑pruning, boosting,
rule tracing/export and Graphviz visualisation.

## Features

- **Scikit-learn API:** `fit(X, y)`, `predict(X)`, `score(X, y)`.
- **Categorical support:** Pass `categorical_features=[0, 2]` to handle categories natively.
- **Sample weights:** Supports `sample_weight` in `fit` for weighted splitting and pruning.
- **Missing values:** Handles missing values using C5.0's fractional propagation strategy.
- **Boosting:** Set `trials=10` to train a boosted ensemble.
- **Rule export:** call `export_rules()` to get a list of human-readable rules.
- **Graphviz export:** call `export_graphviz()` to visualize the tree.
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
from c5py import C5Regressor

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
