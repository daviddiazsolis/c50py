# c5py — C5.0‑style Decision Trees for Python (clean 0.2.0)

`c5py` provides transparent, easily inspectable decision trees modelled on
Quinlan’s C5.0 algorithm.  Both classification and regression trees are
supported and expose a scikit‑learn‑like API.  The implementation is written
from scratch in pure Python/Numpy and includes support for numeric and
categorical variables, missing values, pre‑ and post‑pruning, boosting,
rule tracing/export and Graphviz visualisation.

## Features

- **Classification and regression:** separate `C5Classifier` and `C5Regressor` with
  consistent APIs.
- **Numeric & categorical inputs:** specify categorical columns by index or name;
  or let the model infer them based on dtype.
- **Missing values:** handled gracefully during training and prediction via
  fractional propagation (C5.0 style).
- **Sample weights:** full support for `sample_weight` in `fit`, affecting
  splitting, pruning, and leaf prediction.
- **Pruning:** control tree size with `min_samples_split`, `min_samples_leaf`,
  pessimistic pruning via a confidence factor `cf` and an optional global
  merge pass.
- **Boosting (AdaBoost.M1):** set `trials > 1` to train an ensemble of trees
  using reweighting (C5.0 style).
- **Rule extraction:** retrieve human‑readable antecedent strings with
  `predict_rule` and `export_rules`.  These are only available for single
  trees (`trials=1`).
- **Graphviz export:** visualise your tree using `export_graphviz`.  When
  requesting a DOT file (format=`'dot'`) no external Graphviz binary is
  required; for other formats a system installation of Graphviz is used if
  available, otherwise a `.dot` file is emitted as a fallback.
- **Pretty printing:** call `print_tree` to display the learned splits in a
  readable nested `if`/`else` format (single trees only).

## Documentation

For a comprehensive guide on how to use `c5py`, including advanced features and examples, please see the [Usage Guide](USAGE_GUIDE.md).

## Installation (development mode)

Install the package into your environment in editable mode:

```bash
python -m pip install -e .
```

## Quickstart – Classification (Titanic)

Train a classifier on the Titanic survival dataset and inspect the resulting tree and rules:

```python
import pandas as pd
from time import perf_counter
from c5py import C5Classifier

# Load data and cast categorical columns to string
df = pd.read_csv("titanic.csv")
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X_df = df[features].copy()
X_df["pclass"]   = X_df["pclass"].astype(str)
X_df["sex"]      = X_df["sex"].astype(str)
X_df["embarked"] = X_df["embarked"].astype(str)

X = X_df.values.astype(object)
y = df["survived"].astype(int).values

clf = C5Classifier(
    trials=1,                # single tree; enable boosting with trials>1
    min_samples_split=60,
    min_samples_leaf=20,
    pruning=True, cf=0.25, global_pruning=True,
    random_state=42,
    feature_names=features,
    categorical_features=["pclass", "sex", "embarked"],
    infer_categorical=False, int_as_categorical=False,
    numeric_threshold_strategy="quantile", max_numeric_thresholds=32
)

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
