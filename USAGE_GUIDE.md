# c50py Usage Guide

**`c50py`** is a modern Python implementation of Quinlan's C5.0 algorithm, designed to be a drop-in replacement for scikit-learn's `DecisionTreeClassifier` but with powerful additional features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddiazsolis/c50py/blob/main/examples/c50py_comprehensive_tutorial.ipynb)

## Why C5.0?

While scikit-learn's CART implementation is excellent, C5.0 offers distinct advantages for many real-world datasets:

1.  **Native Categorical Support**: No need for One-Hot Encoding. Splits are based on subsets of categories (e.g., `{A, B} vs {C, D}`), leading to simpler, more interpretable trees.
2.  **Robust Missing Value Handling**: Uses fractional case propagation instead of imputation, preserving data integrity.
3.  **Rule-Based Models**: Can generate easy-to-read rulesets.
4.  **Boosting**: Implements C5.0-style boosting (similar to AdaBoost.M1) for higher accuracy.

---

## Installation

Requires Python 3.9+, NumPy and scikit-learn. pandas is optional (DataFrames are accepted directly); matplotlib is needed for `plot_tree`.

```bash
pip install c50py
```

## Quickstart

### Classification

```python
from c50py import C5Classifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = C5Classifier(min_samples_leaf=2)
clf.fit(X, y)
print(f"Accuracy: {clf.score(X, y):.4f}")
```

### Regression

```python
from c50py import C5Regressor
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
reg = C5Regressor(min_samples_leaf=5)
reg.fit(X, y)
print(f"R^2 Score: {reg.score(X, y):.4f}")
```

---

## Key Features Deep Dive

### 1. Native Categorical Support & Automatic Merging

Standard decision trees (CART) require categorical variables to be One-Hot Encoded. This explodes the feature space and results in deep, hard-to-read trees ("staircase" splits).

**`c50py` handles categories natively.** It finds the optimal split by grouping categories into two subsets.

**Example:**
Imagine a `City` feature with values `{NY, LA, CHI, HOU}`.
*   **CART (One-Hot)**: `if City_NY == 1` then ... else `if City_LA == 1` ...
*   **C5.0**: `if City in {NY, CHI}` then Left else Right.

```python
# Specify categorical features by index or name
clf = C5Classifier(categorical_features=["City", "State"])
# OR let c50py infer them (object/category/bool columns)
clf = C5Classifier(infer_categorical=True)
```

### 2. Missing Value Handling

`c50py` does not require you to fill `NaN` values. It uses **fractional case propagation**:
*   **Training**: If a value is missing at a split, the instance is sent down **both** branches with a weight proportional to the probability of that branch.
*   **Prediction**: The prediction is a weighted average of the results from both branches.

```python
import numpy as np
X = [[1, 2], [np.nan, 5], [3, 6]]
y = [0, 1, 0]
clf = C5Classifier()
clf.fit(X, y) # Works natively!
```

### 3. Rule Extraction & Tracing

You can extract human-readable rules from the tree or trace why a specific prediction was made.

```python
# Get all rules
rules = clf.export_rules(feature_names=["Age", "Income"])
for r in rules:
    print(r)

# Trace a specific prediction
trace = clf.predict_rule([X_test[0]], feature_names=["Age", "Income"])
print(trace[0])
```

### 4. Pruning: what `cf` and `min_samples_leaf` do

`c50py` prunes like C4.5: every leaf with `N` cases and `E` errors is charged the upper limit of the
binomial confidence interval (`AddErrs`), a subtree is replaced by a leaf when the leaf's pessimistic
errors do not exceed the subtree's, and the process runs bottom-up. `cf` is the confidence level:
0.25 (default) prunes moderately, 0.10 or 0.01 prune more. Because pruning is error based, on very
imbalanced targets it can remove leaves that only refined probabilities without changing the
predicted class; if you care about ranking (AUC) rather than accuracy, use `pruning=False` with a
sensible `min_samples_leaf`, or a smaller `cf`.

### 5. Boosting

Enable boosting by setting `trials > 1`. This creates an ensemble of trees, where each subsequent tree focuses on the errors of the previous ones.

```python
# Train a boosted ensemble of 10 trees
clf_boost = C5Classifier(trials=10)
clf_boost.fit(X_train, y_train)
```

### 6. Drawing trees, exactly like scikit-learn

Since 0.4.0, `c50py` draws trees with the same functions, parameters and layout as
`sklearn.tree`. Anything you write for `sklearn.tree.plot_tree` works for a c50py model:

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree as sk_plot_tree
import c50py

fig, axes = plt.subplots(1, 2, figsize=(22, 7))
c50py.plot_tree(c5_model, class_names=["No", "Yes"], filled=True, rounded=True, ax=axes[0])
sk_plot_tree(cart_model, feature_names=cols, class_names=["No", "Yes"], filled=True, rounded=True, ax=axes[1])
```

| Function | Same as | Notes |
|---|---|---|
| `plot_tree(model, ...)` / `model.plot_tree(...)` | `sklearn.tree.plot_tree` | matplotlib figure |
| `export_graphviz(model, ...)` / `model.export_graphviz(...)` | `sklearn.tree.export_graphviz` | DOT text; render with `graphviz.Source(dot)` |
| `export_text(model, ...)` / `model.export_text(...)` | `sklearn.tree.export_text` | text report |

Parameters: `max_depth`, `feature_names`, `class_names` (list or `True`), `label`, `filled`, `impurity`,
`node_ids`, `proportion`, `rounded`, `precision`, `ax`, `fontsize`; plus, for `export_graphviz`,
`out_file`, `leaves_parallel`, `rotate`, `special_characters`, `fontname`. The extra `tree_index`
selects a tree of a boosted model: `clf_boost.plot_tree(tree_index=3)`.

Colours: c50py fills the boxes by default (`filled=True`) with its own palette, teal, violet, gold,
rose, sky and lime, so C5.0 trees stand out next to scikit-learn's orange/blue CART trees. Use
`palette="sklearn"` for scikit-learn's exact colours, a list such as `palette=["#1b9e77", "#d95f02"]`
for your own (one per class, in the order of `classes_`), or `filled=False` for white boxes.

Feature names default to the ones seen in `fit` (a DataFrame's columns or the `feature_names`
argument); with no names, boxes read `x[0], x[1], ...` as in scikit-learn.

What differs inside the boxes comes from the model, not the drawing: classifiers show `entropy`
(C5.0's criterion) instead of `gini`; categorical splits read `feature in {a, b}` with the listed
categories on the `True` (left) branch; and `samples` can be fractional when cases with missing values
were split between branches.

---

## Performance Comparison

In benchmarks against scikit-learn's `DecisionTreeClassifier`, `c50py` often produces:
*   **Simpler Trees**: Significantly fewer nodes for the same accuracy, especially with categorical data.
*   **Comparable Accuracy**: Single trees are competitive; boosted trees often outperform single CART trees.
*   **Better Interpretability**: Due to subset splits and rule extraction.

See the [Comprehensive Tutorial Notebook](examples/c50py_comprehensive_tutorial.ipynb) for a detailed benchmark on the Titanic dataset.
