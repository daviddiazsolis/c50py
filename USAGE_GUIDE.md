# c50py Usage Guide

**`c50py`** is a modern Python implementation of Quinlan's C5.0 algorithm, designed to work as a scikit-learn estimator (it passes `check_estimator`, and works with `Pipeline`, `cross_val_score` and `GridSearchCV`) while keeping what is specific to C5.0.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddiazsolis/c50py/blob/main/examples/c50py_comprehensive_tutorial.ipynb)

## Why C5.0?

While scikit-learn's CART implementation is excellent, C5.0 offers distinct advantages for many real-world datasets:

1.  **Native Categorical Support**: No need for One-Hot Encoding. Splits are based on subsets of categories (e.g., `{A, B} vs {C, D}`), leading to simpler, more interpretable trees.
2.  **Robust Missing Value Handling**: Uses fractional case propagation instead of imputation, preserving data integrity.
3.  **Readable rules**: the rules of the tree, one per leaf (`export_rules`, `apply_rules`), and C5.0's simplified rulesets (`C5RulesClassifier`).
4.  **Boosting and winnowing**: C5.0's boosting (`trials=10`) for higher accuracy and C5.0's winnowing (`winnow=True`) to screen out irrelevant columns.

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
# By default (infer_categorical=True) pandas category/object/string/bool
# columns, and object columns holding strings, are treated as categorical
clf = C5Classifier()
# Add more columns by name or index, e.g. integer codes
clf = C5Classifier(categorical_features=["City", "zip_code"])
# Or use only the columns you list
clf = C5Classifier(infer_categorical=False, categorical_features=["City"])
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

### 3. Rules of the tree and rulesets

**Rules of the tree.** Every leaf is a rule, the conjunction of the tests on its path. They are
mutually exclusive: every case follows exactly one.

```python
for r in clf.export_rules():          # one rule per leaf
    print(r)

clf.apply(X_test)                     # leaf number of each row, in export_rules() order
clf.apply_rules(X_test)               # DataFrame: rule_id, rule, prediction (indexed like X_test)
clf.predict_rule(X_test)              # the rule text of each row
```

**Rulesets.** C5.0's `rules` mode builds a different, smaller model from those rules: tests that do
not pay for themselves are dropped, a subset of rules is selected by minimum description length, and
the rules may overlap (when they do, they vote with their confidence). A default class covers the
cases no rule covers.

```python
from c50py import C5RulesClassifier

ruleset = C5RulesClassifier().fit(X_train, y_train)    # or: clf.build_ruleset(X_train, y_train)
for line in ruleset.export_ruleset():                   # rules with cases, errors, confidence, lift
    print(line)
ruleset.export_ruleset(as_frame=True)                   # the same as a DataFrame
ruleset.apply_ruleset(X_test)                           # rule behind each prediction, rules satisfied
```

`examples/notebooks/04_rules_rulesets_churn_campaigns.ipynb` uses them to split customers at risk of
leaving by reason and assign a retention campaign to each.

**Rules in production.** `to_sql` writes the model as a SQL query, so it can be applied inside a
database without Python:

```python
print(ruleset.to_sql("customers"))     # reproduces ruleset.predict exactly
print(clf.to_sql("customers"))         # one CASE WHEN per leaf, as clf.apply_rules
ruleset.export_ruleset(format="json")  # rules as data
clf.export_rules(format="pandas")      # one DataFrame.query string per rule
```

The regressor has the same tools: `reg.apply_rules(X)` (rule, predicted value and cases of each
row), `reg.to_sql("houses")` and `reg.export_rules(format="json")`.

### 4. Pruning: what `cf` and `min_samples_leaf` do

`c50py` prunes like C4.5: every leaf with `N` cases and `E` errors is charged the upper limit of the
binomial confidence interval (`AddErrs`), a subtree is replaced by a leaf when the leaf's pessimistic
errors do not exceed the subtree's, and the process runs bottom-up. `cf` is the confidence level:
0.25 (default) prunes moderately, 0.10 or 0.01 prune more. Because pruning is error based, on very
imbalanced targets it can remove leaves that only refined probabilities without changing the
predicted class; if you care about ranking (AUC) rather than accuracy, use `pruning=False` with a
sensible `min_samples_leaf`, or a smaller `cf`.

**Imbalanced classes.** `class_weight="balanced"` (or a dict such as `{"fraud": 10, "ok": 1}`)
gives the rare class more weight in the splits, the pruning and the leaf probabilities, as in
scikit-learn's trees; it is also available in `C5RulesClassifier`. The alternative is to keep the
weights and choose the decision threshold on `predict_proba` by the cost of each error.

### 5. Boosting and winnowing

`trials > 1` grows an ensemble the way C5.0 does: each new tree is grown on the training cases
reweighted so that the ones the previous trees got wrong count more, the trees vote with the
confidence of the leaf each case reaches, and boosting stops early when a tree is too accurate or too
inaccurate to help.

```python
clf_boost = C5Classifier(trials=10).fit(X_train, y_train)
len(clf_boost.ensemble_), clf_boost.estimator_errors_
```

`winnow=True` screens the columns before the final tree is grown: a trial tree on half of the data
drops the columns it never uses and those whose removal lowers its errors on the other half.

```python
clf_w = C5Classifier(winnow=True).fit(X_train, y_train)
clf_w.winnowed_features_            # the columns that were dropped
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
