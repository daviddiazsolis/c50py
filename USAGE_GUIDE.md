# c50py Usage Guide

**`c50py`** is a modern Python implementation of Quinlan's C5.0 algorithm, designed to be a drop-in replacement for scikit-learn's `DecisionTreeClassifier` but with powerful additional features.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddiazsolis/c50py/blob/master/examples/c50py_comprehensive_tutorial.ipynb)

## Why C5.0?

While scikit-learn's CART implementation is excellent, C5.0 offers distinct advantages for many real-world datasets:

1.  **Native Categorical Support**: No need for One-Hot Encoding. Splits are based on subsets of categories (e.g., `{A, B} vs {C, D}`), leading to simpler, more interpretable trees.
2.  **Robust Missing Value Handling**: Uses fractional case propagation instead of imputation, preserving data integrity.
3.  **Rule-Based Models**: Can generate easy-to-read rulesets.
4.  **Boosting**: Implements C5.0-style boosting (similar to AdaBoost.M1) for higher accuracy.

---

## Installation

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

### 4. Boosting

Enable boosting by setting `trials > 1`. This creates an ensemble of trees, where each subsequent tree focuses on the errors of the previous ones.

```python
# Train a boosted ensemble of 10 trees
clf_boost = C5Classifier(trials=10)
clf_boost.fit(X_train, y_train)
```

---

## Performance Comparison

In benchmarks against scikit-learn's `DecisionTreeClassifier`, `c50py` often produces:
*   **Simpler Trees**: Significantly fewer nodes for the same accuracy, especially with categorical data.
*   **Comparable Accuracy**: Single trees are competitive; boosted trees often outperform single CART trees.
*   **Better Interpretability**: Due to subset splits and rule extraction.

See the [Comprehensive Tutorial Notebook](examples/c50py_comprehensive_tutorial.ipynb) for a detailed benchmark on the Titanic dataset.
