# c50py: C5.0 decision trees for Python, with native categorical splits

[![PyPI](https://img.shields.io/pypi/v/c50py.svg)](https://pypi.org/project/c50py/)
[![Python](https://img.shields.io/pypi/pyversions/c50py.svg)](https://pypi.org/project/c50py/)
[![Tests](https://github.com/daviddiazsolis/c50py/actions/workflows/tests.yml/badge.svg)](https://github.com/daviddiazsolis/c50py/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/daviddiazsolis/c50py/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddiazsolis/c50py/blob/main/examples/c50py_comprehensive_tutorial.ipynb)

Decision trees were born able to split on categories. Quinlan's ID3, C4.5 and
C5.0 ask questions like *"is the sector in {mining, tourism, construction}?"*.
scikit-learn's CART only splits on numbers, so categorical columns have to be
one-hot encoded first, and one question becomes a staircase of dummy variables.

`c50py` brings the C5.0 way back to Python, as a scikit-learn estimator:

![Same data, CART with one-hot encoding (8 leaves) vs c50py (2 leaves), same test accuracy](https://raw.githubusercontent.com/daviddiazsolis/c50py/main/docs/img/cart_vs_c50py.png)

*Same data, same test accuracy (0.782), both pruned by 5-fold cross-validation.
CART needs three dummy variables and 8 leaves; C5.0 asks one question.
Reproduce it with [`examples/make_readme_figure.py`](examples/make_readme_figure.py).*

## Install

```bash
pip install c50py
```

Python 3.9+, NumPy and scikit-learn. pandas is optional (DataFrames are used
directly) and matplotlib is needed for `plot_tree`.

## Quickstart

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from c50py import C5Classifier

url = "https://raw.githubusercontent.com/daviddiazsolis/c50py/main/examples/titanic.csv"
df = pd.read_csv(url)
X = df[["pclass", "sex", "age", "fare", "embarked"]].astype({"pclass": "category"})
y = df["survived"].map({0: "died", 1: "survived"})
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# no one-hot encoding, no imputation: sex, embarked and pclass are used as
# categories, and the missing ages are handled by the tree
clf = C5Classifier(min_samples_leaf=10).fit(X_tr, y_tr)
print(f"test accuracy: {clf.score(X_te, y_te):.3f}")      # 0.794

for rule in clf.export_rules():
    print(rule)
# sex IN {male} => died
# sex NOT IN {male} AND pclass IN {3} AND fare <= 23.3500 => survived
# sex NOT IN {male} AND pclass IN {3} AND fare > 23.3500 => died
# sex NOT IN {male} AND pclass NOT IN {3} => survived

clf.plot_tree(filled=True, rounded=True)   # same look and options as sklearn.tree.plot_tree
```

## A scikit-learn estimator

`C5Classifier` and `C5Regressor` pass scikit-learn's `check_estimator`, so they
work with `clone`, `Pipeline`, `cross_val_score`, `GridSearchCV` and the rest
of the ecosystem:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(C5Classifier(), {"cf": [0.1, 0.25, 0.5], "min_samples_leaf": [1, 5, 20]}, cv=5)
grid.fit(X_tr, y_tr)
print(grid.best_params_, grid.score(X_te, y_te))
```

## What C5.0 does differently from CART

| | scikit-learn CART | c50py (C5.0) |
|---|---|---|
| Categorical columns | must be encoded (one-hot, ordinal) | split natively: `sector in {a, b, c}` |
| Missing values | imputed, or one learned direction | cases go down both branches with fractional weights, as in C4.5/C5.0 |
| Split criterion | Gini or entropy | gain ratio, among splits with at least average gain, with C4.5's MDL penalty for numeric thresholds |
| Pruning | cost-complexity (`ccp_alpha`) | pessimistic error pruning with confidence factor `cf` (C4.5's `AddErrs`) |
| Boosting | separate `AdaBoostClassifier` | built in: `trials=10` |

## Main features

- **Classification and regression**: `C5Classifier`, `C5Regressor`.
- **Categorical columns detected automatically**: pandas `category`, `object`,
  `string` and `bool` columns, or object columns holding strings. You can also
  list them with `categorical_features=["region", 3]` (names or indices).
- **Missing values** (`None`, `np.nan`, `pd.NA`) in training and prediction.
- **Case weights** through `sample_weight`.
- **Boosting** with `trials > 1`.
- **Readable output**: `export_rules()` (one rule per leaf), `predict_rule(X)`
  (the rule each case follows) and `print_tree()`.
- **Drawings identical to scikit-learn**: `plot_tree`, `export_graphviz` and
  `export_text`, with the same parameters. Boxes use c50py's own palette so a
  C5.0 tree is easy to tell apart from a CART tree (`palette="sklearn"` gives
  scikit-learn's colours).

## Regression

```python
from sklearn.datasets import load_diabetes
from c50py import C5Regressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
reg = C5Regressor(min_samples_leaf=20).fit(X, y)
print(reg.export_rules()[:3])
```

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `cf` | `0.25` | Confidence factor for pruning; smaller values prune more. |
| `min_samples_leaf` | `1` (classifier), `2` (regressor) | Minimum (weighted) cases in each child. |
| `trials` | `1` | Number of boosted trees (classifier). |
| `categorical_features` | `None` | Extra columns to treat as categorical, by name or index. |
| `infer_categorical` | `True` | Detect categorical columns from their dtype or content. |
| `max_categories_exhaustive` | `12` | Up to this many categories every binary subset is evaluated. |
| `max_depth` | `None` | Optional depth limit. |

The [Usage Guide](USAGE_GUIDE.md) covers every option.

## How close is it to Quinlan's C5.0?

`c50py` is a from-scratch Python implementation that follows C4.5/C5.0 in the
split criterion, the handling of missing values and the pessimistic pruning.
Some parts of the original are not (yet) implemented:

- splits are binary; categorical splits group categories into two subsets
  (C5.0's default is one branch per category, with subset grouping as an option);
- no **winnowing** (C5.0's built-in feature selection);
- no **rulesets**: `export_rules()` returns the rules of the tree, one per leaf,
  not C5.0's simplified, independent rules;
- boosting follows AdaBoost.M1/SAMME reweighting, close to but not identical
  with C5.0's;
- no misclassification costs.

Rulesets and winnowing are on the roadmap. Contributions are welcome.

## Citation

If you use `c50py` in research, please cite it (a JOSS submission is in
preparation; meanwhile cite the repository):

```bibtex
@software{diaz_solis_c50py,
  author = {D{\'i}az Sol{\'i}s, David},
  title  = {c50py: C5.0 decision trees for Python},
  url    = {https://github.com/daviddiazsolis/c50py},
  year   = {2026}
}
```

## Contributing and license

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md)
and the [CHANGELOG](CHANGELOG.md). `c50py` is released under the MIT License.
The drawing code is adapted from scikit-learn (BSD-3-Clause).
