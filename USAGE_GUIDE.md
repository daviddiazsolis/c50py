# c50py Usage Guide

Welcome to the comprehensive usage guide for `c50py`. This guide covers everything from installation to advanced features like boosting and rule extraction.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daviddiazsolis/c50py/blob/main/examples/c50py_tour.ipynb)

## Table of Contents
1. [Installation](#installation)
2. [Basic Classification (Titanic)](#basic-classification-titanic)
3. [Basic Regression (Diabetes)](#basic-regression-diabetes)
4. [Advanced Features](#advanced-features)
    - [Categorical Features](#categorical-features)
    - [Missing Values](#missing-values)
    - [Sample Weights](#sample-weights)
    - [Boosting](#boosting)
    - [Rule Extraction](#rule-extraction)
    - [Visualization](#visualization)

## Installation

You can install `c50py` directly from PyPI:

```bash
pip install c50py
```

To use the visualization features, you should also install `graphviz`:

```bash
pip install c50py[graphviz]
```

## Basic Classification (Titanic)

Here's how to train a simple classification tree.

```python
import pandas as pd
from c50py import C5Classifier

# Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df.dropna(subset=["Embarked"]) # Drop rows with missing target or minimal preprocessing

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features].values
y = df["Survived"].values

# Initialize and fit
# We specify categorical features by name to let the model handle them appropriately
clf = C5Classifier(
    min_samples_split=20,
    categorical_features=["Pclass", "Sex", "Embarked"]
)
clf.fit(X, y, feature_names=features)

# Print the tree structure
clf.print_tree()
```

## Basic Regression (Diabetes)

`c50py` also supports regression.

```python
from sklearn.datasets import load_diabetes
from c50py import C5Regressor

data = load_diabetes()
X, y = data.data, data.target
features = data.feature_names

reg = C5Regressor(min_samples_split=10, pruning=True)
reg.fit(X, y, feature_names=features)

print(f"R^2 Score: {reg.score(X, y):.4f}")
```

## Advanced Features

### Categorical Features
`c50py` handles categorical variables natively (without one-hot encoding). You can specify them by index or name.

```python
# By name (requires feature_names in fit or init)
clf = C5Classifier(categorical_features=["Color", "Size"])

# By index
clf = C5Classifier(categorical_features=[0, 3])
```

### Missing Values
`c50py` implements C5.0's "fractional propagation" strategy. If a value is missing during prediction, the instance is split down *all* branches, weighted by the probability of each branch (observed during training).

```python
import numpy as np
# Predict on an instance with a missing value (NaN)
X_test = [[1, np.nan, 3]] 
prediction = clf.predict(X_test)
```

### Sample Weights
You can assign different weights to samples during training. This affects split selection and pruning.

```python
weights = np.ones(len(y))
weights[y == 1] = 5.0 # Give more weight to the positive class
clf.fit(X, y, sample_weight=weights)
```

### Boosting
Enable boosting by setting `trials > 1`. This uses a C5.0-style boosting (similar to AdaBoost.M1) but with reweighting.

```python
# Train an ensemble of 10 trees
boosted_clf = C5Classifier(trials=10)
boosted_clf.fit(X, y)
```

### Rule Extraction
For single trees (`trials=1`), you can extract the rules as human-readable strings.

```python
rules = clf.export_rules()
for r in rules[:3]:
    print(r)
# Output example:
# if Sex in {female} and Pclass <= 2: class 1
```

### Visualization
You can export the tree to Graphviz format.

```python
# Export to .dot file
clf.export_graphviz("tree", format="dot")

# If Graphviz is installed on your system, you can render directly:
# clf.export_graphviz("tree", format="png")
```
