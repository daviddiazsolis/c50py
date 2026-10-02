---
title: 'c50py: A Python Implementation of the C5.0 Decision Tree Algorithm'
tags:
  - Python
  - machine learning
  - decision trees
  - C5.0
  - classification
  - regression
authors:
  - name: David Díaz Solís
    orcid: 0000-0001-7149-0535
    affiliation: 1
affiliations:
 - name: Departamento de Administración, Facultad de Economía y Negocios, Universidad de Chile, Chile
   index: 1
date: 28 November 2025
bibliography: paper.bib
---

# Summary

`c50py` is a modern, pure Python implementation of the C5.0 decision tree algorithm [@quinlan1993c45], originally developed by Ross Quinlan. Its `C5Classifier` and `C5Regressor` are scikit-learn estimators (they pass scikit-learn's `check_estimator` and work in pipelines and model selection) while offering distinct features inherent to C5.0, such as native handling of categorical variables, robust missing value propagation, and rule-based model extraction.

# Statement of need

Decision trees are a fundamental tool in machine learning due to their interpretability and versatility. The Python ecosystem is dominated by `scikit-learn` [@pedregosa2011scikit], which implements the CART (Classification and Regression Trees) algorithm. While CART is powerful, it lacks some key features found in C5.0:

1.  **Native Categorical Support**: CART requires categorical variables to be one-hot encoded, which can lead to sparse data and deep, uninterpretable trees. C5.0 handles categories natively by splitting on subsets of categories (e.g., $\{A, B\}$ vs. $\{C, D\}$), preserving the feature space structure.
2.  **Missing Value Handling**: C5.0 uses a fractional case propagation strategy, where samples with missing values are distributed down all branches with weights proportional to the branch probabilities. This avoids the need for imputation.
3.  **Boosting**: C5.0 includes its own boosting method, which reweights the training cases additively and lets the trees vote with the confidence of their leaves.
4.  **Rules and rulesets**: every path of the tree can be read as a human-readable "if-then" rule, and C5.0 can turn those rules into a compact ruleset of overlapping, simplified rules, a model of its own.
5.  **Winnowing**: C5.0 can screen out irrelevant attributes before growing the final tree.

`c50py` fills this gap by providing a Python-native, easy-to-install package that brings these C5.0 capabilities to the Python data science community, with an API that is familiar to scikit-learn users.

# Features

-   **Scikit-learn compatibility**: passes `check_estimator`; works with `clone`, `Pipeline`, `cross_val_score` and `GridSearchCV`.
-   **Classification and Regression**: Supports both `C5Classifier` and `C5Regressor`.
-   **Native Categorical Splits**: Efficiently handles high-cardinality categorical features.
-   **Missing Value Support**: No imputation required.
-   **Boosting**: Built-in support for boosting trials.
-   **Visualization**: `plot_tree`, `export_graphviz` and `export_text` with the same interface and look as scikit-learn's.
-   **Rules and rulesets**: the rule each case follows (`apply_rules`) and C5.0 rulesets (`C5RulesClassifier`) with confidence and lift.
-   **Winnowing**: C5.0's feature screening (`winnow=True`).

# References
