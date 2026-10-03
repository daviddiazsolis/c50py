---
name: interpretable-tabular-rules
description: Build readable decision trees and rule models on tabular data with categorical columns or missing values (credit, churn, fraud, eligibility, segmentation) using c50py (C5.0) instead of one-hot encoding plus scikit-learn's DecisionTreeClassifier. Use when the user wants a model they can read, explain per row, turn into business rules or deploy in SQL.
---

# Interpretable rules on tabular data with c50py

## Decide first

Ask what the model is for.

- **Explain, segment, audit, deploy as rules** (credit committee, churn campaigns by reason,
  eligibility rules, fraud rules): use `c50py`. Go on below.
- **Only the most accurate score** on large numeric data: use gradient boosting
  (`HistGradientBoostingClassifier`, LightGBM, XGBoost). A single tree of any kind gives up one to
  two points of accuracy on average. `C5Classifier(trials=10)` is a middle ground (C5.0 boosting,
  about as accurate as HistGradientBoosting on mixed data, no longer readable).

Why not `pd.get_dummies` + `DecisionTreeClassifier`: with one dummy column per category, the tree
can only ask about one category at a time. A column with many categories (region, product,
sector) becomes a staircase of splits, or is never used because each dummy separates a small
group and loses against any numeric threshold. C5.0 asks `region IN {North, South, East}` in one
test. On 23 OpenML datasets c50py has 64% fewer leaves than CART with default settings and is
1.2 points more accurate.

## Recipe

```python
# pip install c50py
import pandas as pd
from c50py import C5Classifier, C5RulesClassifier

# 1. Keep the table as it is: categorical columns as pandas "category", NaN allowed.
for c in X.select_dtypes(["object", "string", "bool"]).columns:
    X[c] = X[c].astype("category")
# integer codes that are really categories (area code, branch id) must be cast too

# 2. A tree; tune pruning if it must be small.
from sklearn.model_selection import GridSearchCV
tree = GridSearchCV(C5Classifier(), {"cf": [0.05, 0.1, 0.25], "min_samples_leaf": [2, 10, 25]}, cv=5).fit(X_train, y_train).best_estimator_

# 3. The rule behind each prediction (rules of the tree are mutually exclusive).
tree.apply_rules(X_test)          # rule_id, rule, prediction

# 4. A shorter rule model for people: C5.0 ruleset (rules overlap and vote).
rules = C5RulesClassifier(cf=tree.cf, min_samples_leaf=tree.min_samples_leaf).fit(X_train, y_train)
rules.export_ruleset(as_frame=True)   # conditions, prediction, cases, errors, confidence, lift
rules.apply_ruleset(X_test)           # rule behind each prediction, number of rules satisfied

# 5. Deploy without Python.
sql = rules.to_sql("customers")       # SQL that reproduces rules.predict exactly
spec = rules.export_ruleset(format="json")
```

## Reporting

- Compare with a baseline (`DecisionTreeClassifier` on one-hot data, tuned with `ccp_alpha`) and,
  if accuracy matters, with HistGradientBoosting; report accuracy and AUC with cross-validation,
  plus the number of rules and tests per rule.
- Present rules with their **confidence** (share of covered training cases they get right) and
  **lift** (confidence over the base rate of the class).
- Group rules by reason to propose actions (one campaign per reason), and say that a rule
  describes who, not why: test actions with a control group before rolling them out.
- Imbalanced target (fraud, default, churn under 10%): use `class_weight="balanced"` or choose the
  decision threshold on `predict_proba`; report recall and precision of the rare class, not only
  accuracy.
- With many missing values (30% or more of each column), compare c50py's fractional handling with
  simple imputation by cross-validation and keep the better one.
