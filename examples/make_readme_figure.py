"""Figure for the README: the same data, CART with one-hot encoding vs c50py.

Run from the repository root:  python examples/make_readme_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree as sk_plot_tree

from c50py import C5Classifier

rng = np.random.default_rng(7)
n = 3000
sectors = ["retail", "mining", "farming", "tech", "transport", "tourism", "health", "construction"]
risky = {"mining", "tourism", "construction"}
df = pd.DataFrame({
    "sector": pd.Categorical(rng.choice(sectors, n)),
    "debt_ratio": rng.uniform(0, 1, n).round(2),
})
p = np.where(df.sector.isin(risky), 0.75, 0.2)
p = np.where(df.debt_ratio > 0.7, p + 0.2, p)
y = np.where(rng.uniform(size=n) < p, "default", "pays")

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state=0)

# CART: categorical columns must be one-hot encoded
oh_tr = pd.get_dummies(X_tr, columns=["sector"], dtype=int)
oh_te = pd.get_dummies(X_te, columns=["sector"], dtype=int)
# give CART its best shot: cost-complexity pruning tuned by 5-fold CV
grid = {"ccp_alpha": np.linspace(0, 0.01, 21), "min_samples_leaf": [1, 20, 50]}
cart = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=5).fit(oh_tr, y_tr).best_estimator_

# c50py: the categorical column is used as it is
c5 = GridSearchCV(C5Classifier(), {"cf": [0.05, 0.1, 0.25, 0.5], "min_samples_leaf": [1, 20, 50]},
                  cv=5).fit(X_tr, y_tr).best_estimator_

acc_cart, acc_c5 = cart.score(oh_te, y_te), c5.score(X_te, y_te)
leaves_cart = cart.get_n_leaves()
leaves_c5 = len(c5.export_rules())
print(f"CART + one-hot: test accuracy {acc_cart:.3f}, {leaves_cart} leaves")
print(f"c50py         : test accuracy {acc_c5:.3f}, {leaves_c5} leaves")

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
sk_plot_tree(cart, feature_names=list(oh_tr.columns), class_names=list(cart.classes_),
             filled=True, rounded=True, impurity=False, fontsize=7, ax=axes[0])
axes[0].set_title(f"scikit-learn CART + one-hot encoding (pruning tuned by CV)\n{leaves_cart} leaves, test accuracy {acc_cart:.3f}",
                  fontsize=13)
c5.plot_tree(class_names=list(c5.classes_), filled=True, rounded=True, impurity=False,
             fontsize=9, ax=axes[1])
axes[1].set_title(f"c50py (native categorical splits, tuned by CV)\n{leaves_c5} leaves, test accuracy {acc_c5:.3f}",
                  fontsize=13)
fig.tight_layout()
fig.savefig("docs/img/cart_vs_c50py.png", dpi=110)
