import numpy as np
import os
import pytest
from c50py import C5Regressor


def _tiny_reg_dataset():
    """Return a small regression dataset with a numeric and categorical feature."""
    X = np.array([[1.0, 'A'], [2.0, 'A'], [3.0, 'B'], [4.0, 'B']], dtype=object)
    y = np.array([1.0, 1.5, 2.0, 2.5])
    return X, y


def test_regressor_predictions_shape():
    X, y = _tiny_reg_dataset()
    regr = C5Regressor(min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    regr.fit(X, y)
    pred = regr.predict(X)
    # predictions should have same length as input
    assert pred.shape == y.shape


def test_regressor_rule_and_export():
    X, y = _tiny_reg_dataset()
    regr = C5Regressor(min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    regr.fit(X, y)
    rules = regr.predict_rule(X, feature_names=['num', 'cat'])
    assert len(rules) == len(X)
    exported = regr.export_rules(feature_names=['num', 'cat'])
    # exported rules should include antecedent and value
    assert any('value=' in r for r in exported)


def test_regressor_graphviz_export():
    pytest.importorskip("graphviz")
    X = np.array([[1], [2], [3]])
    y = np.array([1.1, 2.1, 3.1])
    reg = C5Regressor(min_samples_split=2).fit(X, y)
    
    reg.export_graphviz("test_reg_tree", format="dot")
    import os
    if os.path.exists("test_reg_tree.dot"):
        os.remove("test_reg_tree.dot")


def test_regressor_not_fitted():
    regr = C5Regressor()
    with pytest.raises(ValueError):
        regr.predict([[1.0, 'A']])


def test_regressor_missing_values():
    X = np.array([[1.0, 'A'], [2.0, None], [3.0, 'B'], [None, 'A']], dtype=object)
    y = np.array([1.0, 1.0, 2.0, 2.0])
    regr = C5Regressor(min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    regr.fit(X, y)
    pred = regr.predict(X)
    assert len(pred) == len(y)

def test_regressor_split_search_respects_min_samples_leaf():
    """A split that would leave too few cases on one side is not a candidate;
    the tree keeps growing with the best admissible split instead of stopping."""
    import numpy as np
    from c50py import C5Regressor
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 1, 400)
    x2 = rng.uniform(0, 1, 400)
    y = 5.0 * (x1 > 0.5) + rng.normal(0, 0.1, 400)
    y[0] += 50.0                                    # an outlier that a 1-case leaf would isolate
    X = np.c_[x1, x2]
    m = C5Regressor(min_samples_leaf=20, pruning=False).fit(X, y)
    assert not m.tree_.is_leaf
    assert m.score(X, y) > 0.5


def test_regressor_single_category_against_the_rest():
    """The grouping {one category} vs {all others} is evaluated."""
    import numpy as np
    import pandas as pd
    from c50py import C5Regressor
    rng = np.random.default_rng(1)
    cat = rng.choice(list("abcd"), 400)
    y = np.where(cat == "a", 10.0, 0.0) + rng.normal(0, 0.1, 400)
    X = pd.DataFrame({"c": pd.Categorical(cat)})
    m = C5Regressor(min_samples_leaf=5).fit(X, y)
    root = m.tree_
    assert not root.is_leaf
    assert set(root.threshold) in ({"a"}, {"b", "c", "d"})
