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