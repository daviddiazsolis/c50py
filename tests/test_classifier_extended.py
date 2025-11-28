import numpy as np
import os
import pytest
from c5py import C5Classifier


def _tiny_dataset():
    """Return a small classification dataset with a numeric and categorical feature."""
    X = np.array([[1, 'A'], [2, 'A'], [3, 'B'], [4, 'B']], dtype=object)
    y = np.array([0, 0, 1, 1])
    return X, y


def test_classifier_proba_sums_to_one():
    X, y = _tiny_dataset()
    clf = C5Classifier(trials=1, min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    # probabilities for each row should sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_rule_export():
    X, y = _tiny_dataset()
    clf = C5Classifier(trials=1, min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    clf.fit(X, y)
    # trace rule for each sample
    rules = clf.predict_rule(X, feature_names=['num', 'cat'])
    assert len(rules) == len(X)
    # export full tree rules
    tree_rules = clf.export_rules(feature_names=['num', 'cat'], class_names=['no', 'yes'])
    assert len(tree_rules) > 0
    # each exported rule should contain implication symbol
    assert all('=>' in r for r in tree_rules)


def test_classifier_graphviz_export():
    pytest.importorskip("graphviz")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = C5Classifier(min_samples_split=2).fit(X, y)
    
    # Just check it runs without error
    clf.export_graphviz("test_tree", format="dot")
    import os
    if os.path.exists("test_tree.dot"):
        os.remove("test_tree.dot")
    # export Graphviz in dot format – should not require external graphviz binary
    out_path = clf.export_graphviz('test_tree', feature_names=['num', 'cat'],
                                   class_names=['no', 'yes'], format='dot')
    # returned filename must end with .dot
    assert out_path.endswith('.dot')
    # ensure file was written
    assert os.path.exists(out_path)
    # cleanup the generated file
    os.remove(out_path)


def test_classifier_not_fitted_raises():
    clf = C5Classifier()
    with pytest.raises(ValueError):
        clf.predict([[1, 'A']])
    with pytest.raises(ValueError):
        clf.predict_rule([[1, 'A']])


def test_classifier_trials_rule_error():
    X, y = _tiny_dataset()
    # build a boosted ensemble
    clf = C5Classifier(trials=2, min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1], random_state=0)
    clf.fit(X, y)
    # rule tracing and other exports should be unavailable for ensembles
    with pytest.raises(ValueError):
        clf.predict_rule(X)
    with pytest.raises(ValueError):
        clf.export_rules()
    with pytest.raises(ValueError):
        clf.export_graphviz()
    with pytest.raises(ValueError):
        clf.print_tree()


def test_classifier_max_depth():
    X, y = _tiny_dataset()
    # enforce maximum depth of 1
    clf = C5Classifier(trials=1, min_samples_split=2, min_samples_leaf=1,
                       max_depth=1, feature_names=['num', 'cat'], categorical_features=[1])
    clf.fit(X, y)
    # should still produce valid predictions without infinite recursion
    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_classifier_with_missing_values():
    # dataset containing missing values (None)
    X = np.array([[1, 'A'], [2, None], [3, 'B'], [None, 'A']], dtype=object)
    y = np.array([0, 0, 1, 1])
    clf = C5Classifier(trials=1, min_samples_split=2, min_samples_leaf=1,
                       feature_names=['num', 'cat'], categorical_features=[1])
    clf.fit(X, y)
    preds = clf.predict(X)
    # predictions should be of correct length
    assert len(preds) == len(y)