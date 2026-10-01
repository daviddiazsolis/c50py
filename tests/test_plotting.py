"""c50py draws trees exactly like sklearn.tree (plot_tree / export_graphviz / export_text)."""
import numpy as np
import pytest

from c50py import C5Classifier, C5Regressor, plot_tree, export_graphviz, export_text
from c50py.tree import TreeNode
from c50py.regressor import RegrNode

sk_tree = pytest.importorskip("sklearn.tree")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _mirror_classifier(sk, n_features):
    """A C5Classifier whose tree is a node-by-node copy of a fitted sklearn tree."""
    t = sk.tree_
    m = C5Classifier()
    m.classes_, m.n_features_, m.feature_names_, m.trials = sk.classes_, n_features, None, 1

    def build(i):
        n = TreeNode(is_leaf=t.children_left[i] == -1)
        v = t.value[i][0]
        # scikit-learn >= 1.4 stores class fractions in tree_.value; older versions store counts
        counts = v * t.weighted_n_node_samples[i] if np.isclose(v.sum(), 1.0) else v
        n.class_distribution = {c: float(v) for c, v in zip(sk.classes_, counts)}
        n.predicted_class = sk.classes_[np.argmax(counts)]
        if not n.is_leaf:
            n.feature_index, n.threshold, n.split_type = int(t.feature[i]), float(t.threshold[i]), "numeric"
            n.children = {"left": build(t.children_left[i]), "right": build(t.children_right[i])}
        return n

    m.tree_ = build(0)
    m.ensemble_ = [m.tree_]
    return m


def _mirror_regressor(sk, n_features):
    t = sk.tree_
    m = C5Regressor()
    m.n_features_, m.feature_names_ = n_features, None

    def build(i):
        n = RegrNode(is_leaf=t.children_left[i] == -1, predicted_value=float(t.value[i][0, 0]),
                     n_samples=float(t.n_node_samples[i]), sse=float(t.impurity[i] * t.n_node_samples[i]))
        if not n.is_leaf:
            n.feature_index, n.threshold, n.split_type = int(t.feature[i]), float(t.threshold[i]), "numeric"
            n.children = {"left": build(t.children_left[i]), "right": build(t.children_right[i])}
        return n

    m.tree_ = build(0)
    return m


def _signature(anns):
    out = []
    for a in anns:
        bp = a.get_bbox_patch()
        out.append((a.get_text(), tuple(np.round(a.xy, 6)), round(a.get_fontsize(), 4),
                    None if bp is None else tuple(np.round(bp.get_facecolor(), 4))))
    return out


OPTIONS = [dict(), dict(filled=True), dict(filled=True, rounded=True, proportion=True),
           dict(impurity=False, node_ids=True, label="root"), dict(max_depth=1, filled=True, precision=2)]


@pytest.mark.parametrize("opts", OPTIONS)
def test_classifier_drawing_identical_to_sklearn(opts):
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    sk = sk_tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(X, y)
    c5 = _mirror_classifier(sk, X.shape[1])
    kw = dict(opts, feature_names=["sl", "sw", "pl", "pw"], class_names=["a", "b", "c"])
    _, ax1 = plt.subplots(figsize=(12, 6))
    _, ax2 = plt.subplots(figsize=(12, 6))
    assert _signature(sk_tree.plot_tree(sk, ax=ax1, **kw)) == _signature(plot_tree(c5, ax=ax2, **kw))
    plt.close("all")
    assert sk_tree.export_graphviz(sk, **kw) == export_graphviz(c5, **kw)
    assert sk_tree.export_text(sk, feature_names=kw["feature_names"]) == export_text(c5, feature_names=kw["feature_names"])


@pytest.mark.parametrize("opts", OPTIONS)
def test_regressor_drawing_identical_to_sklearn(opts):
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    sk = sk_tree.DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
    c5 = _mirror_regressor(sk, X.shape[1])
    _, ax1 = plt.subplots(figsize=(12, 6))
    _, ax2 = plt.subplots(figsize=(12, 6))
    assert _signature(sk_tree.plot_tree(sk, ax=ax1, **opts)) == _signature(c5.plot_tree(ax=ax2, **opts))
    plt.close("all")
    assert sk_tree.export_graphviz(sk, **opts) == c5.export_graphviz(**opts)


def test_real_c5_tree_with_categories_and_missing():
    rng = np.random.default_rng(0)
    n = 300
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = rng.choice(["a", "b", "c"], n)
    y = np.where((X[:, 0].astype(float) > 0) | (X[:, 1] == "c"), "yes", "no")
    X[rng.random(n) < 0.1, 0] = np.nan
    clf = C5Classifier(categorical_features=[1], min_samples_leaf=5).fit(X, y, feature_names=["num", "cat"])
    _, ax = plt.subplots()
    anns = clf.plot_tree(filled=True, class_names=["no", "yes"], ax=ax)
    plt.close("all")
    texts = " ".join(a.get_text() for a in anns)
    assert "entropy = " in texts and "class = " in texts
    dot = clf.export_graphviz(filled=True)
    assert dot.startswith("digraph Tree {") and "fillcolor" in dot
    rep = clf.export_text(show_weights=True)
    assert rep.startswith("|--- ")


def test_feature_names_length_checked():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    clf = C5Classifier(pruning=False).fit(X, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        clf.export_graphviz(feature_names=["a", "b"])


def test_legacy_graphviz_signature(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    clf = C5Classifier(pruning=False).fit(X, [0, 0, 1, 1])
    path = clf.export_graphviz(str(tmp_path / "t"), format="dot")
    assert path.endswith(".dot") and open(path).read().startswith("digraph Tree")
    assert clf.export_graphviz(format="dot").startswith("digraph Tree")
    out = tmp_path / "t2.dot"
    clf.export_graphviz(str(out))
    assert out.read_text().startswith("digraph Tree")
