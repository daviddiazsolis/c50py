"""Behaviour that follows C4.5/C5.0 (Quinlan 1993; C5.0 release 2.07)."""
import numpy as np
import pandas as pd

from c50py import C5Classifier


def _grouped_categories(n=2000, seed=0):
    """8 sectors in two groups with different default rates, plus a noise column."""
    rng = np.random.default_rng(seed)
    sectors = list("ABCDEFGH")
    sizes = np.array([30, 25, 20, 10, 6, 4, 3, 2], float)          # some rare sectors
    s = rng.choice(sectors, n, p=sizes / sizes.sum())
    risky = np.isin(s, ["A", "D", "F", "H"])
    y = np.where(rng.uniform(size=n) < np.where(risky, 0.8, 0.2), "bad", "good")
    X = pd.DataFrame({"sector": pd.Categorical(s), "noise": rng.normal(size=n)})
    return X, y


def test_categories_are_grouped_not_peeled_one_by_one():
    X, y = _grouped_categories()
    clf = C5Classifier().fit(X, y)
    rules = clf.export_rules()
    # one split separates the two groups; no rule tests "sector" twice
    assert all(r.count("sector") == 1 for r in rules), rules
    assert any("{A, D, F, H}" in r or "{B, C, E, G}" in r for r in rules), rules


def test_large_nodes_do_not_split_off_a_few_cases():
    # the 5 largest values are all "b": peeling them off is the best cut by gain,
    # but C4.5/C5.0 require min(25, 10% of the cases per class) on each side
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    y = np.where(rng.uniform(size=1000) < 0.5, "a", "b")
    y[np.argsort(x)[-5:]] = "b"
    X = pd.DataFrame({"x": x})
    kw = dict(pruning=False, mdl_penalty=False, max_depth=1)
    off = C5Classifier(numeric_min_split=False, **kw).fit(X, y)
    on = C5Classifier(**kw).fit(X, y)
    sizes = lambda m: [sum(ch.class_distribution.values()) for ch in m.tree_.children.values()]
    assert min(sizes(off)) < 25
    assert min(sizes(on)) >= 25


def test_subtree_raising_does_not_grow_this_tree():
    X, y = _grouped_categories(seed=3)
    with_r = C5Classifier().fit(X, y)
    without = C5Classifier(subtree_raising=False).fit(X, y)
    assert len(with_r.export_rules()) <= len(without.export_rules())


def test_global_pruning_only_removes_leaves_within_one_standard_error():
    X, y = _grouped_categories(n=3000, seed=5)
    rng = np.random.default_rng(5)
    X = X.assign(n1=rng.normal(size=len(X)), n2=rng.normal(size=len(X)))
    local = C5Classifier(global_pruning=False).fit(X, y)
    both = C5Classifier().fit(X, y)
    assert len(both.export_rules()) <= len(local.export_rules())
    # the extra training errors stay within one standard error of the local tree's errors
    err_local = 1 - local.score(X, y)
    err_both = 1 - both.score(X, y)
    n = len(y)
    assert (err_both - err_local) * n <= np.sqrt(err_local * n * (1 - err_local)) + 1e-9


def test_categorical_groupings_pay_for_the_search():
    # a 12-category column with no signal must not be split just because one of
    # its 2047 groupings looks good by chance
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"c": pd.Categorical(rng.choice(list("abcdefghijkl"), 300))})
    y = np.where(rng.uniform(size=300) < 0.5, "a", "b")
    assert len(C5Classifier(pruning=False).fit(X, y).export_rules()) == 1
    assert len(C5Classifier(pruning=False, mdl_penalty=False).fit(X, y).export_rules()) > 1


def test_winnowing_drops_noise_columns():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": pd.Categorical(rng.choice(list("xyz"), n))})
    for i in range(6):
        X[f"noise{i}"] = rng.normal(size=n)
    y = np.where((X.a > 0) & (X.b != "z"), "p", "n")
    flip = rng.uniform(size=n) < 0.15
    y[flip] = np.where(y[flip] == "p", "n", "p")
    m = C5Classifier(winnow=True).fit(X, y)
    assert set(m.winnowed_features_) == {f"noise{i}" for i in range(6)}
    assert C5Classifier().fit(X, y).winnowed_features_ == []
    # predictions only use the kept columns
    X2 = X.copy()
    X2[[f"noise{i}" for i in range(6)]] = 0.0
    assert (m.predict(X2) == m.predict(X)).all()


def test_boosting_c50_style_improves_and_stops_early():
    X, y = _grouped_categories(n=1500, seed=11)
    rng = np.random.default_rng(11)
    X = X.assign(n1=rng.normal(size=len(X)))
    y = np.where((X.n1 > 0.8) & (y == "good"), "bad", y)          # a second, numeric reason
    one = C5Classifier().fit(X, y)
    boosted = C5Classifier(trials=10).fit(X, y)
    assert 1 <= len(boosted.ensemble_) <= 10
    assert len(boosted.estimator_errors_) == len(boosted.ensemble_)
    assert boosted.score(X, y) >= one.score(X, y)
    # a perfectly separable problem stops after the first tree
    Xs = pd.DataFrame({"x": np.arange(200.0)})
    ys = np.where(Xs.x < 100, "a", "b")
    assert len(C5Classifier(trials=10).fit(Xs, ys).ensemble_) == 1
