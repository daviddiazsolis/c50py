"""Rules of the tree (one per leaf) and C5.0-style rulesets."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from c50py import C5Classifier, C5RulesClassifier


@pytest.fixture
def churn():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({
        "plan": pd.Categorical(rng.choice(["basic", "plus", "premium", "family"], n)),
        "region": pd.Categorical(rng.choice(["north", "south", "east", "west"], n)),
        "tenure": rng.integers(1, 72, n).astype(float),
        "complaints": rng.poisson(0.6, n).astype(float),
    })
    # two reasons to leave: new customers on the basic plan, and repeated complaints
    p = 0.05 + 0.6 * ((X.plan == "basic") & (X.tenure < 24)) + 0.45 * (X.complaints >= 2)
    y = np.where(rng.uniform(size=n) < p, "churn", "stay")
    return X, y


def test_apply_matches_export_rules(churn):
    X, y = churn
    tree = C5Classifier().fit(X, y)
    ids = tree.apply(X)
    rules = tree.export_rules()
    assert ids.min() >= 0 and ids.max() < len(rules)
    # the prediction of the rule a row follows is the tree's prediction
    preds = np.array([rules[i].rsplit(" => ", 1)[1] for i in ids])
    assert (preds == tree.predict(X)).all()
    table = tree.apply_rules(X.head(10))
    assert list(table.columns) == ["rule_id", "rule", "prediction"]
    assert (table.rule_id.to_numpy() == ids[:10]).all()
    assert (table.index == X.head(10).index).all()


def test_ruleset_is_compact_and_accurate(churn):
    X, y = churn
    tree = C5Classifier().fit(X, y)
    rs = C5RulesClassifier().fit(X, y)
    assert 0 < len(rs.rules_) <= len(tree.export_rules())
    assert rs.score(X, y) >= tree.score(X, y) - 0.03
    assert np.allclose(rs.predict_proba(X).sum(axis=1), 1.0)
    text = rs.export_ruleset()
    assert text[-1].startswith("Default:") and text[0].startswith("Rule 1: IF ")
    frame = rs.export_ruleset(as_frame=True)
    assert {"rule_id", "conditions", "prediction", "cases", "errors", "confidence", "lift"} <= set(frame.columns)
    assert frame.confidence.is_monotonic_decreasing


def test_ruleset_finds_the_two_churn_reasons(churn):
    X, y = churn
    rs = C5RulesClassifier().fit(X, y)
    churn_rules = [r.text(rs.feature_names_) for r in rs.rules_ if r.label == "churn"]
    assert any("complaints" in r for r in churn_rules)
    assert any("plan" in r and "tenure" in r for r in churn_rules)


def test_apply_ruleset_assigns_strongest_rule_or_default(churn):
    X, y = churn
    rs = C5RulesClassifier().fit(X, y)
    table = rs.apply_ruleset(X)
    assert list(table.columns) == ["rule_id", "rule", "n_rules", "prediction"]
    assert (table.prediction.to_numpy() == rs.predict(X)).all()
    assert ((table.rule_id == 0) == (table.n_rules == 0)).all()
    assert (table.loc[table.rule_id == 0, "rule"] == "<default>").all()


def test_build_ruleset_from_fitted_tree_equals_fitting_the_ruleset(churn):
    X, y = churn
    tree = C5Classifier().fit(X, y)
    a = tree.build_ruleset(X, y)
    b = C5RulesClassifier().fit(X, y)
    assert a.export_ruleset() == b.export_ruleset()


def test_ruleset_in_model_selection(churn):
    X, y = churn
    assert cross_val_score(clone(C5RulesClassifier(cf=0.1)), X, y, cv=3).mean() > 0.7


def test_conditions_on_the_same_column_are_merged():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.uniform(0, 10, 2000)})
    y = np.where((X.x > 2) & (X.x <= 4), "a", "b")
    rs = C5RulesClassifier().fit(X, y)
    for r in rs.rules_:
        feats = [c.feature for c in r.conditions]
        ops = [c.op for c in r.conditions]
        assert len(set(zip(feats, ops))) == len(ops)       # no repeated (column, op)
