"""class_weight for the classifiers, apply / apply_rules for the regressor."""
import numpy as np
import pandas as pd
import pytest

from c50py import C5Classifier, C5Regressor, C5RulesClassifier


def _imbalanced(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(0, 1, n), "c": pd.Categorical(rng.choice(list("xyz"), n))})
    y = np.where(rng.random(n) < 0.08, "rare", "common")
    y[(X.c == "x").to_numpy() & (rng.random(n) < 0.3)] = "rare"
    return X, y


@pytest.mark.parametrize("Est", [C5Classifier, C5RulesClassifier])
def test_balanced_class_weight_raises_minority_recall(Est):
    X, y = _imbalanced()
    plain = Est().fit(X, y).predict(X)
    weighted = Est(class_weight="balanced").fit(X, y).predict(X)
    rare = y == "rare"
    assert (weighted[rare] == "rare").mean() > (plain[rare] == "rare").mean()


def test_class_weight_equals_sample_weight():
    X, y = _imbalanced(seed=1)
    cw = {"rare": 4.0, "common": 1.0}
    a = C5Classifier(class_weight=cw).fit(X, y)
    b = C5Classifier().fit(X, y, sample_weight=np.where(y == "rare", 4.0, 1.0))
    assert a.export_rules() == b.export_rules()


def test_regressor_apply_rules_match_export_and_sql_order():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 300), "c": pd.Categorical(rng.choice(list("xyz"), 300))})
    X.loc[::13, "a"] = np.nan
    y = np.where(X.c == "x", 5.0, 0.0) + X.a.fillna(0.5).to_numpy() * 3
    m = C5Regressor(min_samples_leaf=10).fit(X, y)
    ids = m.apply(X)
    rules = m.export_rules()
    assert ids.min() >= 0 and ids.max() < len(rules)
    out = m.apply_rules(X)
    assert list(out.index) == list(X.index)
    assert (out.rule_id.to_numpy() == ids).all()
    complete = X.notna().all(axis=1).to_numpy()
    assert np.allclose(out.prediction.to_numpy()[complete], m.predict(X)[complete])
