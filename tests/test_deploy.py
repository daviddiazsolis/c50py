"""Exports of rules to SQL, JSON and pandas reproduce the models."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from c50py import C5Classifier, C5Regressor, C5RulesClassifier


def _data(n=600, seed=0, missing=0.1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "region": pd.Categorical(rng.choice(["North", "South", "O'Higgins", "East"], n)),
        "plan": pd.Categorical(rng.choice(["pre", "post", "biz"], n)),
        "tenure": rng.integers(1, 100, n).astype(float),
        "spend": rng.normal(50, 15, n).round(1),
    })
    score = (X.region.isin(["North", "O'Higgins"]) * 1.5 + (X.plan == "pre") * 1.0
             - X.tenure / 40 + rng.normal(0, 0.7, n))
    y = np.where(score > 0.5, "leaves", np.where(score > -0.8, "maybe", "stays"))
    for c in X.columns:
        X.loc[rng.random(n) < missing, c] = np.nan
    return X, y


def _sql(query, X):
    con = sqlite3.connect(":memory:")
    D = X.copy()
    for c in D.columns:
        if str(D[c].dtype) == "category":
            D[c] = D[c].astype(object)
    D.to_sql("data", con, index=False)
    return pd.read_sql(query, con)


def test_tree_sql_matches_apply_rules():
    X, y = _data()
    m = C5Classifier(min_samples_leaf=5).fit(X, y)
    out = _sql(m.to_sql(), X)
    ref = m.apply_rules(X)
    assert (out.rule_id.to_numpy() == ref.rule_id.to_numpy()).all()
    assert (out.prediction.to_numpy() == ref.prediction.to_numpy()).all()


def test_tree_pandas_queries_partition_the_rows():
    X, y = _data()
    m = C5Classifier(min_samples_leaf=5).fit(X, y)
    ids = np.full(len(X), -1)
    for k, q in enumerate(m.export_rules(format="pandas")):
        hit = X.query(q).index
        assert (ids[hit] == -1).all()          # rules of the tree are mutually exclusive
        ids[hit] = k
    assert (ids == m.apply(X)).all()


def test_ruleset_sql_matches_predict():
    X, y = _data(seed=1)
    r = C5RulesClassifier(min_samples_leaf=5).fit(X, y)
    out = _sql(r.to_sql(), X)
    assert (out.prediction.to_numpy() == r.predict(X)).all()


def test_json_exports_are_serialisable():
    X, y = _data(seed=2)
    m = C5Classifier(min_samples_leaf=5).fit(X, y)
    r = C5RulesClassifier(min_samples_leaf=5).fit(X, y)
    tree_json = json.loads(json.dumps(m.export_rules(format="json")))
    assert len(tree_json) == len(m.export_rules())
    rs = json.loads(json.dumps(r.export_ruleset(format="json")))
    assert len(rs["rules"]) == len(r.rules_) and rs["default"] == r.default_class_


def test_regressor_sql_matches_predict_without_missing():
    X, _ = _data(seed=3, missing=0.0)
    yv = X.tenure.to_numpy() * 0.5 + np.where(X.region == "North", 10.0, 0.0)
    m = C5Regressor(min_samples_leaf=10).fit(X, yv)
    out = _sql(m.to_sql(), X)
    assert np.allclose(out.prediction.to_numpy(), m.predict(X))


def test_unknown_format():
    X, y = _data()
    m = C5Classifier().fit(X, y)
    with pytest.raises(ValueError):
        m.export_rules(format="xml")
