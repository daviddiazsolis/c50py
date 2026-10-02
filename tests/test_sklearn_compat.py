"""c50py estimators must work anywhere a scikit-learn estimator works."""
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from c50py import C5Classifier, C5Regressor


_SK = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


# The full battery is run on scikit-learn >= 1.6; older test harnesses break
# with recent joblib even for scikit-learn's own trees (pickle/memmap check).
@pytest.mark.skipif(_SK < (1, 6), reason="estimator checks run on scikit-learn >= 1.6")
@parametrize_with_checks([C5Classifier(), C5Regressor()])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


@pytest.fixture
def churn():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "region": pd.Categorical(rng.choice(["north", "south", "east", "west", "centre"], n)),
        "plan": rng.choice(["basic", "premium"], n),          # object column
        "tenure": rng.integers(1, 60, n).astype(float),
    })
    y = np.where(df.region.isin(["north", "south"]) ^ (df.tenure > 30), "leaves", "stays")
    return df, y


def test_get_params_and_clone():
    clf = C5Classifier(cf=0.1, categorical_features=["region"], feature_names=["a", "b"])
    params = clf.get_params()
    assert params["cf"] == 0.1 and params["feature_names"] == ["a", "b"]
    assert clone(clf).get_params() == params
    assert clone(C5Regressor(min_samples_leaf=5)).get_params()["min_samples_leaf"] == 5


def test_model_selection_and_pipeline(churn):
    X, y = churn
    assert cross_val_score(C5Classifier(), X, y, cv=3).mean() > 0.9
    gs = GridSearchCV(C5Classifier(), {"cf": [0.1, 0.25], "min_samples_leaf": [1, 5]}, cv=3).fit(X, y)
    assert gs.best_score_ > 0.9
    assert make_pipeline(C5Classifier()).fit(X, y).score(X, y) > 0.9
    yr = X.tenure * 2 + X.region.isin(["north"]) * 10
    assert GridSearchCV(C5Regressor(), {"cf": [0.1, 0.25]}, cv=3).fit(X, yr).best_score_ > 0.5


def test_categorical_columns_detected_by_default(churn):
    X, y = churn
    clf = C5Classifier().fit(X, y)
    assert clf.categorical_features_ == [0, 1]
    assert clf.feature_names_ == ["region", "plan", "tenure"]
    assert any("region IN {" in r for r in clf.export_rules())
    reg = C5Regressor().fit(X, X.tenure)
    assert list(reg.is_cat_) == [True, True, False]


def test_strings_without_categorical_raise_clear_error(churn):
    X, y = churn
    with pytest.raises(ValueError, match="categorical_features"):
        C5Classifier(infer_categorical=False).fit(X, y)
    # listing them explicitly works
    C5Classifier(infer_categorical=False, categorical_features=["region", "plan"]).fit(X, y)


def test_string_labels_with_class_names(churn, capsys):
    X, y = churn
    clf = C5Classifier().fit(X, y)
    rules = clf.export_rules(class_names=["LEAVES", "STAYS"])
    assert all(r.endswith(("LEAVES", "STAYS")) for r in rules)
    clf.print_tree(class_names=["LEAVES", "STAYS"])
    assert "Predict" in capsys.readouterr().out


def test_missing_values_including_pd_na(churn):
    X, y = churn
    X = X.copy()
    X.loc[:30, "tenure"] = np.nan
    X["plan"] = pd.array(X["plan"].where(X.index % 7 != 0), dtype="string")  # pd.NA
    clf = C5Classifier().fit(X, y)
    assert clf.score(X, y) > 0.85
    assert np.allclose(clf.predict_proba(X).sum(axis=1), 1.0)


def test_zero_weight_cases_are_ignored():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    w = rng.integers(0, 3, 80)
    a = C5Classifier().fit(X, y, sample_weight=w)
    b = C5Classifier().fit(np.repeat(X, w, axis=0), np.repeat(y, w))
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))
