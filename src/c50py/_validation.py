# -*- coding: utf-8 -*-
"""
Input validation shared by ``C5Classifier`` and ``C5Regressor``.

These helpers make both estimators behave like any scikit-learn estimator
(``clone``, ``GridSearchCV``, ``cross_val_score``, ``Pipeline``,
``check_estimator``) while keeping what is specific to C5.0: categorical
columns are kept as they are (strings, categories, booleans) and missing
values (``None``, ``np.nan``, ``pd.NA``) are allowed.
"""
from __future__ import annotations

import inspect

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _sk_validate_data

    def _validate_data(est, X, y="no_validation", **kw):
        return _sk_validate_data(est, X, y, **kw)

except ImportError:  # scikit-learn < 1.6

    def _validate_data(est, X, y="no_validation", **kw):
        return est._validate_data(X, y, **kw)


# ``force_all_finite`` was renamed ``ensure_all_finite`` in scikit-learn 1.6.
_FINITE_KW = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(check_array).parameters
    else "force_all_finite"
)

_CATEGORICAL_DTYPES = ("object", "category", "bool", "boolean", "string", "str")


def _is_categorical_dtype_name(name: str) -> bool:
    name = str(name)
    return name in _CATEGORICAL_DTYPES or name.startswith("string")


def _dtypes_of(X):
    """dtype names of a pandas DataFrame (``None`` for anything else)."""
    if hasattr(X, "dtypes") and hasattr(X, "columns"):
        return [str(d) for d in X.dtypes]
    return None


def _normalise_missing(X: np.ndarray) -> np.ndarray:
    """Turn ``pd.NA`` / ``pd.NaT`` / ``None`` in object arrays into ``np.nan``."""
    if X.dtype != object:
        return X
    try:
        import pandas as pd

        mask = pd.isna(X)
    except ImportError:  # pragma: no cover
        mask = np.vectorize(lambda v: v is None or (isinstance(v, float) and np.isnan(v)), otypes=[bool])(X)
    if mask.any():
        X = X.copy()
        X[mask] = np.nan
    return X


def _common_kw():
    return {"dtype": None, _FINITE_KW: "allow-nan", "accept_sparse": False}


def validate_fit(est, X, y, *, y_numeric: bool):
    """Validate ``X`` and ``y`` in ``fit``; sets ``n_features_in_`` (and
    ``feature_names_in_`` for DataFrames with string column names).

    Returns ``(X, y, dtypes)`` where ``dtypes`` are the pandas dtype names
    (or ``None``) used to detect categorical columns.
    """
    dtypes = _dtypes_of(X)
    X, y = _validate_data(est, X, y, reset=True, y_numeric=y_numeric, **_common_kw())
    return _normalise_missing(X), y, dtypes


def validate_predict(est, X):
    """Validate ``X`` in ``predict``/``predict_proba`` (same columns as in ``fit``)."""
    check_is_fitted(est)
    X = _validate_data(est, X, reset=False, **_common_kw())
    return _normalise_missing(X)


def sample_weights(sample_weight, X) -> np.ndarray:
    w = _check_sample_weight(sample_weight, X, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Negative values in sample_weight are not allowed.")
    if not np.any(w > 0):
        raise ValueError("Sample weights must contain at least one non-zero number.")
    return np.array(w, dtype=float, copy=True)


def resolve_feature_names(est, feature_names, n_features: int):
    """Names used in drawings and rules, by priority: the ``feature_names``
    argument of ``fit``, the ``feature_names`` parameter, the DataFrame
    columns, and finally ``f0, f1, ...``."""
    for names in (feature_names, getattr(est, "feature_names", None),
                  getattr(est, "feature_names_in_", None)):
        if names is not None:
            names = [str(n) for n in names]
            if len(names) != n_features:
                raise ValueError(
                    f"feature_names has {len(names)} names but X has {n_features} columns"
                )
            return names
    return [f"f{i}" for i in range(n_features)]


def categorical_mask(est, X: np.ndarray, dtypes, feature_names) -> np.ndarray:
    """Boolean mask of categorical columns.

    A column is categorical when it is listed in ``categorical_features`` (by
    index or name) or, with ``infer_categorical=True`` (the default), when it
    is a pandas ``category``/``object``/``string``/``bool`` column or an object
    column holding strings or booleans. ``int_as_categorical=True`` adds
    integer columns.
    """
    n_features = X.shape[1]
    mask = np.zeros(n_features, dtype=bool)

    cf = getattr(est, "categorical_features", None)
    if cf is not None:
        if isinstance(cf, (str, int, np.integer)):
            cf = [cf]
        name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
        for c in cf:
            if isinstance(c, str):
                if c not in name_to_idx:
                    raise ValueError(
                        f"categorical feature {c!r} is not a column name; known names: {list(name_to_idx)}"
                    )
                mask[name_to_idx[c]] = True
            else:
                j = int(c)
                if not -n_features <= j < n_features:
                    raise ValueError(f"categorical feature index {j} out of range for {n_features} columns")
                mask[j] = True

    infer = bool(getattr(est, "infer_categorical", True))
    int_as_cat = bool(getattr(est, "int_as_categorical", False))
    for j in range(n_features):
        if mask[j]:
            continue
        col = X[:, j]
        if dtypes is not None:
            name = dtypes[j]
            if infer and _is_categorical_dtype_name(name):
                mask[j] = True
                continue
            if infer and int_as_cat and (name.startswith("int") or name.startswith("Int") or name.startswith("uint")):
                mask[j] = True
                continue
        if col.dtype == object:
            has_text = any(isinstance(v, (str, bool, np.bool_)) for v in col)
            if has_text:
                if infer:
                    mask[j] = True
                else:
                    raise ValueError(
                        f"column {feature_names[j]!r} holds strings or booleans; list it in "
                        "categorical_features or use infer_categorical=True"
                    )
        elif infer and int_as_cat and col.dtype.kind in "iu":
            mask[j] = True
    return mask


def class_name(est, label, class_names):
    """Display name of a class label, with ``class_names`` ordered like ``classes_``."""
    if class_names is None:
        return str(label)
    idx = int(np.flatnonzero(est.classes_ == label)[0])
    return str(class_names[idx])
