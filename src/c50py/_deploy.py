# -*- coding: utf-8 -*-
"""
Export of rules to SQL, JSON and pandas query strings, so that a fitted model
can be applied where Python does not run (a database, a CRM, a BI tool) and
read by other programs.

* Rules of a tree (``C5Classifier``, ``C5Regressor``): one rule per leaf.  A
  row with a missing value (``NULL`` / ``NaN``) on a tested column follows the
  branch that held more training cases, exactly as ``apply`` and
  ``apply_rules`` do.  (``predict`` instead mixes both branches, so for rows
  with missing values the exported rules and ``predict`` can differ.)
* Rulesets (``C5RulesClassifier``): the SQL reproduces ``predict`` exactly:
  each rule a row satisfies votes for its class with its confidence, the class
  with most votes wins (ties go to the default class, then to the first class),
  and rows no rule covers get the default class.  A test on a missing value is
  not satisfied.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ._validation import class_name


# --------------------------------------------------------------------------- literals
def _plain(v: Any):
    if isinstance(v, np.generic):
        v = v.item()
    return v


def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_value(v: Any) -> str:
    v = _plain(v)
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return format(float(v), ".15g") if isinstance(v, float) else str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _py_value(v: Any) -> str:
    v = _plain(v)
    return repr(v)


def _sorted(values):
    return sorted((_plain(v) for v in values), key=lambda x: (str(type(x)), str(x)))


# --------------------------------------------------------------------------- tree paths
def tree_paths(root, *, leaf_value):
    """[(tests, leaf)] in ``export_rules`` order.  A test is
    ``(feature_index, op, value, missing_here)`` where ``missing_here`` says
    whether a missing value is routed to this branch."""
    out = []

    def walk(node, tests):
        if node.is_leaf or not node.children:
            out.append((tests, leaf_value(node)))
            return
        pl, pr = node.branch_weights if node.branch_weights is not None else (0.5, 0.5)
        left_missing = pl >= pr
        j = node.feature_index
        if node.split_type == "numeric":
            t = float(node.threshold)
            walk(node.children["left"], tests + [(j, "<=", t, left_missing)])
            walk(node.children["right"], tests + [(j, ">", t, not left_missing)])
        else:
            S = _sorted(node.threshold)
            walk(node.children["left"], tests + [(j, "in", S, left_missing)])
            walk(node.children["right"], tests + [(j, "not in", S, not left_missing)])

    walk(root, [])
    return out


def _test_sql(name, op, value, missing_here):
    col = sql_ident(name)
    if op in ("<=", ">"):
        core = f"{col} {op} {sql_value(value)}"
    else:
        core = f"{col} {'IN' if op == 'in' else 'NOT IN'} ({', '.join(sql_value(v) for v in value)})"
    if missing_here is None:
        return core                     # NULL makes the test unknown, i.e. not satisfied
    if missing_here:
        return f"({core} OR {col} IS NULL)"
    return f"({core} AND {col} IS NOT NULL)" if op == "not in" else core


def _test_pandas(name, op, value, missing_here):
    col = f"`{name}`"
    if op in ("<=", ">"):
        core = f"{col} {op} {format(float(value), '.15g')}"
    else:
        core = f"{col} {'in' if op == 'in' else 'not in'} [{', '.join(_py_value(v) for v in value)}]"
    known = f"{col} == {col}"           # False for NaN / None
    if missing_here is None:
        return f"({core} and {known})"
    if missing_here:
        return f"({core} or not ({known}))"
    return f"({core} and {known})"


def _test_dict(name, op, value, missing_here):
    d = {"feature": name, "op": op, "value": [_plain(v) for v in value] if op in ("in", "not in") else float(value)}
    if missing_here is not None:
        d["missing_goes_here"] = bool(missing_here)
    return d


# --------------------------------------------------------------------------- trees
def tree_rules_export(model, fmt, feature_names=None, class_names=None, regression=False):
    names = feature_names if feature_names is not None else list(model.feature_names_)
    if regression:
        paths = tree_paths(model.tree_, leaf_value=lambda n: (float(n.predicted_value), float(n.n_samples)))
    else:
        paths = tree_paths(model.tree_, leaf_value=lambda n: (class_name(model, n.predicted_class, class_names),
                                                              {str(_plain(k)): float(v) for k, v in n.class_distribution.items()}))
    if fmt == "json":
        out = []
        for i, (tests, (pred, extra)) in enumerate(paths):
            d = {"rule_id": i, "conditions": [_test_dict(names[j], op, v, m) for j, op, v, m in tests],
                 "prediction": _plain(pred)}
            d["cases" if regression else "class_distribution"] = extra
            out.append(d)
        return out
    if fmt == "pandas":
        return [" and ".join(_test_pandas(names[j], op, v, m) for j, op, v, m in tests) or "index == index"
                for tests, _ in paths]
    raise ValueError(f"unknown format {fmt!r}: use 'text', 'json' or 'pandas'")


def tree_to_sql(model, table, feature_names=None, class_names=None, regression=False):
    names = feature_names if feature_names is not None else list(model.feature_names_)
    if regression:
        paths = tree_paths(model.tree_, leaf_value=lambda n: float(n.predicted_value))
    else:
        paths = tree_paths(model.tree_, leaf_value=lambda n: class_name(model, n.predicted_class, class_names))
    conds = [" AND ".join(_test_sql(names[j], op, v, m) for j, op, v, m in tests) or "TRUE" for tests, _ in paths]
    pred = "\n".join(f"    WHEN {c} THEN {sql_value(p)}" for c, (_, p) in zip(conds, paths))
    rid = "\n".join(f"    WHEN {c} THEN {i}" for i, c in enumerate(conds))
    return (f"-- rules of the tree: one per leaf, mutually exclusive (rule_id = position in export_rules)\n"
            f"SELECT *,\n  CASE\n{rid}\n  END AS rule_id,\n  CASE\n{pred}\n  END AS prediction\n"
            f"FROM {table}")


# --------------------------------------------------------------------------- rulesets
def ruleset_export(model, fmt, feature_names=None, class_names=None):
    names = feature_names if feature_names is not None else list(model.feature_names_)

    def tests(r):
        return [(c.feature, c.op, _sorted(c.value) if c.op in ("in", "not in") else float(c.value)) for c in r.conditions]

    if fmt == "json":
        return {"rules": [{"rule_id": i, "conditions": [_test_dict(names[j], op, v, None) for j, op, v in tests(r)],
                           "prediction": _plain(class_name(model, r.label, class_names)),
                           "vote": int(r.vote), "cases": r.cases, "errors": r.errors,
                           "confidence": round(r.confidence, 4), "lift": round(r.lift, 4)}
                          for i, r in enumerate(model.rules_, start=1)],
                "default": _plain(class_name(model, model.default_class_, class_names)),
                "classes": [_plain(class_name(model, c, class_names)) for c in model.classes_],
                "how_to_predict": "each rule a row satisfies adds its vote to its class; the class with most "
                                  "votes wins (ties: default class, then the first class in 'classes'); "
                                  "a row no rule covers gets the default class; a test on a missing value "
                                  "is not satisfied"}
    if fmt == "pandas":
        return [" and ".join(_test_pandas(names[j], op, v, None) for j, op, v in tests(r)) or "index == index"
                for r in model.rules_]
    raise ValueError(f"unknown format {fmt!r}: use 'text', 'json' or 'pandas'")


def ruleset_to_sql(model, table, feature_names=None, class_names=None):
    names = feature_names if feature_names is not None else list(model.feature_names_)
    classes = list(model.classes_)
    labels = [class_name(model, c, class_names) for c in classes]
    conds = []
    for r in model.rules_:
        t = [_test_sql(names[c.feature], c.op, _sorted(c.value) if c.op in ("in", "not in") else float(c.value), None)
             for c in r.conditions]
        conds.append(" AND ".join(t) or "TRUE")
    vote_cols = []
    for k, c in enumerate(classes):
        terms = [f"CASE WHEN {cond} THEN {int(r.vote)} ELSE 0 END" for cond, r in zip(conds, model.rules_) if r.label == c]
        vote_cols.append("    " + ("\n      + ".join(terms) if terms else "0") + f" AS vote_{k}")
    covered = " OR ".join(f"({c})" for c in conds) or "FALSE"
    d = classes.index(model.default_class_)
    order = [d] + [k for k in range(len(classes)) if k != d]
    whens = [f"    WHEN NOT ({covered}) THEN {sql_value(labels[d])}"]
    for k in order:
        # same rule as predict: the default class wins ties; otherwise the first class with most votes
        if k == d:
            others = [f"vote_{k} >= vote_{o}" for o in range(len(classes)) if o != k]
        else:
            others = [f"vote_{k} {'>' if (o < k or o == d) else '>='} vote_{o}" for o in range(len(classes)) if o != k]
        whens.append(f"    WHEN {' AND '.join(others) or 'TRUE'} THEN {sql_value(labels[k])}")
    lines = "\n".join(whens)
    return (f"-- C5.0 ruleset: rules vote with their confidence (x1000); ties go to the default class,\n"
            f"-- then to the first class; rows covered by no rule get the default class.\n"
            f"WITH votes AS (\n  SELECT *,\n" + ",\n".join(vote_cols) + f"\n  FROM {table}\n)\n"
            f"SELECT *,\n  CASE\n{lines}\n  END AS prediction\nFROM votes")
