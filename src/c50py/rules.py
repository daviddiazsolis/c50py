# -*- coding: utf-8 -*-
"""
c50py.rules
===========

Rulesets in the style of C5.0 (``rules=True`` in the R package C50).

A decision tree can be read as **rules of the tree**: one rule per leaf, the
conjunction of the tests on the path from the root.  Those rules are mutually
exclusive, so each case follows exactly one (``C5Classifier.export_rules``,
``C5Classifier.apply_rules``).

A **ruleset** is a different model built from them, following C5.0's
``rules`` mode (Quinlan 1993, ch. 5, and the later C5.0 release):

1. every node of the pruned tree except the root gives a candidate rule: the
   tests on its path, predicting the node's majority class; tests on the same
   column are merged (``x <= 5 AND x <= 3`` becomes ``x <= 3``);
2. each candidate is *simplified*: tests are dropped one at a time, the one
   whose removal gives the lowest Laplace error rate first, until the rule is
   worth its description length and every further removal would raise its
   error rate; rules that are not worth stating, or barely better than the
   class prior, are discarded;
3. a subset of the candidates is selected that minimises a description
   length: the bits to state the rules (discounted) plus the bits to point out
   and correct the training cases they get wrong, where a case is classified
   by the votes of the rules it satisfies;
4. a default class is chosen for cases that no rule covers.

Rules of a ruleset may overlap: a case can satisfy several rules, possibly of
different classes.  Each votes for its class with its Laplace confidence and
the class with most votes is predicted, as in C5.0.  A test on a missing
value is not satisfied.  A test ``x NOT IN {a}`` is written ``x IN {b, c}``
(the other categories seen in training) when that is shorter; a category
never seen in training then satisfies neither.

The implementation is written independently, in Python and numpy, from the
published descriptions of the method; its results are checked against the R
package C50 in the benchmark of the repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets

from . import _validation as _v
from .tree import C5Classifier, _add_errs, _isnan_scalar

__all__ = ["C5RulesClassifier", "Rule"]

_LN2 = log(2.0)


# --------------------------------------------------------------------------- rules
@dataclass
class Condition:
    """One test of a rule: ``feature <= value``, ``feature > value``,
    ``feature in {...}`` or ``feature not in {...}``."""

    feature: int
    op: str            # "<=", ">", "in", "not in"
    value: Any         # float threshold or frozenset of categories

    def mask(self, X: np.ndarray, cache: dict) -> np.ndarray:
        key = (self.feature, self.op, self.value)
        if key in cache:
            return cache[key]
        col = X[:, self.feature]
        if self.op in ("<=", ">"):
            v = _as_float(col, cache, self.feature)
            with np.errstate(invalid="ignore"):
                m = v <= self.value if self.op == "<=" else v > self.value
        else:
            known = _known(col, cache, self.feature)
            inside = np.zeros(col.shape[0], dtype=bool)
            inside[known] = np.isin(col[known], list(self.value))
            m = inside if self.op == "in" else known & ~inside
        cache[key] = m
        return m

    def text(self, names) -> str:
        name = names[self.feature] if names is not None else f"X[{self.feature}]"
        if self.op in ("<=", ">"):
            return f"{name} {self.op} {self.value:.6g}"
        cats = ", ".join(map(str, sorted(self.value, key=str)))
        return f"{name} {'IN' if self.op == 'in' else 'NOT IN'} {{{cats}}}"


def _as_float(col, cache, j):
    key = ("float", j)
    if key not in cache:
        if col.dtype.kind == "f":
            cache[key] = col
        else:
            cache[key] = np.array([np.nan if _isnan_scalar(v) else float(v) for v in col], dtype=float)
    return cache[key]


def _known(col, cache, j):
    key = ("known", j)
    if key not in cache:
        if col.dtype.kind == "f":
            cache[key] = ~np.isnan(col)
        else:
            cache[key] = np.fromiter((not _isnan_scalar(v) for v in col), count=col.shape[0], dtype=bool)
    return cache[key]


@dataclass
class Rule:
    """A rule ``IF conditions THEN label`` with its training statistics."""

    conditions: list
    label: Any
    cases: float = 0.0           # (weighted) training cases covered
    errors: float = 0.0          # covered cases of another class
    confidence: float = 0.0      # Laplace: (cases - errors + 1) / (cases + 2)
    lift: float = 0.0            # confidence / prior probability of the class
    class_distribution: dict = field(default_factory=dict)
    vote: int = 0                # Laplace confidence in thousandths, used to vote

    def mask(self, X, cache) -> np.ndarray:
        m = np.ones(X.shape[0], dtype=bool)
        for c in self.conditions:
            m &= c.mask(X, cache)
        return m

    def text(self, names) -> str:
        if not self.conditions:
            return "<always>"
        return " AND ".join(c.text(names) for c in self.conditions)


def _merge_conditions(conds: list) -> list:
    """Merge tests on the same column into one (tightest interval / set)."""
    out, by_feat = [], {}
    for c in conds:
        by_feat.setdefault(c.feature, []).append(c)
    for j, cs in by_feat.items():
        lo = max((c.value for c in cs if c.op == ">"), default=None)
        hi = min((c.value for c in cs if c.op == "<="), default=None)
        if lo is not None:
            out.append(Condition(j, ">", lo))
        if hi is not None:
            out.append(Condition(j, "<=", hi))
        ins = [c.value for c in cs if c.op == "in"]
        outs = [c.value for c in cs if c.op == "not in"]
        if ins:
            allowed = frozenset.intersection(*map(frozenset, ins))
            if outs:
                allowed = allowed - frozenset.union(*map(frozenset, outs))
            out.append(Condition(j, "in", allowed))
        elif outs:
            out.append(Condition(j, "not in", frozenset.union(*map(frozenset, outs))))
    first = {}
    for i, c in enumerate(conds):            # keep the order in which columns first appear
        first.setdefault(c.feature, i)
    return sorted(out, key=lambda c: (first[c.feature], c.op))


def _tree_rules(tree: C5Classifier) -> list:
    """The rules of the tree: one per leaf, in the order of ``export_rules``."""
    rules = []

    def walk(node, conds):
        if node.is_leaf:
            rules.append(Rule(list(conds), node.predicted_class))
            return
        j = node.feature_index
        if node.split_type == "numeric":
            walk(node.children["left"], conds + [Condition(j, "<=", float(node.threshold))])
            walk(node.children["right"], conds + [Condition(j, ">", float(node.threshold))])
        else:
            S = frozenset(node.threshold)
            walk(node.children["left"], conds + [Condition(j, "in", S)])
            walk(node.children["right"], conds + [Condition(j, "not in", S)])

    walk(tree.tree_, [])
    return rules


# --------------------------------------------------------------------------- ruleset construction
THEORY_FRAC = 0.23     # weight of the cost of stating the rules against the cost of their errors
MIN_ITEMS = 2          # a rule must get at least this many training cases right


def _log2_fact(n: float) -> float:
    return lgamma(n + 1.0) / _LN2 if n > 0 else 0.0


def _log2_binom(n: float, k: float) -> float:
    if k <= 0 or k >= n:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / _LN2


def _split_info(a: float, b: float) -> float:
    """Bits needed to say which of a + b cases belong to the first group."""
    def f(x):
        return x * np.log2(x) if x > 0 else 0.0
    return f(a + b) - f(a) - f(b)


class _Coder:
    """Description length, in bits, of a test: which column it tests (log2 of
    the number of columns) plus which threshold or which categories."""

    def __init__(self, X, cache, is_cat):
        n, p = X.shape
        self.column_bits = float(np.log2(max(p, 1)))
        self.value_bits, self.present = {}, {}
        for j in range(p):
            col = X[:, j]
            known = _known(col, cache, j)
            if is_cat[j]:
                vals, counts = np.unique(np.asarray(col[known].tolist(), dtype=object).astype(str),
                                         return_counts=True)
                self.present[j] = len(vals)
                q = counts / max(n, 1)
                # one category: its information content in the column, on average
                self.value_bits[j] = float(-(q * np.log2(q)).sum()) if len(q) else 0.0
            else:
                cuts = len(np.unique(_as_float(col, cache, j)[known])) - 1
                self.value_bits[j] = 1.0 + 0.5 * np.log2(cuts) if cuts > 1 else 0.0

    def bits(self, c: Condition) -> float:
        if c.op in ("<=", ">"):
            return self.column_bits + self.value_bits[c.feature]
        m = self.present.get(c.feature, 2)
        k = len(c.value) if c.op == "in" else m - len(c.value)
        if c.op == "in" and len(c.value) == 1:
            return self.column_bits + self.value_bits[c.feature]
        k = min(max(k, 1), max(m - 1, 1))
        return self.column_bits + _log2_binom(m, k)

    def rule_bits(self, conds) -> float:
        k = len(conds)
        return sum(self.bits(c) for c in conds) + (np.log2(k) if k > 0 else 0.0) - _log2_fact(k)


def _laplace(errors: float, total: float) -> float:
    return 0.5 if total < 1e-9 else (errors + 1.0) / (total + 2.0)


def _simplify(conds, label, X, pos, w, coder, cache, class_weight, N):
    """
    Drop the tests of a candidate rule that do not pay for themselves.

    At each step the test whose removal gives the lowest Laplace error rate
    ``(errors + 1) / (cases + 2)`` is found.  It is removed unless the rule is
    already worth stating (its information about the class, in bits, is at
    least THEORY_FRAC times the bits needed to state its tests) and removing
    the test would make the error rate worse.  Returns the remaining tests and
    whether the rule is good enough to be a candidate for the ruleset.
    """
    conds = list(conds)
    if not conds:
        return conds, False
    M = np.array([c.mask(X, cache) for c in conds])
    neg = ~pos
    base = _split_info(class_weight, N - class_weight)
    while True:
        fails = (~M).sum(axis=0)
        sat = fails == 0
        tot0, err0 = float(w[sat].sum()), float(w[sat & neg].sum())
        one = fails == 1
        tot = np.array([tot0 + float(w[one & ~M[d]].sum()) for d in range(len(conds))])
        err = np.array([err0 + float(w[one & ~M[d] & neg].sum()) for d in range(len(conds))])
        pess0 = _laplace(min(err0, tot0), tot0)
        pess = np.array([_laplace(min(e, t), t) for e, t in zip(err, tot)])
        right0 = tot0 - err0
        gain = base - _split_info(right0, err0) - _split_info(class_weight - right0, N - class_weight - err0)
        cost = sum(coder.bits(c) for c in conds) - _log2_fact(len(conds))
        best = len(pess) - 1 - int(np.argmin(pess[::-1]))       # ties: the last test
        worth = THEORY_FRAC * cost <= gain
        if len(conds) == 1 or (worth and pess[best] > pess0):
            break
        del conds[best]
        M = np.delete(M, best, axis=0)
    ok = tot0 > 0.99 and worth
    return conds, ok


def _candidates(tree, X, y, w, coder, cache, readable, min_leaf):
    """Candidate rules: the path to every node of the tree except the root,
    simplified; duplicates are dropped."""
    N = float(w.sum())
    class_weight = {k: float(w[y == k].sum()) for k in tree.classes_}
    seen, out = {}, []

    def total(node):
        return float(sum(node.class_distribution.values())) if node.class_distribution else 0.0

    def visit(node, path):
        if not node.is_leaf:
            j = node.feature_index
            if node.split_type == "numeric":
                tests = [Condition(j, "<=", float(node.threshold)), Condition(j, ">", float(node.threshold))]
            else:
                S = frozenset(node.threshold)
                tests = [Condition(j, "in", S), Condition(j, "not in", S)]
            for branch, test in zip(("left", "right"), tests):
                child = node.children[branch]
                if total(child) < min_leaf:
                    continue
                visit(child, path + [test])
        if path and total(node) >= 1:
            label = node.predicted_class
            pos = y == label
            conds = readable(_merge_conditions(path))
            conds, ok = _simplify(conds, label, X, pos, w, coder, cache, class_weight[label], N)
            if not ok:
                return
            m = Rule(conds, label).mask(X, cache)
            cover, correct = float(w[m].sum()), float(w[m & pos].sum())
            prior = class_weight[label] / N
            if (correct + 1.0) / ((cover + 2.0) * prior) < 0.95:
                return
            vote = int(1000.0 * (correct + 1.0) / (cover + 2.0) + 0.5)
            key = (label, frozenset((c.feature, c.op, c.value) for c in conds))
            if key in seen:
                r = out[seen[key]]
                r.vote = max(r.vote, vote)
                return
            r = Rule(conds, label, cases=cover, errors=cover - correct)
            r.vote = vote
            seen[key] = len(out)
            out.append(r)

    visit(tree.tree_, [])
    return out


def _message_length(n_rules, rule_bits, errs, n_cases, bits_err, bits_ok, class_bits):
    """Total description length (in hundredths of a bit, as an integer):
    the rules, discounted by THEORY_FRAC and by the irrelevance of their
    order, plus the bits to point out and correct the training errors."""
    theory = THEORY_FRAC * max(0.0, rule_bits - _log2_fact(n_rules))
    return int(round(100.0 * (theory + errs * bits_err + (n_cases - errs) * bits_ok + errs * class_bits)))


def _select(cands, X, y, w, classes, coder, cache, est_err_rate):
    """
    Choose the ruleset: start from a greedy cover of each class, then add or
    drop one rule at a time while the description length goes down.  A case
    is classified by the votes of the rules it satisfies (each rule votes for
    its class with its Laplace confidence); a case no rule covers counts as
    an error.
    """
    R, n, K = len(cands), X.shape[0], len(classes)
    if R == 0:
        return []
    cls_index = {k: i for i, k in enumerate(classes)}
    yi = np.array([cls_index[v] for v in y])
    F = np.array([r.mask(X, cache) for r in cands])                    # (R, n)
    lab = np.array([cls_index[r.label] for r in cands])
    vote = np.array([r.vote for r in cands], dtype=np.int64)
    correct = np.array([r.cases - r.errors for r in cands])
    bits = np.array([coder.rule_bits(r.conditions) for r in cands])
    W = float(w.sum())

    # initial theory: per class, add the most confident rules while they add coverage
    inside = np.zeros(R, dtype=np.int8)                                # 0 out, 1 in, 2 tried
    for k in range(K):
        covered = np.zeros(n, dtype=bool)
        remaining, false_pos = float(w[yi == k].sum()), 0.0
        while remaining > false_pos:
            pool = np.where((lab == k) & (inside == 0) & (correct >= MIN_ITEMS))[0]
            if pool.size == 0:
                break
            b = pool[int(np.argmax(vote[pool]))]
            new = F[b] & ~covered
            tp, fp = float(w[new & (yi == k)].sum()), float(w[new & (yi != k)].sum())
            if tp - fp <= MIN_ITEMS + 1e-9:
                inside[b] = 2
            else:
                remaining -= tp
                false_pos += fp
                inside[b] = 1
                covered |= F[b]
    sel = inside == 1

    rate = min(est_err_rate, 0.45) if est_err_rate > 0.5 else est_err_rate
    rate = min(max(rate, 1e-6), 1 - 1e-6)
    bits_err, bits_ok = -np.log2(rate), -np.log2(1.0 - rate)
    class_bits = float(np.log2(K - 1)) if K > 2 else 0.0

    V = np.zeros((n, K), dtype=np.int64)
    for r in np.where(sel)[0]:
        V[F[r], lab[r]] += vote[r]

    def wrong(Vrows, yrows):
        return (Vrows.max(axis=1) == 0) | (Vrows.argmax(axis=1) != yrows)

    errs = float(w[wrong(V, yi)].sum())

    def delta(r):
        rows = F[r]
        Vr = V[rows].copy()
        Vr[:, lab[r]] += -vote[r] if sel[r] else vote[r]
        return float(w[rows][wrong(Vr, yi[rows])].sum() - w[rows][wrong(V[rows], yi[rows])].sum())

    last = -1
    while True:
        count, rbits = int(sel.sum()), float(bits[sel].sum())
        current = _message_length(count, rbits, errs, W, bits_err, bits_ok, class_bits)
        best, best_cost, best_delta = -1, current, 0.0
        for r in range(R):
            if r == last:
                continue
            if sel[r]:
                d = delta(r)
                alt = _message_length(count - 1, rbits - bits[r], errs + d, W, bits_err, bits_ok, class_bits)
            else:
                if errs < 1e-3:
                    continue
                d = delta(r)
                alt = _message_length(count + 1, rbits + bits[r], errs + d, W, bits_err, bits_ok, class_bits)
            if alt < best_cost or (alt == best_cost and sel[r]):
                best, best_cost, best_delta = r, alt, d
        if best < 0:
            break
        V[F[best], lab[best]] += -vote[best] if sel[best] else vote[best]
        sel[best] = not sel[best]
        errs += best_delta
        last = best
    return [cands[r] for r in np.where(sel)[0]]


def build_ruleset(tree: C5Classifier, X: np.ndarray, y: np.ndarray, w: np.ndarray, cf: float = 0.25):
    """Rules of ``tree`` -> simplified, selected ruleset.

    Returns ``(rules, default class, default class scores)``."""
    classes = tree.classes_
    cache: dict = {}
    n_features = X.shape[1]
    is_cat = list(getattr(tree, "is_cat_", [False] * n_features))
    coder = _Coder(X, cache, is_cat)

    # categories seen in training, to state "not in {a}" as "in {b, c}" when shorter
    categories = {j: frozenset(X[_known(X[:, j], cache, j), j].tolist())
                  for j in range(n_features) if is_cat[j]}

    def readable(conds):
        out = []
        for c in conds:
            if c.op == "not in" and c.feature in categories:
                rest = categories[c.feature] - c.value
                if rest and len(rest) <= len(c.value):
                    c = Condition(c.feature, "in", rest)
            out.append(c)
        return out

    min_leaf = float(getattr(tree, "min_samples_leaf", 2))
    cands = _candidates(tree, X, y, w, coder, cache, readable, min_leaf)

    # expected error rate of the ruleset: that of the tree on its training cases
    def leaf_errors(node):
        if node.is_leaf:
            d = node.class_distribution or {}
            return float(sum(d.values()) - max(d.values(), default=0.0))
        return sum(leaf_errors(ch) for ch in node.children.values())
    N, K = float(w.sum()), len(classes)
    est = (leaf_errors(tree.tree_) + K - 1) / (N + K)
    selected = _select(cands, X, y, w, classes, coder, cache, est)

    # statistics, confidence, lift
    prior = {k: float(w[y == k].sum()) / N for k in classes}
    for r in selected:
        m = r.mask(X, cache)
        r.cases = float(w[m].sum())
        r.errors = r.cases - float(w[m & (y == r.label)].sum())
        r.confidence = (r.cases - r.errors + 1.0) / (r.cases + 2.0)
        r.lift = r.confidence / prior[r.label] if prior[r.label] > 0 else 0.0
        r.class_distribution = {k: float(w[m & (y == k)].sum()) for k in classes}
    selected.sort(key=lambda r: (-r.confidence, -r.cases))

    # default class: uncovered cases, smoothed, plus the prior of the class
    covered = np.zeros(X.shape[0], dtype=bool)
    for r in selected:
        covered |= r.mask(X, cache)
    unc = {k: float(w[~covered & (y == k)].sum()) for k in classes}
    tot_unc = 1e-3 + sum(unc.values())
    score = {k: (unc[k] + 1.0) / (tot_unc + 2.0) + prior[k] for k in classes}
    default = classes[0]
    for k in classes:
        if score[k] > score[default]:
            default = k
    return selected, default, score


# --------------------------------------------------------------------------- estimator
class C5RulesClassifier(ClassifierMixin, BaseEstimator):
    """
    Ruleset classifier in the style of C5.0's ``rules`` mode.

    A :class:`C5Classifier` tree is grown and pruned with the same parameters,
    and its rules are generalised and selected into a compact ruleset (see the
    module documentation).  The ruleset is the model: ``predict`` and
    ``predict_proba`` use the rules, not the tree.

    Parameters
    ----------
    cf, min_samples_leaf, min_samples_split, max_depth, global_pruning, subtree_raising,
    numeric_min_split, mdl_penalty, gain_ratio_avg_gain, categorical_features,
    infer_categorical, int_as_categorical, max_categories_exhaustive, winnow, feature_names
        As in :class:`C5Classifier`.  They control the tree the rules are
        built from; ``cf`` (pruning of that tree) is the main one.
    class_weight : dict, "balanced" or None, default=None
        As in :class:`C5Classifier`.  With class weights, the cases, errors
        and confidence reported for each rule are weighted.

    Attributes
    ----------
    rules_ : list of Rule
        The ruleset, sorted by confidence.  Each rule has ``conditions``,
        ``label``, ``cases``, ``errors``, ``confidence`` (Laplace estimate) and
        ``lift`` (confidence / prior of the class).
    default_class_ : label
        Prediction for cases that no rule covers.
    tree_model_ : C5Classifier
        The tree the rules come from.
    classes_, n_features_in_, feature_names_, feature_names_in_
        As in scikit-learn.

    Notes
    -----
    Tests on a missing value are not satisfied, so a case with missing values
    may be covered by fewer rules (and fall to the default class).
    """

    def __init__(
        self,
        *,
        cf=0.25,
        min_samples_leaf=2,
        min_samples_split=2,
        max_depth=None,
        global_pruning=True,
        subtree_raising=True,
        numeric_min_split=True,
        mdl_penalty=True,
        gain_ratio_avg_gain=True,
        categorical_features=None,
        infer_categorical=True,
        int_as_categorical=False,
        max_categories_exhaustive=12,
        winnow=False,
        feature_names=None,
        class_weight=None,
    ):
        self.cf = cf
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.global_pruning = global_pruning
        self.subtree_raising = subtree_raising
        self.numeric_min_split = numeric_min_split
        self.mdl_penalty = mdl_penalty
        self.gain_ratio_avg_gain = gain_ratio_avg_gain
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.int_as_categorical = int_as_categorical
        self.max_categories_exhaustive = max_categories_exhaustive
        self.winnow = winnow
        self.feature_names = feature_names
        self.class_weight = class_weight

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"allow_nan": True}

    def _tree_params(self):
        return {k: getattr(self, k) for k in (
            "cf", "min_samples_leaf", "min_samples_split", "max_depth", "global_pruning",
            "subtree_raising", "numeric_min_split", "mdl_penalty", "gain_ratio_avg_gain",
            "categorical_features", "infer_categorical", "int_as_categorical",
            "max_categories_exhaustive", "winnow", "feature_names")}

    def fit(self, X, y, sample_weight=None):
        Xraw = X
        X, y, _ = _v.validate_fit(self, X, y, y_numeric=False)
        check_classification_targets(y)
        w = _v.sample_weights(sample_weight, X)
        if self.class_weight is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            w = w * compute_sample_weight(self.class_weight, y)
        tree = C5Classifier(**self._tree_params()).fit(Xraw, y, sample_weight=w)
        self._set_from_tree(tree, X, y, w)
        return self

    def _set_from_tree(self, tree, X, y, w):
        self.tree_model_ = tree
        self.classes_ = tree.classes_
        self.feature_names_ = tree.feature_names_
        self.n_features_ = X.shape[1]
        keep = w > 0
        X, y, w = X[keep], y[keep], w[keep]
        self.rules_, self.default_class_, score = build_ruleset(tree, X, y, w, float(self.cf))
        s = np.array([score[c] for c in self.classes_], dtype=float)
        # class scores for cases no rule covers (their maximum is the default class)
        self.default_distribution_ = s / s.sum()
        return self

    # ----------------------------------------------------------------- prediction
    def _fired(self, X):
        cache: dict = {}
        return np.array([r.mask(X, cache) for r in self.rules_]).reshape(len(self.rules_), X.shape[0])

    def _votes(self, F):
        """Sum of the votes (Laplace confidences) of the rules that fire, per class."""
        k = len(self.classes_)
        V = np.zeros((F.shape[1], k))
        index = {c: i for i, c in enumerate(self.classes_)}
        for r, fired in zip(self.rules_, F):
            V[fired, index[r.label]] += r.vote
        return V

    def predict_proba(self, X):
        """
        Class probabilities.  A case that no rule covers gets the scores of
        the default class.  Otherwise each rule it satisfies contributes its
        class distribution on the training cases (Laplace-smoothed), weighted
        by its vote; the class with most votes (the prediction, as in C5.0)
        is guaranteed to have the highest probability.
        """
        X = _v.validate_predict(self, X)
        F = self._fired(X)
        k = len(self.classes_)
        out = np.tile(self.default_distribution_, (X.shape[0], 1))
        if F.size:
            D = np.array([[(r.class_distribution.get(c, 0.0) + 1.0) / (r.cases + k) for c in self.classes_]
                          for r in self.rules_])
            vote = np.array([r.vote for r in self.rules_], dtype=float)
            wts = F.T * vote[None, :]                  # (n, R)
            s = wts.sum(axis=1)
            hit = s > 0
            out[hit] = (wts[hit] @ D) / s[hit, None]
            # make the voted class the most probable one
            V = self._votes(F)[hit]
            best = self._vote_winner(V)
            P = out[hit]
            rows = np.arange(P.shape[0])
            top = P.argmax(axis=1)
            swap = top != best
            if swap.any():
                a, b = P[rows[swap], top[swap]].copy(), P[rows[swap], best[swap]].copy()
                P[rows[swap], top[swap]], P[rows[swap], best[swap]] = b, a
            out[hit] = P
        return out

    def _vote_winner(self, V):
        """Class with most votes; ties go to the default class, then to the first class."""
        d = int(np.where(self.classes_ == self.default_class_)[0][0])
        best = np.full(V.shape[0], d)
        for c in range(V.shape[1]):
            better = V[np.arange(V.shape[0]), c] > V[np.arange(V.shape[0]), best]
            best[better] = c
        return best

    def predict(self, X):
        proba = self.predict_proba(X)          # checks that the model is fitted
        return self.classes_[np.argmax(proba, axis=1)]

    def export_ruleset(self, *, feature_names=None, class_names=None, as_frame=False, format="text"):
        """
        The ruleset, one rule per line (or a DataFrame with ``as_frame=True``):
        ``Rule k: IF <tests> THEN <class>  [cases, errors, confidence, lift]``.
        Rules are sorted by confidence; the last line is the default class.

        ``format="json"`` returns a dict with the rules (conditions,
        prediction, vote, statistics), the default class and how to combine
        the votes, for other programs; ``format="pandas"`` returns one
        ``DataFrame.query`` string per rule.  See also :meth:`to_sql`.
        """
        if format != "text":
            from ._deploy import ruleset_export
            return ruleset_export(self, format, feature_names, class_names)
        names = feature_names if feature_names is not None else self.feature_names_
        rows = []
        for i, r in enumerate(self.rules_, start=1):
            rows.append(dict(rule_id=i, conditions=r.text(names),
                             prediction=_v.class_name(self, r.label, class_names),
                             cases=round(r.cases, 2), errors=round(r.errors, 2),
                             confidence=round(r.confidence, 3), lift=round(r.lift, 2)))
        if as_frame:
            import pandas as pd
            return pd.DataFrame(rows)
        out = [f"Rule {d['rule_id']}: IF {d['conditions']} THEN {d['prediction']}  "
               f"[cases={d['cases']:g}, errors={d['errors']:g}, confidence={d['confidence']:.3f}, "
               f"lift={d['lift']:.2f}]" for d in rows]
        out.append(f"Default: {_v.class_name(self, self.default_class_, class_names)}")
        return out

    def to_sql(self, table="data", *, feature_names=None, class_names=None):
        """
        The ruleset as one SQL query that reproduces :meth:`predict`: every
        rule adds its vote to its class, the class with most votes wins (ties
        go to the default class, then to the first class), and rows that no
        rule covers get the default class.  Returns
        ``SELECT *, vote_0, vote_1, ..., prediction``.
        """
        if not hasattr(self, "rules_"):
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        from ._deploy import ruleset_to_sql
        return ruleset_to_sql(self, table, feature_names, class_names)

    def apply_ruleset(self, X, *, feature_names=None, class_names=None):
        """
        For each row: the strongest rule it satisfies among those that predict
        the winning class (highest confidence), how many rules it satisfies,
        and the prediction.  Rows that satisfy no
        rule get ``rule_id = 0`` and the default class.  Returns a DataFrame.
        """
        import pandas as pd
        index = X.index if hasattr(X, "index") and hasattr(X, "columns") else None
        X = _v.validate_predict(self, X)
        names = feature_names if feature_names is not None else self.feature_names_
        F = self._fired(X)
        pred = self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        n_fired = F.sum(axis=0) if F.size else np.zeros(X.shape[0], dtype=int)
        first = np.zeros(X.shape[0], dtype=int)
        if F.size:
            agree = F & (np.array([r.label for r in self.rules_], dtype=object)[:, None] == pred[None, :])
            first = np.where(agree.any(axis=0), np.argmax(agree, axis=0) + 1, 0)
        text = ["<default>" if k == 0 else self.rules_[k - 1].text(names) for k in first]
        return pd.DataFrame({
            "rule_id": first,
            "rule": text,
            "n_rules": n_fired,
            "prediction": [_v.class_name(self, p, class_names) for p in pred],
        }, index=index)
