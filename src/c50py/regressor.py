"""Clean C5.0-style decision tree regressor (Quinlan-inspired).
This module implements a C5.0-like regression tree with pruning and rule export.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Iterable
import math
import numpy as np

# ----------------------------- Helpers -----------------------------

def _isnan_scalar(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except Exception:
        return False

def _as_float_array(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float)

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0.0 else default

def _wmean(y: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0.0:
        return 0.0
    return float((w * y).sum() / sw)

def _wsse(y: np.ndarray, w: np.ndarray) -> float:
    # SSE = sum w * (y - mean)^2 = (sum w*y^2) - (sum w*y)^2 / (sum w)
    sw = float(w.sum())
    if sw <= 0.0:
        return 0.0
    sy = float((w * y).sum())
    sy2 = float((w * y * y).sum())
    return sy2 - (sy * sy) / sw

def _choose(n: int, k: int) -> float:
    # safe combinatorial count as float
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    if k == 0:
        return 1.0
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return float(num // den)

def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF of standard normal (Acklam's approximation)."""
    # clamp
    p = min(max(p, 1e-12), 1 - 1e-12)
    # coefficients
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ----------------------------- Node -----------------------------

@dataclass
class RegrNode:
    is_leaf: bool
    predicted_value: float
    n_samples: float
    sse: float
    feature_index: Optional[int] = None
    split_type: Optional[str] = None  # "numeric" or "categorical"
    threshold: Optional[Any] = None   # float for numeric; set for categorical (left set)
    children: Optional[Dict[str, 'RegrNode']] = None  # "left", "right"
    branch_weights: Optional[Tuple[float, float]] = None  # (p_left, p_right)

    @property
    def n_leaves(self) -> int:
        if self.is_leaf or not self.children:
            return 1
        return sum(ch.n_leaves for ch in self.children.values())

# ----------------------------- Regressor -----------------------------

from ._export import _TreeExportMixin
from sklearn.base import BaseEstimator, RegressorMixin
from . import _validation as _v
from .tree import _subset_masks

class C5Regressor(_TreeExportMixin, RegressorMixin, BaseEstimator):
    r"""
    C5Regressor(min_samples_split=2, min_samples_leaf=2, pruning=True,
                cf=0.25, global_pruning=True, categorical_features=None,
                infer_categorical=True, int_as_categorical=False, max_categories=50,
                max_categories_exhaustive=12, mdl_penalty_strength=0.0,
                min_sse_gain=0.0, feature_names=None, random_state=None, verbose=0)

    A C5.0-like regression tree with a scikit-learn–style API.

    **Core behavior**

    - **Split criterion**: weighted **SSE reduction**. Numeric thresholds are evaluated at
      midpoints between distinct sorted values. Categorical features are split into two
      groups of categories: every grouping is evaluated up to `max_categories_exhaustive`
      categories; above that, the categories are ordered by their mean target and the cuts of
      that order are evaluated, which finds the best grouping for squared error (Fisher, 1958).
      Only splits that leave at least `min_samples_leaf` of weight on each side are candidates.
      An optional MDL-like penalty (`mdl_penalty_strength`) regularizes categorical groupings.
    - **Missing values**: During training, samples with missing values on the splitting
      feature are fractionally assigned to both children in proportion to observed weight.
      During prediction, the output is the weighted combination of both branches using
      the stored branch weights.
    - **Pre-pruning**: `min_samples_split` and `min_samples_leaf` enforced on effective weight.
    - **Post-pruning**: a mild bottom-up pruning controlled by `cf` (a subtree is replaced by
      a leaf when its SSE reduction is below about z²·σ² of the node). It removes few splits,
      so `min_samples_leaf` is the parameter to tune, for example by cross-validation.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum effective weight at a node to allow splitting.
    min_samples_leaf : int, default=2
        Minimum effective weight required in each child after the split.
    pruning : bool, default=True
        Whether to run pessimistic pruning.
    cf : float, default=0.25
        Confidence factor in (0, 1), as in C5.0: smaller values prune more.
    global_pruning : bool, default=True
        Apply a simple global merge step after local pruning.
    categorical_features : sequence of int or str, optional
        Indices or names of categorical columns; requires `feature_names` when using names.
    infer_categorical : bool, default=True
        Automatically mark `object`/`category`/`bool` columns as categorical.
    int_as_categorical : bool, default=False
        If True, treat some integer columns as categorical when cardinality is manageable.
    max_categories : int, default=50
        Maximum number of categories stored per feature (safety cap).
    max_categories_exhaustive : int, default=12
        Up to this cardinality, the subset search is exhaustive; above it, an ordered scan is used.
    mdl_penalty_strength : float, default=0.0
        Adds an MDL-like penalty to subset splits on categorical features.
    min_sse_gain : float, default=0.0
        Minimal SSE improvement required to accept a split.
    feature_names : sequence of str, optional
        Column names (used with `categorical_features` by name and in textual exports).
    random_state : int, optional
        Reserved for reproducibility; training itself is deterministic.
    verbose : int, default=0
        Verbosity level (0 = silent).
    numeric_threshold_strategy, max_numeric_thresholds
        Accepted for symmetry with ``C5Classifier``; the regressor currently
        evaluates every numeric threshold.

    Attributes
    ----------
    tree_ : RegrNode
        Root of the trained regression tree.
    is_cat_ : ndarray of shape (n_features,)
        Boolean mask of categorical features.
    cat_values_ : dict[int, tuple]
        Per-feature tuple of known categories seen during fitting (capped).
    """

    def __init__(self,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 pruning=True,
                 cf=0.25,
                 global_pruning=True,
                 categorical_features=None,
                 infer_categorical=True,
                 int_as_categorical=False,
                 max_categories=50,
                 max_categories_exhaustive=12,
                 mdl_penalty_strength=0.0,
                 min_sse_gain=0.0,
                 feature_names=None,
                 random_state=None,
                 verbose=0,
                 numeric_threshold_strategy="all",
                 max_numeric_thresholds=64):
        # scikit-learn convention: store the parameters exactly as given.
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruning = pruning
        self.cf = cf
        self.global_pruning = global_pruning
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.int_as_categorical = int_as_categorical
        self.max_categories = max_categories
        self.max_categories_exhaustive = max_categories_exhaustive
        self.mdl_penalty_strength = mdl_penalty_strength
        self.min_sse_gain = min_sse_gain
        self.feature_names = feature_names
        self.random_state = random_state
        self.verbose = verbose
        self.numeric_threshold_strategy = numeric_threshold_strategy
        self.max_numeric_thresholds = max_numeric_thresholds

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):  # scikit-learn < 1.6
        return {"allow_nan": True}

    # ----------------------------- Public API -----------------------------

    def fit(self, X, y, sample_weight=None, feature_names=None):
        """Build the regression tree.  ``X`` may be a pandas DataFrame with
        numeric, categorical and missing values (see ``C5Classifier.fit``)."""
        X, y, dtypes = _v.validate_fit(self, X, y, y_numeric=True)
        X = np.asarray(X, dtype=object)
        y = _as_float_array(y)
        w = _v.sample_weights(sample_weight, X)
        n, m = X.shape
        self.n_features_ = m
        self.feature_names_ = _v.resolve_feature_names(self, feature_names, m)
        self.is_cat_ = _v.categorical_mask(self, X, dtypes, self.feature_names_)
        self.cat_values_ = {}
        keep = w > 0  # zero-weight cases are ignored, as in C5.0
        if not keep.all():
            X, y, w = X[keep], y[keep], w[keep]
            n = X.shape[0]
        # Gather categorical values up to cap
        for j in range(m):
            if self.is_cat_[j]:
                col = X[:, j]
                known = [v for v in col if not _isnan_scalar(v)]
                # cap
                uniq = []
                seen = set()
                for v in known:
                    if v in seen:
                        continue
                    seen.add(v)
                    uniq.append(v)
                    if len(uniq) >= self.max_categories:
                        break
                self.cat_values_[j] = tuple(uniq)

        # Build tree with full weight vector
        self.tree_ = self._build_tree(X, y, w)
        # Pruning
        if self.pruning and self.tree_ is not None:
            self._prune(self.tree_, X, y, w)
            if self.global_pruning:
                self._global_merge(self.tree_, X, y, w)
        return self

    def predict(self, X):
        X = np.asarray(_v.validate_predict(self, X), dtype=object)
        out = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            out[i] = self._predict_instance(x, self.tree_)
        return out

    # ----------------------------- Pretty / Rules / Graphviz -----------------------------

    def _maybe_feature_names(self, feature_names):
        return feature_names if feature_names is not None else getattr(self, "feature_names_", None)

    def print_tree(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Pretty‑print the fitted regression tree to ``stdout``.

        Parameters
        ----------
        feature_names : list[str], optional
            Alternative names for the features.  Defaults to those provided at
            construction time.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        fn = self._maybe_feature_names(feature_names)
        self._print_node(self.tree_, "", fn)

    def _print_node(self, node, indent="", fn=None):
        if node is None:
            print(f"{indent}<empty>")
            return
        if node.is_leaf:
            print(f"{indent}Predict {node.predicted_value:.4f} (N={node.n_samples:.2f})")
            return
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            print(f"{indent}if {name} <= {node.threshold:.6g}:")
            self._print_node(node.children["left"], indent + "  ", fn)
            print(f"{indent}else:")
            self._print_node(node.children["right"], indent + "  ", fn)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            print(f"{indent}if {name} in {S}:")
            self._print_node(node.children["left"], indent + "  ", fn)
            print(f"{indent}else:")
            self._print_node(node.children["right"], indent + "  ", fn)

    def export_rules(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Export all decision rules in the fitted regression tree.

        Each rule describes a path from the root to a leaf and reports the
        predicted numeric value along with the effective sample weight.  The
        antecedent of a rule is a conjunction of conditions on the input
        features.  Custom feature names can be supplied; if omitted the
        names provided at construction time are used.

            If the model has not been fitted.
        """
        # Guard against calling before fit
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        fn = self._maybe_feature_names(feature_names)
        rules: List[str] = []
        self._collect_rules(self.tree_, [], rules, fn)
        return rules

    def _collect_rules(self, node, parts: List[str], rules: List[str], fn=None):
        if node is None:
            return
        if node.is_leaf:
            antecedent = " AND ".join(parts) if parts else "<root>"
            rules.append(f"{antecedent} => value={node.predicted_value:.6g} (N={node.n_samples:.2f})")
            return
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            left = parts + [f"{name} <= {node.threshold:.6g}"]
            self._collect_rules(node.children["left"], left, rules, fn)
            right = parts + [f"{name} > {node.threshold:.6g}"]
            self._collect_rules(node.children["right"], right, rules, fn)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            left = parts + [f"{name} IN {S}"]
            self._collect_rules(node.children["left"], left, rules, fn)
            right = parts + [f"{name} NOT IN {S}"]
            self._collect_rules(node.children["right"], right, rules, fn)

    def predict_rule(self, X: Iterable[Any], feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Return the decision rule antecedent for each input sample.

        For each row in ``X`` this method traces the path from the root to the
        corresponding leaf in the fitted tree and returns a human‑readable
        string describing the conditions encountered.  When the model has
        not been fitted a ``ValueError`` is raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to trace through the tree.
        feature_names : list[str], optional
            Names for the input features; defaults to those provided at
            construction time.

        Returns
        -------
        list[str]
            A list of antecedent strings, one per input sample.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        Xp = np.asarray(X, dtype=object)
        fn = self._maybe_feature_names(feature_names)
        return [self._trace_rule(x, self.tree_, fn) for x in Xp]

    def _trace_rule(self, x, node, fn=None, parts=None) -> str:
        parts = parts or []
        if node is None:
            return "<empty>"
        if node.is_leaf:
            return " AND ".join(parts) if parts else "<root>"
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            if _isnan_scalar(x[node.feature_index]):
                parts.append(f"{name} MISSING")
                return " AND ".join(parts)
            if float(x[node.feature_index]) <= node.threshold:
                parts.append(f"{name} <= {node.threshold:.6g}")
                return self._trace_rule(x, node.children["left"], fn, parts)
            else:
                parts.append(f"{name} > {node.threshold:.6g}")
                return self._trace_rule(x, node.children["right"], fn, parts)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            in_left = (not _isnan_scalar(x[node.feature_index])) and (x[node.feature_index] in node.threshold)
            parts.append(f"{name} " + ("IN " if in_left else "NOT IN ") + S)
            return self._trace_rule(x, node.children["left" if in_left else "right"], fn, parts)

    # ----------------------------- Core training -----------------------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> RegrNode:
        n_eff = float(w.sum())
        mu = _wmean(y, w)
        sse_parent = _wsse(y, w)
        node = RegrNode(is_leaf=True, predicted_value=mu, n_samples=n_eff, sse=sse_parent)

        # Pre-pruning: need enough effective weight to consider splits
        if n_eff < self.min_samples_split:
            return node

        best = self._best_split(X, y, w, sse_parent)
        if best is None:
            return node
        j, split_type, thr, gain_raw, p_left, left_mask, right_mask, miss_mask = best

        # Enforce min_sse_gain on raw improvement
        if gain_raw < self.min_sse_gain:
            return node

        # Build children weights with fractional missing
        w_left = np.zeros_like(w)
        w_right = np.zeros_like(w)
        w_left[left_mask] = w[left_mask]
        w_right[right_mask] = w[right_mask]
        if miss_mask.any():
            pl = p_left
            pr = 1.0 - pl
            w_left[miss_mask] += w[miss_mask] * pl
            w_right[miss_mask] += w[miss_mask] * pr

        # Child effective weights
        nL = float(w_left.sum())
        nR = float(w_right.sum())
        if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
            return node

        # Create internal node
        node.is_leaf = False
        node.feature_index = j
        node.split_type = "numeric" if split_type == "numeric" else "categorical"
        node.threshold = thr
        node.children = {}
        node.branch_weights = (p_left, 1.0 - p_left)

        # Recurse
        node.children["left"] = self._build_tree(X, y, w_left)
        node.children["right"] = self._build_tree(X, y, w_right)

        # Update node stats (after children) for pruning
        node.n_samples = n_eff
        node.sse = sse_parent
        node.predicted_value = mu
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, sse_parent: float):
        """
        Best split of the cases with positive weight, by reduction of the
        weighted sum of squared errors (SSE).  Cases with a missing value on
        the candidate feature are shared between the two children in
        proportion to the known weight on each side (as in C4.5/C5.0), and a
        split is only a candidate if both children keep at least
        ``min_samples_leaf`` of weight.  Numeric thresholds and categorical
        groupings are evaluated with vectorised prefix sums.
        """
        n = X.shape[0]
        rows = np.flatnonzero(w > 0)
        if rows.size < 2:
            return None
        Xs, ys, ws = X[rows], y[rows], w[rows]
        msl = float(self.min_samples_leaf)
        best, best_score = None, -float("inf")
        # ties (up to rounding) go to the first candidate, so that weighting a
        # case by 2 and duplicating it grow the same tree
        tol = 1e-9 * max(1.0, abs(float(sse_parent)))

        def sse(sw_, sy_, sy2_):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(sw_ > 0, sy2_ - sy_ * sy_ / np.where(sw_ > 0, sw_, 1.0), 0.0)

        for j in range(Xs.shape[1]):
            col = Xs[:, j]
            is_cat = bool(self.is_cat_[j])
            if col.dtype.kind == "f":
                known = ~np.isnan(col)
            else:
                known = np.fromiter((not _isnan_scalar(v) for v in col), count=col.shape[0], dtype=bool)
            if not known.any():
                continue
            wm, ym = ws[~known], ys[~known]
            sw_m, sy_m, sy2_m = float(wm.sum()), float((wm * ym).sum()), float((wm * ym * ym).sum())
            wk, yk = ws[known], ys[known]

            if not is_cat:
                v = col[known].astype(float)
                order = np.argsort(v, kind="mergesort")
                v, yo, wo = v[order], yk[order], wk[order]
                cut = np.nonzero(v[:-1] != v[1:])[0]
                if cut.size == 0:
                    continue
                cw, cy, cy2 = np.cumsum(wo), np.cumsum(wo * yo), np.cumsum(wo * yo * yo)
                SW, SY, SY2 = cw[-1], cy[-1], cy2[-1]
                swL, syL, sy2L = cw[cut], cy[cut], cy2[cut]
                swR, syR, sy2R = SW - swL, SY - syL, SY2 - sy2L
                pl = swL / SW
                left_sets = None
            else:
                vals = col[known]
                cats = sorted(set(vals.tolist()), key=str)
                k = len(cats)
                if k <= 1:
                    continue
                index = {c: i for i, c in enumerate(cats)}
                ci = np.fromiter((index[c] for c in vals), count=vals.shape[0], dtype=int)
                aw = np.bincount(ci, weights=wk, minlength=k)
                ay = np.bincount(ci, weights=wk * yk, minlength=k)
                ay2 = np.bincount(ci, weights=wk * yk * yk, minlength=k)
                if k <= self.max_categories_exhaustive:
                    M = _subset_masks(k)
                else:
                    # ordering the categories by their mean and cutting the
                    # ordered list gives the best SSE grouping (Fisher, 1958)
                    mean = np.where(aw > 0, ay / np.where(aw > 0, aw, 1.0), 0.0)
                    ordered = np.argsort(mean, kind="mergesort")
                    M = np.zeros((k - 1, k))
                    for t in range(1, k):
                        M[t - 1, ordered[:t]] = 1.0
                swL, syL, sy2L = M @ aw, M @ ay, M @ ay2
                SW, SY, SY2 = aw.sum(), ay.sum(), ay2.sum()
                swR, syR, sy2R = SW - swL, SY - syL, SY2 - sy2L
                pl = swL / SW
                left_sets = M
            pr = 1.0 - pl
            swLe, swRe = swL + pl * sw_m, swR + pr * sw_m
            gain = sse_parent - (sse(swLe, syL + pl * sy_m, sy2L + pl * sy2_m)
                                 + sse(swRe, syR + pr * sy_m, sy2R + pr * sy2_m))
            ok = (swL > 0) & (swR > 0) & (swLe >= msl) & (swRe >= msl) & (gain > 0)
            if not ok.any():
                continue
            score = gain.copy()
            if is_cat and self.mdl_penalty_strength > 0.0:
                sizes = left_sets.sum(axis=1)
                kk = left_sets.shape[1]
                score = score - self.mdl_penalty_strength * np.array(
                    [math.log2(_choose(kk, int(s_)) + 1e-9) for s_ in sizes])
            score = np.where(ok, score, -np.inf)
            top = float(score.max())
            i = int(np.flatnonzero(score >= top - tol)[0])
            if not top > best_score + tol:
                continue
            best_score = top
            known_glob = np.zeros(n, dtype=bool); known_glob[rows[known]] = True
            left_known = np.zeros(n, dtype=bool)
            if not is_cat:
                thr = 0.5 * (v[cut[i]] + v[cut[i] + 1])
                left_known[rows[known]] = col[known].astype(float) <= thr
                split = (j, "numeric", float(thr))
            else:
                sel = {cats[c] for c in np.flatnonzero(left_sets[i] > 0)}
                left_known[rows[known]] = np.fromiter((x in sel for x in col[known]), count=int(known.sum()), dtype=bool)
                split = (j, "categorical", sel)
            right_known = known_glob & ~left_known
            miss = np.zeros(n, dtype=bool); miss[rows[~known]] = True
            best = split + (float(gain[i]), float(pl[i]), left_known, right_known, miss)
        return best

    # ----------------------------- Pruning -----------------------------

    def _prune(self, node: RegrNode, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Bottom-up pessimistic pruning using cf -> z multiplier."""
        if node is None or node.is_leaf:
            return
        # Collect weights for children by routing fractionally at this node
        j = node.feature_index
        split_type = node.split_type
        col = X[:, j]
        known_mask = np.array([not _isnan_scalar(v) for v in col])
        left_known = None
        if split_type == "numeric":
            thr = node.threshold
            left_known = np.zeros_like(known_mask)
            left_known[known_mask] = (col[known_mask].astype(float) <= thr)
        else:
            left_known = np.zeros_like(known_mask)
            left_known[known_mask] = np.array([v in node.threshold for v in col[known_mask]], dtype=bool)

        right_known = known_mask & (~left_known)
        miss_mask = ~known_mask
        pl, pr = node.branch_weights if node.branch_weights is not None else (0.5, 0.5)

        w_left = np.zeros_like(w)
        w_right = np.zeros_like(w)
        w_left[left_known] = w[left_known]
        w_right[right_known] = w[right_known]
        if miss_mask.any():
            w_left[miss_mask] += w[miss_mask] * pl
            w_right[miss_mask] += w[miss_mask] * pr

        # Recurse
        self._prune(node.children["left"], X, y, w_left)
        self._prune(node.children["right"], X, y, w_right)

        # Decide pruning at this node
        # Parent leaf SSE (already stored as node.sse)
        sse_parent = node.sse
        # Subtree SSE = sum of leaf SSE under children
        sse_subtree = self._sum_leaf_sse(node)

        # Estimate variance at parent leaf
        n_eff = float(w.sum())
        mu = _wmean(y, w)
        sigma2 = 0.0
        if n_eff > 1.0:
            sigma2 = _wsse(y, w) / max(n_eff - 1.0, 1.0)

        z = _norm_ppf(1.0 - self.cf)
        penalty = (z * z) * sigma2

        if sse_parent <= (sse_subtree + penalty):
            # prune
            node.is_leaf = True
            node.children = None
            node.split_type = None
            node.feature_index = None
            node.threshold = None
            node.branch_weights = None

    def _sum_leaf_sse(self, node: RegrNode) -> float:
        if node is None:
            return 0.0
        if node.is_leaf or not node.children:
            return node.sse
        return self._sum_leaf_sse(node.children["left"]) + self._sum_leaf_sse(node.children["right"])

    def _global_merge(self, node: RegrNode, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        """One-pass global merge: if both children are leaves, compare SSEs and merge if parent leaf better (with small penalty)."""
        if node is None or node.is_leaf or not node.children:
            return
        L = node.children["left"]
        R = node.children["right"]
        self._global_merge(L, X, y, w)  # recurse down
        self._global_merge(R, X, y, w)
        if L.is_leaf and R.is_leaf:
            sse_parent = node.sse
            sse_children = L.sse + R.sse
            z = _norm_ppf(1.0 - self.cf)
            penalty = (z * z) * 0.0  # very small or zero extra penalty here
            if sse_parent <= (sse_children + penalty):
                node.is_leaf = True
                node.children = None
                node.split_type = None
                node.feature_index = None
                node.threshold = None
                node.branch_weights = None

    # ----------------------------- Prediction -----------------------------

    def _predict_instance(self, x, node: RegrNode) -> float:
        if node.is_leaf or not node.children:
            return node.predicted_value
        j = node.feature_index
        if node.split_type == "numeric":
            v = x[j]
            if _isnan_scalar(v):
                pl, pr = node.branch_weights if node.branch_weights is not None else (0.5, 0.5)
                return pl * self._predict_instance(x, node.children["left"]) + \
                       pr * self._predict_instance(x, node.children["right"])
            return self._predict_instance(x, node.children["left" if float(v) <= node.threshold else "right"])
        else:
            v = x[j]
            if _isnan_scalar(v):
                pl, pr = node.branch_weights if node.branch_weights is not None else (0.5, 0.5)
                return pl * self._predict_instance(x, node.children["left"]) + \
                       pr * self._predict_instance(x, node.children["right"])
            return self._predict_instance(x, node.children["left" if v in node.threshold else "right"])