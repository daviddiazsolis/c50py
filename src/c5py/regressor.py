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

class C5Regressor:
    r"""
    C5Regressor(min_samples_split=2, min_samples_leaf=2, pruning=True,
                cf=0.25, global_pruning=True, categorical_features=None,
                infer_categorical=True, int_as_categorical=False, max_categories=50,
                max_categories_exhaustive=12, mdl_penalty_strength=0.0,
                min_sse_gain=0.0, feature_names=None, random_state=None, verbose=0)

    A C5.0-like regression tree with a scikit-learn–style API.

    **Core behavior**

    - **Split criterion**: weighted **SSE reduction**. Numeric thresholds are evaluated at
      midpoints between distinct sorted values. Categorical features use subset splits
      (exhaustive up to `max_categories_exhaustive`, ordered scan otherwise). An optional
      MDL-like penalty (`mdl_penalty_strength`) helps regularize large-cardinality subsets.
    - **Missing values**: During training, samples with missing values on the splitting
      feature are fractionally assigned to both children in proportion to observed weight.
      During prediction, the output is the weighted combination of both branches using
      the stored branch weights.
    - **Pre-pruning**: `min_samples_split` and `min_samples_leaf` enforced on effective weight.
    - **Post-pruning**: pessimistic pruning controlled by `cf`, implemented as a modest SSE
      inflation (z²·σ²) to penalize small leaves, plus a conservative global merge pass.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum effective weight at a node to allow splitting.
    min_samples_leaf : int, default=2
        Minimum effective weight required in each child after the split.
    pruning : bool, default=True
        Whether to run pessimistic pruning.
    cf : float, default=0.25
        Confidence factor in (0, 1). Larger values prune **more** aggressively.
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
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 2,
                 pruning: bool = True,
                 cf: float = 0.25,
                 global_pruning: bool = True,
                 categorical_features: Optional[Iterable[int | str]] = None,
                 infer_categorical: bool = True,
                 int_as_categorical: bool = False,
                 max_categories: int = 50,
                 max_categories_exhaustive: int = 12,
                 mdl_penalty_strength: float = 0.0,
                 min_sse_gain: float = 0.0,
                 feature_names: Optional[List[str]] = None,
                 random_state: Optional[int] = None,
                 verbose: int = 0, numeric_threshold_strategy: str = 'quantile', max_numeric_thresholds: int = 64):
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.pruning = bool(pruning)
        self.cf = float(cf)
        self.global_pruning = bool(global_pruning)
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self.infer_categorical = bool(infer_categorical)
        self.int_as_categorical = bool(int_as_categorical)
        self.max_categories = int(max_categories)
        self.max_categories_exhaustive = int(max_categories_exhaustive)
        self.mdl_penalty_strength = float(mdl_penalty_strength)
        self.min_sse_gain = float(min_sse_gain)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.random_state = random_state
        self.verbose = int(verbose)

        # Fitted attributes
        self.tree_: Optional[RegrNode] = None
        self.is_cat_: Optional[np.ndarray] = None
        self.cat_values_: Dict[int, Tuple[Any, ...]] = {}
        self.feature_names_: Optional[List[str]] = self.feature_names

    # ----------------------------- Public API -----------------------------

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=object)
        y = _as_float_array(y)
        n, m = X.shape
        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = _as_float_array(sample_weight).copy()
            if w.shape[0] != n:
                raise ValueError("sample_weight must have same length as y")

        if self.feature_names is not None and len(self.feature_names) != m:
            raise ValueError("feature_names length must match X.shape[1]")
        self.feature_names_ = list(self.feature_names) if self.feature_names is not None else None

        self.is_cat_ = self._infer_categorical_features(X)
        # Gather categorical values up to cap
        self.cat_values_.clear()
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
        X = np.asarray(X, dtype=object)
        if self.tree_ is None:
            raise ValueError("Model is not fitted.")
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

        Parameters
        ----------
        feature_names : list[str], optional
            Names for the input features.

        Returns
        -------
        list[str]
            A list of rule strings.  Each string has the form
            ``"<antecedent> => value=<prediction> (N=<weight>)"``.

        Raises
        ------
        ValueError
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

    def export_graphviz(self, filename: str = "c5_reg_tree", feature_names: Optional[List[str]] = None,
                        format: str = "png") -> str:
        """
        Export the regression tree to Graphviz format.

        This method mirrors the behaviour of :meth:`C5Classifier.export_graphviz`.  If
        ``format='dot'`` the DOT source is written directly to disk without
        invoking the external ``dot`` binary.  For other formats the method
        attempts to call Graphviz; if it is unavailable the method falls back
        to writing a ``.dot`` file.

        Parameters
        ----------
        filename : str, default="c5_reg_tree"
            Basename of the output file.
        feature_names : list[str], optional
            Names for the input features.  Defaults to those provided at
            construction time.
        format : str, default="png"
            Desired output format for Graphviz; ``'dot'`` writes only a DOT
            file.

        Returns
        -------
        str
            Path to the written file.  If a fallback to ``.dot`` occurs the
            returned path reflects that extension.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        fn = self._maybe_feature_names(feature_names)
        # create graphviz graph lazily; import locally to avoid hard dependency
        try:
            from graphviz import Digraph
        except Exception as e:
            raise RuntimeError("Please install the 'graphviz' Python package.") from e
        dot = Digraph(comment="C5Regressor", format=format)
        self._add_graph_nodes(dot, self.tree_, "root", fn)
        # dot-only output does not require calling the external binary
        if format.lower() == "dot":
            path = f"{filename}.dot"
            dot.save(path)
            return path
        try:
            dot.render(filename, cleanup=True)
            return f"{filename}.{format}"
        except Exception:
            fallback_path = f"{filename}.dot"
            dot.save(fallback_path)
            return fallback_path

    def _add_graph_nodes(self, dot, node, node_id: str, fn=None):
        if node is None:
            dot.node(node_id, "<empty>")
            return
        if node.is_leaf:
            dot.node(node_id, f"Leaf\nvalue={node.predicted_value:.6g}\nN={node.n_samples:.2f}")
            return
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            label = f"{name} <= {node.threshold:.6g}\nN={node.n_samples:.2f}"
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            label = f"{name} in {S}\nN={node.n_samples:.2f}"
        dot.node(node_id, label)
        # children
        left_id = node_id + "L"
        right_id = node_id + "R"
        dot.edge(node_id, left_id, label="True")
        dot.edge(node_id, right_id, label="False")
        self._add_graph_nodes(dot, node.children["left"], left_id, fn)
        self._add_graph_nodes(dot, node.children["right"], right_id, fn)

    # ----------------------------- Core training -----------------------------

    def _infer_categorical_features(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[1]
        is_cat = np.zeros(m, dtype=bool)
        if self.categorical_features is not None:
            try:
                seq = list(self.categorical_features)
            except TypeError:
                seq = [self.categorical_features]
            if len(seq) > 0 and isinstance(seq[0], str):
                if self.feature_names is None:
                    raise ValueError("feature_names must be provided when categorical_features are given by name.")
                name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
                for name in seq:
                    idx = name_to_idx.get(name, None)
                    if idx is not None:
                        is_cat[idx] = True
            else:
                for j in seq:
                    try:
                        is_cat[int(j)] = True
                    except Exception:
                        pass
        if self.infer_categorical:
            for j in range(m):
                col = X[:, j]
                if is_cat[j]:
                    continue
                if col.dtype == object:
                    # consider strings/bools as categorical
                    any_str = any((isinstance(v, str) for v in col if not _isnan_scalar(v)))
                    any_bool = any((isinstance(v, (bool, np.bool_)) for v in col if not _isnan_scalar(v)))
                    if any_str or any_bool:
                        is_cat[j] = True
                elif self.int_as_categorical and np.issubdtype(col.dtype, np.integer):
                    # treat integer as categorical if cardinality is not too large
                    uniq = set(v for v in col if not _isnan_scalar(v))
                    if len(uniq) <= self.max_categories:
                        is_cat[j] = True
        return is_cat

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
        n, m = X.shape
        best = None
        best_score = -float("inf")

        # Precompute totals for missing assignment
        for j in range(m):
            col = X[:, j]
            is_cat = bool(self.is_cat_[j])
            known_mask = np.array([not _isnan_scalar(v) for v in col])
            w_known = w * known_mask
            w_miss = w * (~known_mask)
            sw_known = float(w_known.sum())
            sw_miss = float(w_miss.sum())
            if sw_known <= 0.0:  # no known values -> cannot split on this feature
                continue

            if not is_cat:
                # numeric split
                vals = col[known_mask].astype(float, copy=False)
                yy = y[known_mask]
                ww = w[known_mask]
                if len(vals) <= 1:
                    continue
                # sort by vals
                order = np.argsort(vals, kind="mergesort")
                v = vals[order]
                yk = yy[order]
                wk = ww[order]

                # prefix sums
                sw = np.cumsum(wk)
                sy = np.cumsum(wk * yk)
                sy2 = np.cumsum(wk * yk * yk)
                SW = float(sw[-1]); SY = float(sy[-1]); SY2 = float(sy2[-1])

                # candidates at boundaries where value changes
                # mid = (v[i] + v[i+1]) / 2
                unique_boundaries = np.nonzero(v[:-1] != v[1:])[0]
                if unique_boundaries.size == 0:
                    continue

                for i in unique_boundaries:
                    swL = float(sw[i]);    syL = float(sy[i]);    sy2L = float(sy2[i])
                    swR = SW - swL;        syR = SY - syL;        sy2R = SY2 - sy2L
                    if swL <= 0 or swR <= 0:
                        continue
                    # proportions for missing
                    pl = swL / (swL + swR)
                    pr = 1.0 - pl
                    # augment with missing
                    swL_eff = swL + pl * sw_miss
                    swR_eff = swR + pr * sw_miss
                    syL_eff = syL + pl * float((w_miss * y).sum())
                    syR_eff = syR + pr * float((w_miss * y).sum())
                    sy2L_eff = sy2L + pl * float((w_miss * y * y).sum())
                    sy2R_eff = sy2R + pr * float((w_miss * y * y).sum())

                    # SSE children
                    sseL = sy2L_eff - (syL_eff * syL_eff) / swL_eff
                    sseR = sy2R_eff - (syR_eff * syR_eff) / swR_eff
                    gain_raw = sse_parent - (sseL + sseR)

                    # Score (no MDL penalty for numeric)
                    score = gain_raw

                    if score > best_score and gain_raw > 0:
                        thr = 0.5 * (v[i] + v[i+1])
                        # masks on known
                        left_known = np.zeros(n, dtype=bool); right_known = np.zeros(n, dtype=bool)
                        # apply only on known
                        mask_known_glob = known_mask
                        left_known[mask_known_glob] = (vals <= thr)
                        right_known[mask_known_glob] = ~left_known[mask_known_glob]
                        best = (j, "numeric", float(thr), float(gain_raw), float(pl), left_known, right_known, ~known_mask)
                        best_score = score

            else:
                # categorical split
                # collect per-category aggregates
                cats = []
                agg = {}  # cat -> (sw, sy, sy2)
                for i, v in enumerate(col):
                    wi = w[i]
                    if wi <= 0: 
                        continue
                    if _isnan_scalar(v):
                        continue
                    agg.setdefault(v, [0.0, 0.0, 0.0])
                    a = agg[v]
                    a[0] += wi
                    a[1] += wi * y[i]
                    a[2] += wi * y[i] * y[i]
                cats = list(agg.keys())
                k = len(cats)
                if k <= 1:
                    continue

                # exhaustion vs ordered scan
                def eval_subset(selected: set) -> Tuple[float, float, float, float, set]:
                    # return (gain_raw, score, pl, subset_set)
                    # compute left sums by summing selected categories
                    swL = sum(agg[c][0] for c in selected)
                    syL = sum(agg[c][1] for c in selected)
                    sy2L = sum(agg[c][2] for c in selected)
                    swR = sum(agg[c][0] for c in cats) - swL
                    syR = sum(agg[c][1] for c in cats) - syL
                    sy2R = sum(agg[c][2] for c in cats) - sy2L
                    if swL <= 0 or swR <= 0:
                        return (-1e18, -1e18, 0.5, 0.0, selected)

                    pl = swL / (swL + swR)
                    # missing augmentation
                    sw_miss = float((w * (~known_mask)).sum())
                    sy_miss = float((w * (~known_mask) * y).sum())
                    sy2_miss = float((w * (~known_mask) * y * y).sum())
                    swL_eff = swL + pl * sw_miss
                    swR_eff = swR + (1.0 - pl) * sw_miss
                    syL_eff = syL + pl * sy_miss
                    syR_eff = syR + (1.0 - pl) * sy_miss
                    sy2L_eff = sy2L + pl * sy2_miss
                    sy2R_eff = sy2R + (1.0 - pl) * sy2_miss
                    sseL = sy2L_eff - (syL_eff * syL_eff) / swL_eff
                    sseR = sy2R_eff - (syR_eff * syR_eff) / swR_eff
                    gain_raw = sse_parent - (sseL + sseR)

                    # MDL-like penalty
                    if self.mdl_penalty_strength > 0.0:
                        s = len(selected)
                        penalty = math.log2(_choose(k, s) + 1e-9)
                        score = gain_raw - self.mdl_penalty_strength * penalty
                    else:
                        score = gain_raw
                    return (gain_raw, score, pl, 0.0, selected)

                best_local = None
                best_local_score = -float("inf")

                if k <= self.max_categories_exhaustive:
                    # exhaustive (avoid symmetric duplicates by fixing first category in left)
                    cats_sorted = sorted(cats, key=lambda c: str(c))
                    fixed = cats_sorted[0]
                    cat_idx = {c:i for i,c in enumerate(cats_sorted)}
                    total_masks = (1 << (k - 1)) - 1  # exclude empty and full
                    for mask in range(1, total_masks):
                        selected = {fixed}
                        for i in range(1, k):
                            if (mask >> (i - 1)) & 1:
                                selected.add(cats_sorted[i])
                        if 0 < len(selected) < k:
                            graw, score, pl, _, sel = eval_subset(selected)
                            if score > best_local_score and graw > 0:
                                best_local_score = score
                                best_local = (graw, score, pl, sel)
                else:
                    # ordered scan by category mean (weighted)
                    stats = []
                    for c in cats:
                        swc, syc, _ = agg[c]
                        mu = syc / swc if swc > 0 else 0.0
                        stats.append((mu, c))
                    stats.sort()
                    ordered = [c for _, c in stats]
                    # prefix splits
                    for t in range(1, k):  # 1..k-1
                        selected = set(ordered[:t])
                        graw, score, pl, _, sel = eval_subset(selected)
                        if score > best_local_score and graw > 0:
                            best_local_score = score
                            best_local = (graw, score, pl, sel)

                if best_local is not None:
                    graw, score, pl, sel = best_local[0], best_local[1], best_local[2], best_local[3]
                    # Build masks for known values
                    left_known = np.zeros(n, dtype=bool); right_known = np.zeros(n, dtype=bool)
                    for i, v in enumerate(col):
                        if _isnan_scalar(v):
                            continue
                        if v in sel:
                            left_known[i] = True
                        else:
                            right_known[i] = True
                    miss_mask = np.array([_isnan_scalar(v) for v in col])
                    if score > best_score:
                        best_score = score
                        best = (j, "categorical", set(sel), float(graw), float(pl), left_known, right_known, miss_mask)

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