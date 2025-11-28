# -*- coding: utf-8 -*-
"""
c5py.tree
=========

This module implements a C5.0‑style decision tree classifier inspired by the
work of Quinlan.  It supports both numeric and categorical predictors, missing
values, pre‑pruning via ``min_samples_split``/``min_samples_leaf``, post‑pruning
via a confidence factor, optional global pruning, AdaBoost‑style boosting
(the ``trials`` argument) and a scikit‑learn–like API.

In addition to the core training and prediction routines, the classifier
provides utilities for rule tracing, rule export, pretty printing of the tree
and Graphviz export.  These helpers operate only when the classifier is a
single tree (``trials=1``) – boosting ensembles cannot be unrolled into a
single set of rules.

The module also contains a private ``TreeNode`` class which holds the data
structure for each node in the tree (internal or leaf).
"""

# ---
# The implementation below follows scikit‑learn conventions.  It was adapted
# from an earlier prototype and cleaned to improve readability and maintainability.


# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from itertools import combinations


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _isnan_scalar(v) -> bool:
    return (v is None) or (isinstance(v, float) and np.isnan(v))

def _entropy(dist_vec: np.ndarray) -> float:
    tot = dist_vec.sum()
    if tot <= 0:
        return 0.0
    p = dist_vec / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _split_info(children: list[np.ndarray]) -> float:
    tot = sum(d.sum() for d in children)
    if tot <= 0:
        return 0.0
    w = [d.sum() / tot for d in children if d.sum() > 0]
    return float(-sum(wi * np.log2(wi) for wi in w))

def _gain_ratio(parent: np.ndarray, children: list[np.ndarray]) -> float:
    g = _entropy(parent) - sum(d.sum()/max(parent.sum(), 1e-12) * _entropy(d) for d in children)
    s = _split_info(children)
    return float(g / s) if s > 0 else 0.0

def _z_from_cf(cf: float) -> float:
    """
    Approximate a one‑sided z‑score from a confidence factor.

    The post‑pruning procedure uses a ``confidence factor`` to inflate error
    rates on leaves when deciding whether to prune.  A smaller value implies
    less pruning.  For a handful of commonly used confidence factors the
    z‑score is tabulated explicitly; for others a simple linear approximation
    is used.

    Parameters
    ----------
    cf : float
        Confidence factor in (0, 1).  Values near zero correspond to roughly
        two‑sigma pessimistic errors (z≈1.96); values near one imply very
        heavy pruning.

    Returns
    -------
    float
        An approximate z‑score corresponding to the given confidence factor.
    """
    table = {0.25: 1.150, 0.20: 1.282, 0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    if cf in table:
        return table[cf]
    # simple linear approximation between known points
    return max(0.5, 1.150 + (0.25 - cf) * 3.2)

# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------
class TreeNode:
    """Internal representation of a single node in a decision tree.

    Parameters
    ----------
    is_leaf : bool, default=False
        Whether the node represents a terminal leaf.  Leaf nodes carry a
        predicted class and class distribution; internal nodes carry splitting
        information.
    numeric_threshold_strategy : str, default="quantile"
        Retained for backward compatibility; not used directly by the node.
    max_numeric_thresholds : int, default=64
        Retained for backward compatibility; not used directly by the node.

    Attributes
    ----------
    is_leaf : bool
        True if this node is terminal.
    feature_index : int or None
        Index of the feature used for the split at this node; ``None`` for
        leaves.
    threshold : float or set or None
        Numeric threshold for numeric splits or a set of categories for
        categorical splits; ``None`` for leaves.
    children : dict
        Mapping ``{"left": TreeNode, "right": TreeNode}`` for internal nodes.
    split_type : {"numeric", "categorical"} or None
        Indicates the type of split performed at this node.
    predicted_class : int or None
        Majority class label stored at a leaf.
    class_distribution : dict or None
        Dictionary mapping class labels to counts within this node.
    """

    def __init__(self, *, is_leaf: bool = False,
                 numeric_threshold_strategy: str = "quantile",
                 max_numeric_thresholds: int = 64):
        self.numeric_threshold_strategy = str(numeric_threshold_strategy)
        self.max_numeric_thresholds = int(max_numeric_thresholds)
        self.is_leaf: bool = is_leaf
        self.feature_index: int | None = None
        # float for numeric splits; set for categorical splits
        self.threshold: float | set | None = None
        # {"left": TreeNode, "right": TreeNode} for internal nodes
        self.children: dict = {}
        # "numeric" or "categorical"
        self.split_type: str | None = None
        self.predicted_class: int | None = None
        # dict mapping class -> count
        self.class_distribution: dict | None = None
        # (p_left, p_right) for missing value distribution
        self.branch_weights: tuple[float, float] | None = None

# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------
class C5Classifier(BaseEstimator, ClassifierMixin):
    """
    Decision tree classifier inspired by Quinlan's C5.0.

    This estimator builds a single decision tree when ``trials=1`` or an
    ensemble of trees via AdaBoost.M1 when ``trials>1``.  Splits are chosen
    using the gain ratio criterion and support both numeric and categorical
    features as well as missing values.  The training procedure supports
    optional pre‑pruning (``min_samples_split``/``min_samples_leaf``),
    pessimistic post‑pruning controlled by a confidence factor (``cf``), and
    a global pruning pass.  Booster ensembles are formed by re‑sampling the
    training set with probability weights and combining predictions with
    log‑odds weights.

    Parameters
    ----------
    trials : int, default=1
        Number of trees in the ensemble.  A value of 1 fits a single tree.
        Values greater than 1 enable AdaBoost.M1 boosting.  Rule tracing,
        pretty printing, rule export and Graphviz export are only available
        when ``trials=1``.
    min_samples_split : int, default=2
        Minimum number of training samples required to allow a split.  Using
        very small values can lead to extremely deep trees; consider setting
        ``max_depth`` or increasing this value for large datasets.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child after a split.
    pruning : bool, default=True
        Whether to perform pessimistic post‑pruning.  If ``False``,
        ``cf`` and ``global_pruning`` are ignored.
    cf : float, default=0.25
        Confidence factor for pessimistic pruning.  Smaller values produce
        larger trees; larger values prune more aggressively.
    global_pruning : bool, default=True
        Apply a secondary global merge step after local pruning.
    random_state : int or None, default=None
        Random seed used for boosting.  Ignored when ``trials=1``.
    feature_names : list[str] or None, default=None
        Optional list of feature names used for rule/graph exports.  If
        ``categorical_features`` are specified by name then ``feature_names``
        must also be provided.
    categorical_features : list[int|str] or None, default=None
        Indices or names of categorical input features.  If names are used
        ``feature_names`` must be provided.  All other features are treated as
        numeric.
    infer_categorical : bool, default=False
        If ``True``, attempt to infer categorical features based on dtype.
        When dtype information is correct it is recommended to leave this
        disabled and specify categorical columns explicitly.
    int_as_categorical : bool, default=False
        Whether to treat integer columns as categorical when cardinality is
        manageable.  Only used when ``infer_categorical=True``.
    max_categories_exhaustive : int, default=12
        Maximum cardinality for exhaustive subset search on categorical
        features.  Above this value a simpler one‑vs‑rest strategy is used.
    numeric_threshold_strategy : str, default="quantile"
        Strategy used to subsample candidate thresholds for numeric features.
        Currently only ``"quantile"`` is supported.
    max_numeric_thresholds : int, default=32
        When ``numeric_threshold_strategy="quantile"`` the number of candidate
        thresholds per feature is capped at this value.
    max_depth : int or None, default=None
        Maximum depth of the tree.  If ``None`` the depth is unbounded.
        Limiting the depth can help prevent excessive recursion on noisy data.
    verbose : int, default=0
        Verbosity level.  Currently unused.

    Notes
    -----
    - The API follows the scikit‑learn estimator conventions for ``fit``,
      ``predict`` and ``predict_proba``.
    - Rule tracing and export utilities (`predict_rule`, `export_rules`,
      `export_graphviz`, `print_tree`) are only available for single trees
      (``trials=1``).  Calling them on an ensemble raises a ``ValueError``.
    """
    def _maybe_feature_names(self, feature_names=None):
        return feature_names


    def __init__(
        self,
        *,
        trials: int = 1,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        pruning: bool = True,
        cf: float = 0.25,
        global_pruning: bool = True,
        random_state: int | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int | str] | None = None,
        infer_categorical: bool = False,
        int_as_categorical: bool = False,
        max_categories_exhaustive: int = 12,
        numeric_threshold_strategy: str = "quantile",
        max_numeric_thresholds: int = 32,
        max_depth: int | None = None,
        verbose: int = 0,
    ):
        """Clean C5.0-style classifier (Quinlan-inspired)."""
        self.trials = int(trials)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.pruning = bool(pruning)
        self.cf = float(cf)
        self.global_pruning = bool(global_pruning)
        self.random_state = random_state

        self.feature_names_ = feature_names
        self.categorical_features = categorical_features
        self.infer_categorical = bool(infer_categorical)
        self.int_as_categorical = bool(int_as_categorical)
        self.max_categories_exhaustive = int(max_categories_exhaustive)

        self.numeric_threshold_strategy = str(numeric_threshold_strategy)
        self.max_numeric_thresholds = int(max_numeric_thresholds)
        self.max_depth = max_depth
        self.verbose = int(verbose)

        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X, y, sample_weight=None, feature_names=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is None:
            w = np.ones(len(y), dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if len(w) != len(y):
                raise ValueError("sample_weight must have the same length as y")

        # Map categorical feature names to indices
        n_features = X.shape[1]
        if feature_names is not None:
             if len(feature_names) != n_features:
                 raise ValueError("feature_names length must match X.shape[1]")
             self.feature_names_ = list(feature_names)
        elif self.feature_names_ is None:
            self.feature_names_ = [f"f{i}" for i in range(n_features)]
        cf = self.categorical_features
        if cf is not None:
            if len(cf) and isinstance(cf[0], str):
                name_to_idx = {n:i for i,n in enumerate(self.feature_names_)}
                self.categorical_features_ = [name_to_idx[n] for n in cf]
            else:
                self.categorical_features_ = list(map(int, cf))
        else:
            self.categorical_features_ = []
        # --- determine categorical mask strictly from categorical_features ---
        n_features = X.shape[1]
        self.n_features_ = n_features
        cats = set()
        if getattr(self, 'categorical_features', None) is not None:
            cf = list(self.categorical_features)
            if len(cf) and isinstance(cf[0], str):
                if getattr(self, 'feature_names_', None) is None:
                    raise ValueError('feature_names must be provided when using categorical_features by name')
                name_to_idx = {n:i for i,n in enumerate(self.feature_names_)}
                cf = [name_to_idx[c] for c in cf]
            cats = set(int(i) for i in cf)
        self.is_cat_ = [ (i in cats) for i in range(n_features) ]
        
        self.classes_ = np.unique(y)
        if self.trials == 1:
            # build a single tree and record its depth
            self.tree_ = self._build_tree(X, y, w, depth=0)
            if self.pruning:
                self._prune_local(self.tree_)
                if self.global_pruning:
                    self._prune_global(self.tree_)
        else:
            self._fit_boosting(X, y, w)
        return self

    def predict(self, X):
        """
        Predict class labels for the provided samples.

        For single trees (``trials=1``) this returns the class associated
        with the leaf reached by each instance.  For boosted ensembles the
        classes are determined by aggregating the individual tree votes via
        :meth:`predict_proba` and selecting the maximum probability class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.  Missing values may be represented by ``None`` or
            ``numpy.nan``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
        # Check that the model is fitted.  For a single tree the ``tree_``
        # attribute is created in fit; for an ensemble ``ensemble_`` and
        # ``alphas_`` are created.
        if self.trials == 1:
            if getattr(self, 'tree_', None) is None:
                raise ValueError("Estimator not fitted. Call fit(...) first.")
        else:
            if not getattr(self, 'ensemble_', None):
                raise ValueError("Estimator not fitted. Call fit(...) first.")
        X = np.asarray(X)
        if self.trials == 1:
            return np.array([self._predict_instance(x, self.tree_) for x in X])
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for the provided samples.

        For single trees (``trials=1``) this returns the posterior class
        distribution associated with each leaf.  For boosted ensembles the
        probabilities are computed by aggregating the weighted vote of each
        tree via the boosting weights ``alphas_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.  Missing values may be represented by ``None`` or
            ``numpy.nan``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        ValueError
            If the estimator has not been fitted.
        """
        X = np.asarray(X)
        # Single-tree case: ensure the tree exists
        if self.trials == 1:
            if getattr(self, 'tree_', None) is None:
                raise ValueError("Estimator not fitted. Call fit(...) first.")
            return np.array([self._predict_proba_instance(x, self.tree_) for x in X])
        # Ensemble case: ensure the ensemble has been trained
        if not getattr(self, 'ensemble_', None):
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        n, k = len(X), len(self.classes_)
        acc = np.zeros((n, k), dtype=float)
        for tree, alpha in zip(self.ensemble_, self.alphas_):
            acc += alpha * np.array([self._predict_proba_instance(x, tree) for x in X])
        # Normalise to probability simplex
        row_sum = acc.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        acc /= row_sum
        return acc

    def predict_rule(self, X, feature_names=None):
        """
        Return the decision rule (antecedent) followed by each input instance.

        Each returned string describes the conjunction of conditions leading
        from the root to the leaf used to predict the class of the instance.
        Only available for single trees (``trials=1``); calling this on an
        ensemble will raise a ``ValueError``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.  Missing values may be represented by ``None`` or
            ``numpy.nan``.
        feature_names : list[str], optional
            Alternative names for the features.  If omitted, the names passed
            to the estimator at construction time are used.

        Returns
        -------
        list[str]
            A list of antecedent strings, one per input sample.
        """
        # Only single trees support rule tracing; boosted ensembles lack a
        # single unrolled structure.
        if self.trials != 1:
            raise ValueError("predict_rule only available when trials=1")
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        Xp = np.asarray(X, dtype=object)
        fn = self._maybe_feature_names(feature_names)
        return [self._trace_rule(x, self.tree_, fn) for x in Xp]

    def export_rules(self, *, feature_names=None, class_names=None):
        """
        Export all decision rules in the tree as a list of human‑readable strings.

        Only available for single trees (``trials=1``).  Each rule has the form
        ``<antecedent> => <predicted class>`` where the antecedent is a
        conjunction of conditions from root to leaf.  Feature and class names
        can optionally be supplied.

        Parameters
        ----------
        feature_names : list[str], optional
            Names for the input features.  Defaults to those provided at
            construction time.
        class_names : list[str], optional
            Names for the classes, ordered according to ``self.classes_``.

        Returns
        -------
        list[str]
            List of rule strings.
        """
        # Exporting rules is only supported for single trees
        if self.trials != 1:
            raise ValueError("export_rules available only when trials=1")
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        rules: list[str] = []
        self._collect_rules(self.tree_, [], rules, feature_names, class_names)
        return rules

    def export_graphviz(self, filename: str | None = None, *, feature_names=None,
                        class_names=None, format: str = "png") -> str:
        """
        Export the tree structure in Graphviz format.

        For single trees (``trials=1``) this method produces a representation of
        the learned decision structure using the `graphviz` Python package.
        The output can be generated in various formats supported by Graphviz,
        including images (``'png'``, ``'pdf'``, etc.) and plain DOT files.  When
        requesting a DOT file (``format='dot'``) no external Graphviz binary is
        required; the DOT source is written directly to disk.  For other
        formats this method attempts to invoke the system ``dot`` command; if
        it is unavailable the method falls back to writing a ``.dot`` file
        instead.

        Parameters
        ----------
        filename : str or None, default=None
            Basename of the output file (the extension is determined by
            ``format``). If None, the DOT source code is returned as a string
            and no file is written.
        feature_names : list[str], optional
            Custom names for the input features.  Defaults to the names
            provided at construction time.
        class_names : list[str], optional
            Custom names for the classes, ordered according to
            :attr:`classes_`.
        format : str, default="png"
            Desired output format for Graphviz.  Supported values include
            ``'png'``, ``'pdf'``, ``'svg'`` and ``'dot'``.  The special value
            ``'dot'`` writes the DOT source directly and does not call the
            external ``dot`` command.

        Returns
        -------
        str
            Path to the written file, or the DOT source code if filename is None.

        Raises
        ------
        ValueError
            If the estimator is not fitted or if ``trials != 1``.
        """
        # Graphviz export is only supported for single trees
        if self.trials != 1:
            raise ValueError("export_graphviz only available when trials=1")
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        # Build the Graphviz object
        try:
            import graphviz
        except ImportError:
            raise RuntimeError("Graphviz is required for export_graphviz but not installed.")
        dot = graphviz.Digraph(format=format)
        self._add_graph_nodes(dot, self.tree_, "0", feature_names, class_names)
        
        if filename is None:
            return dot.source
            
        # If the user requests a dot file we avoid invoking the external
        # Graphviz binary entirely: save the dot source and return.
        if format.lower() == "dot":
            path = f"{filename}.dot"
            dot.save(path)
            return path
        # Otherwise attempt to render using the installed dot executable.
        try:
            dot.render(filename, cleanup=True)
            return f"{filename}.{format}"
        except Exception:
            # On failure (e.g. missing graphviz binary) fall back to a dot file
            fallback_path = f"{filename}.dot"
            dot.save(fallback_path)
            return fallback_path

    def print_tree(self, feature_names=None, class_names=None):
        """
        Pretty‑print the decision tree to ``stdout``.

        Only available for single trees (``trials=1``).  For ensembles a
        ``ValueError`` is raised.  If ``feature_names`` and ``class_names`` are
        provided they will be used in place of raw indices and integer class
        labels.

        Parameters
        ----------
        feature_names : list[str], optional
            Alternative names for the features.
        class_names : list[str], optional
            Alternative names for the classes, ordered like ``self.classes_``.
        """
        # Pretty printing is only supported for single trees
        if self.trials != 1:
            raise ValueError("print_tree only available when trials=1")
        if getattr(self, 'tree_', None) is None:
            raise ValueError("Estimator not fitted. Call fit(...) first.")
        fn = self._maybe_feature_names(feature_names)
        cn = class_names
        self._print_node(self.tree_, "", fn, cn)


    def _fit_boosting(self, X, y, w):
        # C5.0 boosting: reweighting, not resampling?
        # Actually C5.0 uses a mix, but standard Adaboost uses reweighting.
        # The previous implementation used resampling.
        # Let's switch to reweighting now that we support weights.
        
        n = len(y)
        # Normalize weights to sum to n? Or keep them as is?
        # Adaboost usually normalizes to sum to 1, but we can scale.
        # Let's keep the input weights as the starting point.
        curr_w = w.copy()
        
        self.ensemble_, self.alphas_ = [], []
        eps = 1e-10
        
        # Number of classes
        K = len(self.classes_)

        for _ in range(self.trials):
            # Fit tree with current weights
            tree = self._build_tree(X, y, curr_w, depth=0)
            if self.pruning:
                self._prune_local(tree)
                if self.global_pruning:
                    self._prune_global(tree)
            
            # Predict class (hard prediction)
            pred = np.array([self._predict_instance(x, tree) for x in X])
            
            # Calculate error
            incorrect = (pred != y)
            err = np.sum(curr_w * incorrect) / np.sum(curr_w)
            
            if err <= eps:
                # Perfect classifier, stop
                self.ensemble_.append(tree)
                self.alphas_.append(10.0) # Arbitrary large alpha? Or just 1?
                break
            
            if err >= 0.5:
                # Worse than random (for binary), stop or reset?
                # C5.0 might handle this differently, but for now let's stop
                if len(self.ensemble_) == 0:
                     self.ensemble_.append(tree)
                     self.alphas_.append(1.0)
                break

            # Calculate alpha (SAMME or Adaboost.M1)
            # Adaboost.M1: alpha = 0.5 * log((1-err)/err)
            # SAMME: alpha = log((1-err)/err) + log(K-1)
            if K > 2:
                 alpha = np.log((1 - err) / err) + np.log(K - 1)
            else:
                 alpha = 0.5 * np.log((1 - err) / err)
            
            # Update weights
            # w <- w * exp(alpha * (pred != y))
            # But for SAMME/M1 we usually increase weight of incorrect.
            curr_w *= np.exp(alpha * incorrect)
            
            # Normalize
            curr_w /= curr_w.sum()
            curr_w *= n # Scale back to sum to n, to keep n_eff reasonable
            
            self.ensemble_.append(tree)
            self.alphas_.append(alpha)

        if len(self.ensemble_) == 0:
            # Fallback
            self.tree_ = self._build_tree(X, y, w, depth=0)
            if self.pruning:
                self._prune_local(self.tree_)
                if self.global_pruning:
                    self._prune_global(self.tree_)
            self.ensemble_ = [self.tree_]
            self.alphas_ = [1.0]

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------
    def _predict_instance(self, x, node: TreeNode):
        if node.is_leaf:
            return node.predicted_class
            
        val = x[node.feature_index]
        if _isnan_scalar(val):
            # Missing value: recurse both ways and aggregate
            # For hard classification, this is tricky. C5.0 usually sums probabilities.
            # So we should probably use _predict_proba_instance logic and take argmax.
            probs = self._predict_proba_instance(x, node)
            return self.classes_[np.argmax(probs)]
            
        if node.split_type == "numeric":
            if float(val) <= node.threshold:
                return self._predict_instance(x, node.children["left"])
            else:
                return self._predict_instance(x, node.children["right"])
        else:
            if val in node.threshold:
                return self._predict_instance(x, node.children["left"])
            else:
                return self._predict_instance(x, node.children["right"])

    def _predict_proba_instance(self, x, node: TreeNode):
        if node.is_leaf:
            tot = sum(node.class_distribution.values())
            if tot <= 0:
                return np.full(len(self.classes_), 1.0 / len(self.classes_))
            return np.array([node.class_distribution.get(c, 0) / tot for c in self.classes_])

        val = x[node.feature_index]
        if _isnan_scalar(val):
            # Weighted average of children
            if node.branch_weights is None:
                # Should not happen if trained correctly, but fallback
                pl, pr = 0.5, 0.5
            else:
                pl, pr = node.branch_weights
            
            left_probs = self._predict_proba_instance(x, node.children["left"])
            right_probs = self._predict_proba_instance(x, node.children["right"])
            return pl * left_probs + pr * right_probs

        if node.split_type == "numeric":
            if float(val) <= node.threshold:
                return self._predict_proba_instance(x, node.children["left"])
            else:
                return self._predict_proba_instance(x, node.children["right"])
        else:
            if val in node.threshold:
                return self._predict_proba_instance(x, node.children["left"])
            else:
                return self._predict_proba_instance(x, node.children["right"])

    # ------------------------------------------------------------------
    # Tree construction (gain ratio)
    # ------------------------------------------------------------------
    def _build_tree(self, X, y, w, depth: int = 0) -> TreeNode:
        """
        Recursively build a decision tree from ``X`` and ``y``.

        A new :class:`TreeNode` is created for each split.  If the stopping
        conditions are met (insufficient samples, purity, exhausted depth or
        no beneficial split) a leaf is returned instead.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data for the current node.
        y : ndarray of shape (n_samples,)
            Target labels for the current node.
        w : ndarray of shape (n_samples,)
            Sample weights.
        depth : int, default=0
            Current depth of the node.  Used to enforce ``max_depth``.

        Returns
        -------
        TreeNode
            A fully populated subtree or a leaf node.
        """
        # Stop if not enough samples or the node is pure
        # Use effective count (sum of weights) for min_samples_split check? 
        # C5.0 uses cases count (sum of weights)
        n_eff = w.sum()
        if n_eff < self.min_samples_split or len(np.unique(y)) == 1:
            return self._create_leaf(y, w)
        # Honour max_depth parameter
        if self.max_depth is not None and depth >= int(self.max_depth):
            return self._create_leaf(y, w)
        # Find the best split
        feat, thr, stype, pl, pr = self._best_split(X, y, w)
        if feat is None:
            return self._create_leaf(y, w)
        node = TreeNode(is_leaf=False)
        node.feature_index = feat
        node.threshold = thr
        node.split_type = stype
        node.class_distribution = self._class_distribution(y, w)
        node.predicted_class = max(node.class_distribution, key=node.class_distribution.get)
        node.branch_weights = (pl, pr)
        
        vals = X[:, feat]
        
        # Identify missing values
        if vals.dtype.kind == "f":
            known = ~np.isnan(vals)
            miss = np.isnan(vals)
        else:
            known = np.array([not _isnan_scalar(v) for v in vals], bool)
            miss = ~known
            
        # Handle numeric and categorical splits separately
        if stype == "numeric":
            left_mask = known & (vals.astype(float) <= thr)
            right_mask = known & (vals.astype(float) > thr)
        else:
            in_group = np.isin(vals, list(thr))
            left_mask = known & in_group
            right_mask = known & (~in_group)
            
        # Distribute missing values
        # We need to pass down weights.
        # w_left = w[left_mask] + pl * w[miss]
        # But we need to construct the child datasets.
        # The child dataset should contain the left_mask instances AND the miss instances.
        # But the miss instances need to have their weights adjusted.
        
        # Construct left child data
        # Indices for left child: left_mask OR miss
        left_indices = np.where(left_mask | miss)[0]
        right_indices = np.where(right_mask | miss)[0]
        
        if len(left_indices) == 0:
            node.children["left"] = self._create_leaf(y, w) # Should be empty leaf? Or parent's class?
        else:
            X_left = X[left_indices]
            y_left = y[left_indices]
            w_left = w[left_indices].copy()
            
            # Adjust weights for missing values in left child
            # We need to know which ones were missing in the original X, relative to left_indices
            # miss[left_indices] is not correct because left_indices is a list of indices into X
            # We want to multiply w of missing instances by pl
            
            # Boolean mask of missing values within the subset
            # subset_miss_mask = miss[left_indices]
            # w_left[subset_miss_mask] *= pl
            
            # Let's do it cleanly:
            # Iterate over left_indices, if it was missing in parent, scale weight
            # Vectorized:
            miss_in_left = miss[left_indices]
            w_left[miss_in_left] *= pl
            
            node.children["left"] = self._build_tree(X_left, y_left, w_left, depth + 1)

        if len(right_indices) == 0:
            node.children["right"] = self._create_leaf(y, w)
        else:
            X_right = X[right_indices]
            y_right = y[right_indices]
            w_right = w[right_indices].copy()
            
            miss_in_right = miss[right_indices]
            w_right[miss_in_right] *= pr
            
            node.children["right"] = self._build_tree(X_right, y_right, w_right, depth + 1)
            
        return node

    def _create_leaf(self, y, w) -> TreeNode:
        leaf = TreeNode(is_leaf=True)
        leaf.class_distribution = self._class_distribution(y, w)
        leaf.predicted_class = max(leaf.class_distribution, key=leaf.class_distribution.get)
        return leaf

    def _class_distribution(self, y, w) -> dict:
        # Weighted counts
        dist = {}
        for cls in self.classes_:
            mask = (y == cls)
            dist[cls] = float(w[mask].sum())
        return dist

    def _class_distribution_vector(self, y, w) -> np.ndarray:
        vec = np.zeros(len(self.classes_), dtype=float)
        for i, cls in enumerate(self.classes_):
            mask = (y == cls)
            vec[i] = w[mask].sum()
        return vec


    def _best_split(self, X, y, w):
        """Compute best split by Gain Ratio (fast) with weights."""
        import numpy as np
        classes = self.classes_
        parent = self._class_distribution_vector(y, w)
        best_gain, best_feat, best_thr, best_type = -1.0, None, None, None
        best_pl, best_pr = 0.5, 0.5
        
        n_features = X.shape[1]
        cats = set(getattr(self, "categorical_features_", getattr(self, "categorical_features", []) or []))
        
        # Total weight
        total_w = w.sum()
        if total_w <= 0:
            return None, None, None, None, None
            
        for j in range(n_features):
            col = X[:, j]
            is_cat = j in cats
            if col.dtype.kind == "f":
                known = ~np.isnan(col); v_known = col[known]; y_known = y[known]; w_known = w[known]
            else:
                known = col != None; v_known = col[known]; y_known = y[known]; w_known = w[known]
            
            if v_known.size == 0:
                continue
            
            # Fraction of known values
            w_known_sum = w_known.sum()
            frac_known = w_known_sum / total_w
            
            if is_cat:
                # Weighted categorical split
                vals, inverse = np.unique(v_known, return_inverse=True)
                val_dists = {}
                for idx, v in enumerate(vals):
                    mask = (inverse == idx)
                    val_dists[v] = self._class_distribution_vector(y_known[mask], w_known[mask])
                
                val_weights = {v: d.sum() for v, d in val_dists.items()}
                sorted_vals = sorted(val_dists.keys(), key=lambda x: val_weights[x], reverse=True)
                candidates = sorted_vals[:int(getattr(self, "max_categories_exhaustive", 12))]
                
                # Exhaustive subset search
                import itertools
                n_cats = len(candidates)
                
                # If too many categories, fallback to one-vs-rest or heuristic?
                # For now, we respect max_categories_exhaustive.
                # We iterate combinations of size 1 to n_cats // 2
                
                parent_known = self._class_distribution_vector(y_known, w_known)
                
                # Optimization: pre-calculate dists for candidates
                cand_dists = [val_dists[v] for v in candidates]
                
                for r in range(1, (n_cats // 2) + 1):
                    for subset_indices in itertools.combinations(range(n_cats), r):
                        # Construct left node distribution
                        left = np.zeros_like(parent_known)
                        subset_vals = []
                        for idx in subset_indices:
                            left += cand_dists[idx]
                            subset_vals.append(candidates[idx])
                        
                        right = parent_known - left
                        
                        # Check min_samples_leaf
                        if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                            continue

                        # Gain ratio on known values
                        gr = _gain_ratio(parent_known, [left, right])
                        
                        # Penalize by fraction of known values (C5.0 logic)
                        gr *= frac_known
                        
                        if gr > best_gain:
                            swL = left.sum()
                            swR = right.sum()
                            pl = swL / (swL + swR) if (swL + swR) > 0 else 0.5
                            best_gain, best_feat, best_thr, best_type = float(gr), j, frozenset(subset_vals), "categorical"
                            best_pl, best_pr = pl, 1.0 - pl
            else:
                # Weighted numeric split
                vv = v_known.astype(float, copy=False)
                if vv.size <= 1:
                    continue
                order = np.argsort(vv, kind="mergesort")
                v = vv[order]; yk = y_known[order]; wk = w_known[order]
                
                bd = np.nonzero(v[:-1] != v[1:])[0]
                if bd.size == 0:
                    continue
                if getattr(self, "numeric_threshold_strategy", "quantile") == "quantile":
                    k = int(getattr(self, "max_numeric_thresholds", 32))
                    if bd.size > k:
                        idxs = np.linspace(0, bd.size-1, num=k, dtype=int); bd = bd[idxs]
                
                idx_map = {c:i for i,c in enumerate(classes)}; K = len(classes)
                y_idx = np.fromiter((idx_map[c] for c in yk), count=yk.shape[0], dtype=int)
                
                M = np.zeros((y_idx.shape[0], K), dtype=float)
                M[np.arange(y_idx.shape[0]), y_idx] = wk
                
                SW = M.cumsum(axis=0); total = SW[-1] # This is parent_known
                
                # Parent entropy for gain ratio
                parent_known = total
                
                for i in bd:
                    left = SW[i]; right = total - left
                    gr = _gain_ratio(parent_known, [left, right])
                    gr *= frac_known
                    
                    if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                        continue
                    
                    if gr > best_gain:
                        thr = 0.5 * (v[i] + v[i+1])
                        swL = left.sum()
                        swR = right.sum()
                        pl = swL / (swL + swR) if (swL + swR) > 0 else 0.5
                        best_gain, best_feat, best_thr, best_type = float(gr), j, float(thr), "numeric"
                        best_pl, best_pr = pl, 1.0 - pl
                        
        return best_feat, best_thr, best_type, best_pl, best_pr

    def _node_error_rate(self, node: TreeNode) -> tuple[float, float]:
        dist = node.class_distribution
        N = float(sum(dist.values()))
        if N <= 0:
            return 1.0, 0.0
        err_leaf = 1.0 - (max(dist.values()) / N)
        return err_leaf, N

    def _pessimistic(self, err_rate: float, N: float) -> float:
        if N <= 0:
            return 1.0
        z = _z_from_cf(self.cf)
        se = np.sqrt(err_rate * max(1.0 - err_rate, 0.0) / max(N, 1.0))
        return min(1.0, err_rate + z * se)

    def _prune_local(self, node: TreeNode) -> tuple[float, float]:
        if node.is_leaf:
            err_leaf, N = self._node_error_rate(node)
            return self._pessimistic(err_leaf, N), N

        total_N = 0.0
        pess_sum = 0.0
        for ch in node.children.values():
            pess, n_ch = self._prune_local(ch)
            pess_sum += pess * n_ch
            total_N += n_ch
        pess_subtree = pess_sum / max(total_N, 1.0)

        err_leaf, N_here = self._node_error_rate(node)
        pess_leaf = self._pessimistic(err_leaf, N_here)

        if pess_leaf <= pess_subtree:
            node.is_leaf = True
            node.children = {}
            return pess_leaf, N_here
        return pess_subtree, total_N

    def _prune_global(self, node: TreeNode):
        if node.is_leaf:
            return
        # Estimate pessimistic error of the subtree (average of children)
        child_pess_sum, child_N = 0.0, 0.0
        for ch in node.children.values():
            err_leaf_ch, N_ch = self._node_error_rate(ch)
            pess_ch = self._pessimistic(err_leaf_ch, N_ch)
            child_pess_sum += pess_ch * max(N_ch, 1.0)
            child_N += max(N_ch, 1.0)
        pess_sub = child_pess_sum / max(child_N, 1.0)

        err_leaf, N_here = self._node_error_rate(node)
        pess_leaf = self._pessimistic(err_leaf, N_here)

        if pess_leaf <= pess_sub:
            node.is_leaf = True
            node.children = {}
            return

        for ch in list(node.children.values()):
            self._prune_global(ch)

    # ------------------------------------------------------------------
    # Rule tracing / Graphviz / printing helpers
    # ------------------------------------------------------------------
    def _trace_rule(self, x, node: TreeNode, fn=None, parts=None):
        parts = parts or []
        if node.is_leaf:
            return " AND ".join(parts) if parts else "<root>"
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            if _isnan_scalar(x[node.feature_index]):
                parts.append(f"{name} MISSING")
                return " AND ".join(parts)
            if float(x[node.feature_index]) <= node.threshold:
                parts.append(f"{name} <= {node.threshold:.4f}")
                return self._trace_rule(x, node.children["left"], fn, parts)
            else:
                parts.append(f"{name} > {node.threshold:.4f}")
                return self._trace_rule(x, node.children["right"], fn, parts)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            in_left = (not _isnan_scalar(x[node.feature_index])) and (x[node.feature_index] in node.threshold)
            parts.append(f"{name} " + ("IN " if in_left else "NOT IN ") + S)
            return self._trace_rule(x, node.children["left" if in_left else "right"], fn, parts)
    def _collect_rules(self, node: TreeNode, parts, rules, fn, cn):
        if node.is_leaf:
            body = " AND ".join(parts) if parts else "<root>"
            pred = cn[node.predicted_class] if cn is not None else str(node.predicted_class)
            rules.append(f"{body} => {pred}")
            return
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            left = f"{name} <= {node.threshold:.4f}"
            right = f"{name} > {node.threshold:.4f}"
            self._collect_rules(node.children["left"],  parts + [left],  rules, fn, cn)
            self._collect_rules(node.children["right"], parts + [right], rules, fn, cn)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            left = f"{name} IN {S}"
            right = f"{name} NOT IN {S}"
            self._collect_rules(node.children["left"],  parts + [left],  rules, fn, cn)
            self._collect_rules(node.children["right"], parts + [right], rules, fn, cn)

    def _add_graph_nodes(self, dot, node: TreeNode, name: str, fn, cn):
        if node.is_leaf:
            pred = cn[node.predicted_class] if cn is not None else str(node.predicted_class)
            dot.node(name, f"class={pred}\n{dict(node.class_distribution)}",
                    shape="box", style="filled", color="lightgrey")
            return
        fname = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            label = f"{fname} <= {node.threshold:.4f}"
        else:
            label = f"{fname} ∈ " + "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
        dot.node(name, label, shape="ellipse", style="filled", color="lightblue")
        l_id, r_id = name + "L", name + "R"
        self._add_graph_nodes(dot, node.children["left"],  l_id, fn, cn)
        self._add_graph_nodes(dot, node.children["right"], r_id, fn, cn)
        dot.edge(name, l_id, label="True")
        dot.edge(name, r_id, label="False")

    def _print_node(self, node: TreeNode, indent="", fn=None, cn=None):
        if node.is_leaf:
            pred = cn[node.predicted_class] if cn is not None else str(node.predicted_class)
            print(f"{indent}Predict {pred} | dist={dict(node.class_distribution)}")
            return
        name = (fn[node.feature_index] if (fn is not None and 0 <= node.feature_index < len(fn))
                else f"X[{node.feature_index}]")
        if node.split_type == "numeric":
            print(f"{indent}if {name} <= {node.threshold:.4f}:")
            self._print_node(node.children["left"], indent + "  ", fn, cn)
            print(f"{indent}else:")
            self._print_node(node.children["right"], indent + "  ", fn, cn)
        else:
            S = "{" + ", ".join(map(str, sorted(node.threshold))) + "}"
            print(f"{indent}if {name} in {S}:")
            self._print_node(node.children["left"], indent + "  ", fn, cn)
            print(f"{indent}else:")
            self._print_node(node.children["right"], indent + "  ", fn, cn)
