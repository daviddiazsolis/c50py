"""
Benchmark: c50py vs scikit-learn CART (one-hot encoded) vs the original C5.0 (R package C50).

Single decision trees on OpenML classification datasets with categorical
features. For every dataset the same 5 stratified folds are used for all
models. Reported per fold: test accuracy, number of leaves, mean rule length
(conditions per leaf), mean number of distinct columns per rule, number of
original columns used, fit time.

Models
------
c50py         C5Classifier() with C5.0's defaults (cf=0.25, minCases=2)
c50py_cv      c50py with cf tuned by 3-fold inner CV over {0.05, 0.1, 0.25, 0.5}
c50py_winnow  C5Classifier(winnow=True)
c50py_rules   C5RulesClassifier(): C5.0-style ruleset
c50py_boost   C5Classifier(trials=10)
cart          DecisionTreeClassifier() on one-hot encoded data (scikit-learn defaults)
cart_cv       CART with ccp_alpha and min_samples_leaf tuned by 5-fold inner CV
hgb           HistGradientBoostingClassifier with native categorical support (black-box reference)
C5.0_R        R package C50, C5.0() with its defaults (needs Rscript and C50 installed);
C5.0_R_rules  C5.0(rules = TRUE); C5.0_R_boost: C5.0(trials = 10)

Usage
-----
    python benchmarks/run_benchmark.py            # all datasets, results/benchmark_folds.csv
    python benchmarks/run_benchmark.py 31 29      # only these OpenML ids
"""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import c50py
from c50py import C5Classifier, C5RulesClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

# OpenML classification datasets with categorical columns (n <= ~50k)
DATASETS = [2, 3, 10, 13, 24, 25, 26, 29, 31, 38, 42, 46, 50, 55, 56, 188,
            1461, 1590, 6332, 23381, 40701, 40975, 40981]
MAX_N = 10_000          # larger datasets are subsampled (stratified) to keep pure-Python fits short
SEED = 0
N_FOLDS = 5
# MODELS=c50py,c50py_cv to rerun only some models; TAG names the output files
MODELS = os.environ.get("MODELS", "c50py,c50py_cv,c50py_winnow,c50py_rules,c50py_boost,cart,cart_cv,hgb,"
                                  "C5.0_R,C5.0_R_rules,C5.0_R_boost").split(",")
TAG = os.environ.get("TAG", "")


# --------------------------------------------------------------------------- data
def load(did: int):
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"{did}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    d = fetch_openml(data_id=did, as_frame=True, parser="pandas")
    out = (d.data, d.target, d.details["name"])
    pickle.dump(out, open(cache, "wb"))
    return out


def prepare(did: int):
    X, y, name = load(did)
    y = y.astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)
    for c in X.columns:                      # categorical -> pandas category, booleans included
        if str(X[c].dtype) in ("object", "str", "string", "bool", "boolean"):
            X[c] = X[c].astype("category")
    X = X.loc[:, X.nunique(dropna=True) > 1]  # constant columns carry no information
    if len(X) > MAX_N:
        idx = (pd.Series(np.arange(len(y))).groupby(y, group_keys=False)
               .apply(lambda s: s.sample(frac=MAX_N / len(y), random_state=SEED)).sort_values().to_numpy())
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y, name


def is_cat(s: pd.Series) -> bool:
    return str(s.dtype) == "category"


# --------------------------------------------------------------------------- tree stats
def c50py_stats(model: C5Classifier):
    """leaves, mean conditions per rule, mean distinct columns per rule, columns used."""
    depths, distinct, feats = [], [], set()

    def walk(node, path):
        if node.is_leaf:
            depths.append(len(path))
            distinct.append(len(set(path)))
            return
        feats.add(node.feature_index)
        for ch in node.children.values():
            walk(ch, path + [node.feature_index])

    walk(model.tree_, [])
    return len(depths), float(np.mean(depths)), float(np.mean(distinct)), len(feats)


def cart_stats(model: DecisionTreeClassifier, col_of_feature: np.ndarray):
    t = model.tree_
    depths, distinct, stack = [], [], [(0, ())]
    while stack:
        node, path = stack.pop()
        if t.children_left[node] == -1:
            depths.append(len(path))
            distinct.append(len(set(path)))
        else:
            col = col_of_feature[t.feature[node]]
            stack += [(t.children_left[node], path + (col,)), (t.children_right[node], path + (col,))]
    used = t.feature[t.feature >= 0]
    return len(depths), float(np.mean(depths)), float(np.mean(distinct)), len(set(col_of_feature[used]))


# --------------------------------------------------------------------------- encoders for CART
class OneHot:
    """One-hot encode the categorical columns (missing -> its own category); keep numeric NaN
    (scikit-learn trees handle missing numeric values natively since 1.3)."""

    def fit(self, X: pd.DataFrame):
        self.cat = [c for c in X.columns if is_cat(X[c])]
        self.num = [c for c in X.columns if c not in self.cat]
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.cat:
            self.enc.fit(self._cats(X))
        cols = list(X.columns)
        col_of = [cols.index(c) for c in self.num]
        if self.cat:
            for c, cats in zip(self.cat, self.enc.categories_):
                col_of += [cols.index(c)] * len(cats)
        self.col_of_feature = np.array(col_of)
        return self

    def _cats(self, X):
        return X[self.cat].astype(object).where(X[self.cat].notna(), "__missing__").astype(str)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = [X[self.num].to_numpy(dtype=float)]
        if self.cat:
            parts.append(self.enc.transform(self._cats(X)))
        return np.hstack(parts)


# --------------------------------------------------------------------------- models
def run_c50py(Xtr, ytr, Xte, yte, variant: str = "default"):
    t0 = time.perf_counter()
    if variant == "cv":
        gs = GridSearchCV(C5Classifier(), {"cf": [0.05, 0.1, 0.25, 0.5]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=SEED), error_score="raise")
        m = gs.fit(Xtr, ytr).best_estimator_
    elif variant == "winnow":
        m = C5Classifier(winnow=True).fit(Xtr, ytr)
    elif variant == "boost":
        m = C5Classifier(trials=10).fit(Xtr, ytr)
    elif variant == "rules":
        m = C5RulesClassifier().fit(Xtr, ytr)
    else:
        m = C5Classifier().fit(Xtr, ytr)
    fit = time.perf_counter() - t0
    acc = m.score(Xte, yte)
    if variant == "boost":
        return dict(acc=acc, leaves=np.nan, rule_len=np.nan, rule_vars=np.nan, features=np.nan,
                    fit_s=fit, n_trees=len(m.ensemble_))
    if variant == "rules":
        conds = [len(r.conditions) for r in m.rules_] or [0]
        dist = [len({c.feature for c in r.conditions}) for r in m.rules_] or [0]
        used = {c.feature for r in m.rules_ for c in r.conditions}
        return dict(acc=acc, leaves=len(m.rules_), rule_len=float(np.mean(conds)),
                    rule_vars=float(np.mean(dist)), features=len(used), fit_s=fit)
    leaves, rlen, rvars, nfeat = c50py_stats(m)
    return dict(acc=acc, leaves=leaves, rule_len=rlen, rule_vars=rvars, features=nfeat, fit_s=fit)


def run_hgb(Xtr, ytr, Xte, yte):
    """scikit-learn's HistGradientBoostingClassifier with native categorical support:
    a strong black-box reference."""
    def prep(X):
        X = X.copy()
        for c in X.columns:
            if is_cat(X[c]):
                X[c] = X[c].cat.codes.replace(-1, np.nan) if hasattr(X[c], "cat") else X[c]
        return X
    cats = [is_cat(Xtr[c]) for c in Xtr.columns]
    # categories coded consistently on train + test
    allx = pd.concat([Xtr, Xte])
    for c in allx.columns:
        if is_cat(allx[c]):
            allx[c] = allx[c].astype(str).where(allx[c].notna()).astype("category")
    A, B = prep(allx.iloc[:len(Xtr)]), prep(allx.iloc[len(Xtr):])
    t0 = time.perf_counter()
    m = HistGradientBoostingClassifier(categorical_features=cats, random_state=SEED).fit(A, ytr)
    fit = time.perf_counter() - t0
    return dict(acc=m.score(B, yte), leaves=np.nan, rule_len=np.nan, rule_vars=np.nan,
                features=np.nan, fit_s=fit)


def run_cart(Xtr, ytr, Xte, yte, tuned: bool):
    oh = OneHot().fit(Xtr)
    A, B = oh.transform(Xtr), oh.transform(Xte)
    t0 = time.perf_counter()
    if tuned:
        path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(A, ytr)
        alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.98, 25)))
        gs = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                          {"ccp_alpha": alphas, "min_samples_leaf": [1, 2, 5, 10]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
        m = gs.fit(A, ytr).best_estimator_
    else:
        m = DecisionTreeClassifier(random_state=SEED).fit(A, ytr)
    fit = time.perf_counter() - t0
    leaves, rlen, rvars, nfeat = cart_stats(m, oh.col_of_feature)
    return dict(acc=m.score(B, yte), leaves=leaves, rule_len=rlen, rule_vars=rvars, features=nfeat, fit_s=fit)


R_SCRIPT = r"""
suppressMessages(library(C50))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], stringsAsFactors = FALSE, check.names = FALSE, na.strings = c("", "NA"))
meta_lines <- readLines(args[2])
meta <- list(features = strsplit(meta_lines[1], ",")[[1]], cat = strsplit(meta_lines[2], ",")[[1]])
meta$cat <- meta$cat[nzchar(meta$cat)]
for (c in meta$cat) d[[c]] <- factor(d[[c]])
d$.y <- factor(d$.y)
out <- data.frame()
for (k in sort(unique(d$.fold))) {
  tr <- d[d$.fold != k, ]; te <- d[d$.fold == k, ]
  X <- tr[, meta$features, drop = FALSE]
  for (mode in c("tree", "rules", "boost")) {
    t0 <- proc.time()[["elapsed"]]
    m <- C5.0(x = X, y = tr$.y, rules = (mode == "rules"), trials = ifelse(mode == "boost", 10, 1))
    fit <- proc.time()[["elapsed"]] - t0
    pred <- predict(m, te[, meta$features, drop = FALSE])
    nfeat <- tryCatch(sum(C5imp(m, metric = "usage")$Overall > 0), error = function(e) NA)
    size <- ifelse(mode == "boost", NA, m$size[1])
    out <- rbind(out, data.frame(fold = k, mode = mode, acc = mean(as.character(pred) == as.character(te$.y)),
                                 leaves = size, features = ifelse(mode == "boost", NA, nfeat), fit_s = fit))
  }
}
write.csv(out, args[3], row.names = FALSE)
"""


def run_r(X, y, folds):
    if shutil.which("Rscript") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        feats = [f"v{i}" for i in range(X.shape[1])]        # safe names for R
        d = X.copy()
        d.columns = feats
        d[".y"], d[".fold"] = y.to_numpy(), folds
        d.to_csv(tmp / "d.csv", index=False)
        cats = [f for f, c in zip(feats, X.columns) if is_cat(X[c])]
        (tmp / "meta.txt").write_text(",".join(feats) + "\n" + ",".join(cats) + "\n")
        (tmp / "run.R").write_text(R_SCRIPT)
        subprocess.run(["Rscript", str(tmp / "run.R"), str(tmp / "d.csv"), str(tmp / "meta.txt"),
                        str(tmp / "out.csv")], check=True, capture_output=True)
        return pd.read_csv(tmp / "out.csv")


# --------------------------------------------------------------------------- driver
def bench_dataset(did: int) -> pd.DataFrame:
    X, y, name = prepare(did)
    folds = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        folds[te] = k
    rows = []
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        args = (X[tr], y[tr], X[te], y[te])
        for model, fn in [("c50py", lambda: run_c50py(*args)),
                          ("c50py_cv", lambda: run_c50py(*args, variant="cv")),
                          ("c50py_winnow", lambda: run_c50py(*args, variant="winnow")),
                          ("c50py_rules", lambda: run_c50py(*args, variant="rules")),
                          ("c50py_boost", lambda: run_c50py(*args, variant="boost")),
                          ("cart", lambda: run_cart(*args, tuned=False)),
                          ("cart_cv", lambda: run_cart(*args, tuned=True)),
                          ("hgb", lambda: run_hgb(*args))]:
            if model not in MODELS:
                continue
            rows.append(dict(dataset=name, openml_id=did, model=model, fold=k, **fn()))
    r = run_r(X, y, folds) if any(m.startswith("C5.0_R") for m in MODELS) else None
    if r is not None:
        names = {"tree": "C5.0_R", "rules": "C5.0_R_rules", "boost": "C5.0_R_boost"}
        for _, row in r.iterrows():
            if names[row["mode"]] not in MODELS:
                continue
            rows.append(dict(dataset=name, openml_id=did, model=names[row["mode"]], fold=int(row.fold),
                             acc=row.acc, leaves=row.leaves, rule_len=np.nan, rule_vars=np.nan,
                             features=row.features, fit_s=row.fit_s))
    df = pd.DataFrame(rows)
    df["n"], df["p"], df["p_cat"] = len(X), X.shape[1], sum(is_cat(X[c]) for c in X.columns)
    RESULTS.mkdir(exist_ok=True)
    df["c50py_version"] = c50py.__version__
    df.to_csv(RESULTS / f"folds{TAG}_{did}.csv", index=False)
    print(f"done {name}", flush=True)
    return df


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or DATASETS
    for did in ids:          # download sequentially, then run in parallel
        load(did)
    n_jobs = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
    dfs = Parallel(n_jobs=n_jobs)(delayed(bench_dataset)(d) for d in ids)
    pd.concat(dfs).to_csv(RESULTS / f"benchmark_folds{TAG}.csv", index=False)
