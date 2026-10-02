"""
Summarise the benchmark: overall means, paired Wilcoxon tests across datasets,
results by type of data, per-dataset tables and a figure.

    python benchmarks/analyze_benchmark.py

Reads results/benchmark_folds{TAG}.csv written by run_benchmark.py (TAG is the
environment variable C50PY_TAG, "_final" by default), so that every model comes
from the same run, with the same folds and the same version of c50py.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
C50PY_TAG = os.environ.get("C50PY_TAG", "_final")

LABEL = {"c50py": "c50py tree (C5.0 defaults)", "c50py_cv": "c50py tree, cf tuned by CV",
         "c50py_winnow": "c50py tree, winnow=True", "c50py_rules": "c50py ruleset (C5RulesClassifier)",
         "c50py_boost": "c50py boosting, trials=10",
         "cart": "CART + one-hot (sklearn defaults)", "cart_cv": "CART + one-hot, pruning tuned by CV",
         "hgb": "HistGradientBoosting (native categories)",
         "C5.0_R": "C5.0 original (R), tree", "C5.0_R_rules": "C5.0 original (R), rules=TRUE",
         "C5.0_R_boost": "C5.0 original (R), trials=10"}
ORDER = list(LABEL)
SINGLE = ["c50py", "c50py_cv", "c50py_winnow", "c50py_rules", "cart", "cart_cv",
          "C5.0_R", "C5.0_R_rules"]
COMPARISONS = [
    # fidelity to the original C5.0
    ("c50py", "C5.0_R"), ("c50py_rules", "C5.0_R_rules"), ("c50py_boost", "C5.0_R_boost"),
    # against scikit-learn's tree
    ("c50py", "cart"), ("c50py_cv", "cart_cv"), ("c50py", "cart_cv"), ("c50py_rules", "cart_cv"),
    # c50py options against the default tree
    ("c50py_rules", "c50py"), ("c50py_winnow", "c50py"), ("c50py_cv", "c50py"),
    # readable models against gradient boosting
    ("c50py", "hgb"), ("c50py_rules", "hgb"), ("c50py_boost", "hgb"),
]


def load() -> pd.DataFrame:
    d = pd.read_csv(RES / f"benchmark_folds{C50PY_TAG}.csv")
    return d.drop_duplicates(["openml_id", "model", "fold"])


def data_type(row) -> str:
    if row.p_cat == 0:
        return "numeric only"
    if row.p_cat == row.p:
        return "categorical only"
    return "mixed"


def main():
    d = load()
    per = d.groupby(["dataset", "model"])[["acc", "leaves", "rule_len", "rule_vars", "features", "fit_s"]].mean()
    meta = d.groupby("dataset")[["n", "p", "p_cat"]].first()
    meta["type"] = meta.apply(data_type, axis=1)
    models = [m for m in ORDER if m in per.index.get_level_values("model")]
    acc = per["acc"].unstack()[models]
    leaves = per["leaves"].unstack()[models]
    complete = acc.dropna().index
    acc, leaves = acc.loc[complete], leaves.loc[complete]
    version = d.c50py_version.dropna().iloc[0] if "c50py_version" in d else "?"

    out = [f"# Benchmark results (c50py {version}, {len(complete)} OpenML datasets, 5-fold CV)\n",
           "Leaves are rules for the rulesets. Boosted models and HGB have no single size.\n",
           "## Mean over datasets\n"]
    summ = per.loc[complete].groupby("model").mean().loc[models]
    summ.index = [LABEL[m] for m in summ.index]
    out.append(summ.rename(columns={"acc": "accuracy", "leaves": "leaves / rules", "rule_len": "conditions/rule",
                                    "rule_vars": "distinct vars/rule", "features": "columns used",
                                    "fit_s": "fit time (s)"}).round(3).to_markdown())

    out.append("\n\n## Paired comparisons across datasets (Wilcoxon signed-rank)\n")
    out.append("Wins/ties/losses count datasets where the first model is more accurate by more than "
               "0.05 points, within 0.05 points, or less accurate. The leaves ratio is the geometric "
               "mean of (leaves of the first) / (leaves of the second); below 1 means smaller.\n")
    rows = []
    for a, b in COMPARISONS:
        if a not in acc or b not in acc:
            continue
        da = acc[a] - acc[b]
        p_acc = wilcoxon(acc[a], acc[b]).pvalue if (da != 0).any() else 1.0
        row = {"comparison": f"{a} vs {b}",
               "mean accuracy difference": round(da.mean(), 4),
               "wins/ties/losses": f"{(da > 0.0005).sum()}/{(da.abs() <= 0.0005).sum()}/{(da < -0.0005).sum()}",
               "p (accuracy)": round(p_acc, 3), "leaves ratio": "", "p (leaves)": ""}
        if a in SINGLE and b in SINGLE:
            la, lb = np.log(leaves[a]), np.log(leaves[b])
            row["leaves ratio"] = round(float(np.exp(np.mean(la - lb))), 2)
            row["p (leaves)"] = round(wilcoxon(la, lb).pvalue, 4) if (la != lb).any() else 1.0
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False))

    out.append("\n\n## By type of data: mean accuracy\n")
    types = meta.loc[complete, "type"]
    out.append(f"Datasets per type: {types.value_counts().to_dict()}\n")
    out.append(acc.groupby(types).mean().T.rename(index=LABEL).round(3).to_markdown())
    out.append("\n\n## By type of data: mean leaves / rules\n")
    out.append(leaves[[m for m in SINGLE if m in leaves]].groupby(types).mean().T.rename(index=LABEL)
               .round(1).to_markdown())

    out.append("\n\n## Per dataset: accuracy\n")
    out.append(meta.loc[complete].join(acc.round(3)).to_markdown())
    out.append("\n\n## Per dataset: leaves / rules\n")
    out.append(leaves[[m for m in SINGLE if m in leaves]].round(1).to_markdown())
    out.append("\n\n## Per dataset: mean distinct columns per rule\n")
    rv = per["rule_vars"].unstack()
    out.append(rv[[m for m in ["c50py", "c50py_cv", "c50py_rules", "cart", "cart_cv"] if m in rv]]
               .loc[complete].round(2).to_markdown())
    (RES / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    figure(acc, leaves)


def figure(acc: pd.DataFrame, leaves: pd.DataFrame):
    """Two small multiples sharing the dataset axis: accuracy and leaves (log)."""
    models = ["c50py", "cart_cv", "C5.0_R"]
    color = {"c50py": "#2a78d6", "cart_cv": "#eb6834", "C5.0_R": "#1baf7a"}
    marker = {"c50py": "o", "cart_cv": "s", "C5.0_R": "D"}
    name = {"c50py": "c50py (C5.0 defaults)", "cart_cv": "CART + one-hot, tuned by CV",
            "C5.0_R": "C5.0 original (R)"}
    order = (leaves["cart_cv"] / leaves["c50py"]).sort_values().index
    y = np.arange(len(order))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(order) + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    fig.patch.set_facecolor("#fcfcfb")
    for ax, data, title, log in [(axes[0], acc, "Test accuracy", False),
                                 (axes[1], leaves, "Leaves (log scale, fewer is simpler)", True)]:
        ax.set_facecolor("#fcfcfb")
        for i in y:
            vals = data.loc[order[i], models].to_numpy(dtype=float)
            ax.plot([np.nanmin(vals), np.nanmax(vals)], [i, i], color=grid, lw=2, zorder=1)
        for m in models:
            ax.scatter(data.loc[order, m], y, s=46, color=color[m], marker=marker[m], label=name[m],
                       edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        if log:
            ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9, length=0)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.6 / len(order) - 0.04), ncol=3,
                   frameon=False, fontsize=9.5, labelcolor=ink)
    fig.suptitle("c50py vs CART with one-hot encoding vs the original C5.0, "
                 f"{len(order)} OpenML datasets, 5-fold CV", x=0.01, ha="left", fontsize=12, color=ink)
    fig.savefig(RES / "benchmark.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
