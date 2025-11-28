import pandas as pd, numpy as np
from time import perf_counter
from c5py import C5Regressor

df = pd.read_csv("diabetes.csv")
y = df["target"].values
Xdf = df.drop(columns=["target"])
X = Xdf.values.astype(object)
feats = list(Xdf.columns)

reg = C5Regressor(
    min_samples_split=30, min_samples_leaf=10,
    pruning=True, cf=0.25, global_pruning=True,
    feature_names=feats, random_state=42,
    infer_categorical=False, int_as_categorical=False,
    numeric_threshold_strategy="quantile", max_numeric_thresholds=64
)

t0 = perf_counter(); reg.fit(X, y); print(f"fit: {perf_counter()-t0:.3f} s")
try:
    reg.export_graphviz("diabetes_tree", feature_names=feats, format="dot")
except RuntimeError as e:
    print(f"Skipping Graphviz export: {e}")
reg.print_tree()
