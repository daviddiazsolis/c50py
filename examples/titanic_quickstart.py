import pandas as pd
from time import perf_counter
from c5py import C5Classifier

df = pd.read_csv("titanic.csv")
feats = ["pclass","sex","age","sibsp","parch","fare","embarked"]

Xdf = df[feats].copy()
Xdf["pclass"]   = Xdf["pclass"].astype(str)
Xdf["sex"]      = Xdf["sex"].astype(str)
Xdf["embarked"] = Xdf["embarked"].astype(str)

X = Xdf.values.astype(object)
y = df["survived"].astype(int).values

clf = C5Classifier(
    trials=1, min_samples_split=60, min_samples_leaf=20,
    pruning=True, cf=0.25, global_pruning=True, random_state=42,
    feature_names=feats,
    categorical_features=["pclass","sex","embarked"],
    infer_categorical=False, int_as_categorical=False,
    numeric_threshold_strategy="quantile", max_numeric_thresholds=32
)

t0 = perf_counter(); clf.fit(X, y); print(f"fit: {perf_counter()-t0:.3f} s")
clf.print_tree(feature_names=feats, class_names=["No","Yes"])
try:
    clf.export_graphviz("titanic_tree_clean", feature_names=feats, class_names=["No","Yes"], format="dot")
except RuntimeError as e:
    print(f"Skipping Graphviz export: {e}")
