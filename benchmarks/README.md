# Benchmark

c50py against scikit-learn's CART (one-hot encoded), HistGradientBoosting and the original C5.0 (R
package `C50`) on 23 OpenML classification datasets with categorical columns. Every model sees the same 5 stratified folds.

```bash
pip install -e ".[dev]" scipy tabulate
python benchmarks/run_benchmark.py          # downloads the data from OpenML, writes results/folds_*.csv
python benchmarks/analyze_benchmark.py      # results/summary.md and results/benchmark.png
```

The C5.0 baseline needs R with the `C50` package (`install.packages("C50")`); without `Rscript` it is
skipped. `MODELS=c50py,c50py_cv TAG=_new python benchmarks/run_benchmark.py` reruns only some models
and names the output files with the tag; `analyze_benchmark.py` reads `results/benchmark_folds{C50PY_TAG}.csv`
(`C50PY_TAG=_final` by default), so that all models come from the same run.

Models: `c50py` (defaults, C5.0's `cf=0.25` and `minCases=2`), `c50py_cv` (`cf` tuned by 3-fold inner
CV), `c50py_winnow` (`winnow=True`), `c50py_rules` (`C5RulesClassifier()`), `c50py_boost`
(`trials=10`), `cart` (scikit-learn defaults), `cart_cv` (`ccp_alpha` and `min_samples_leaf` tuned by
5-fold inner CV), `hgb` (`HistGradientBoostingClassifier` with native categorical support), `C5.0_R`
(`C5.0()` defaults), `C5.0_R_rules` (`rules = TRUE`), `C5.0_R_boost` (`trials = 10`). Datasets larger than 10,000 rows are subsampled (stratified).
Metrics per fold: test accuracy, leaves, conditions per rule, distinct columns per rule, columns used,
fit time.
