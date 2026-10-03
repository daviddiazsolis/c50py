# Notes for AI coding agents working on this repository

## Layout

- `src/c50py/tree.py`: `C5Classifier` (split search, pessimistic pruning, subtree raising,
  C5.0 global pruning, winnowing, C5.0 boosting, rules of the tree).
- `src/c50py/rules.py`: `C5RulesClassifier` (C5.0 rulesets).
- `src/c50py/regressor.py`: `C5Regressor`.
- `src/c50py/_deploy.py`: exports of rules to SQL, JSON and pandas queries.
- `src/c50py/_validation.py`: input validation shared by the estimators (scikit-learn conventions).
- `src/c50py/_export.py`: `plot_tree`, `export_graphviz`, `export_text` (adapted from scikit-learn).
- `tests/`: pytest suite, including scikit-learn's `check_estimator` and fidelity tests against C4.5/C5.0.
- `benchmarks/`: benchmark against CART, HistGradientBoosting and the R package C50.
- `examples/notebooks/`: documentation notebooks (built and executed, outputs committed).

## Rules

- Run `python -m pytest -q` before proposing a change; all tests must pass.
- Keep the estimators scikit-learn compatible: no logic or conversion in `__init__`, validate in
  `fit`, fitted attributes end in `_`.
- Do not change defaults or results silently: a change that alters fitted trees goes in
  `CHANGELOG.md` and, if it is a fidelity change, is checked with `benchmarks/run_benchmark.py`.
- The implementation is original Python/NumPy code written from the published descriptions of
  C4.5/C5.0; do not copy code from the GPL C source of C5.0.
- Text style in docs: no em dashes, no arrows, plain sentences.
- Releases: bump `version` in `pyproject.toml` and `src/c50py/__init__.py`, add a CHANGELOG entry,
  then `gh release create vX.Y.Z`; the publish workflow uploads to PyPI.
