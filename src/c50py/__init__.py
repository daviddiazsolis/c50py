# c50py/__init__.py
"""
c50py: C5.0-like Decision Trees in pure Python (scikit-learn style).

Exports:
    - C5Classifier
    - C5Regressor
    - C5RulesClassifier (C5.0-style rulesets)
    - plot_tree, export_graphviz, export_text  (same API and look as sklearn.tree)
"""
from .tree import C5Classifier
from .regressor import C5Regressor
from .rules import C5RulesClassifier
from ._export import plot_tree, export_graphviz, export_text

__all__ = ["C5Classifier", "C5RulesClassifier", "C5Regressor", "plot_tree", "export_graphviz", "export_text"]
__version__ = "0.5.0"
