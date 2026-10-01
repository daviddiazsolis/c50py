# c50py/__init__.py
"""
c50py: C5.0-like Decision Trees in pure Python (scikit-learn style).

Exports:
    - C5Classifier
    - C5Regressor
    - plot_tree, export_graphviz, export_text  (same API and look as sklearn.tree)
"""
from .tree import C5Classifier
from .regressor import C5Regressor
from ._export import plot_tree, export_graphviz, export_text

__all__ = ["C5Classifier", "C5Regressor", "plot_tree", "export_graphviz", "export_text"]
__version__ = "0.4.1"
