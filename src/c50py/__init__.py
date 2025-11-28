# c50py/__init__.py
"""
c50py: C5.0-like Decision Trees in pure Python (scikit-learn style).

Exports:
    - C5Classifier
    - C5Regressor
"""
from .tree import C5Classifier
from .regressor import C5Regressor

__all__ = ["C5Classifier", "C5Regressor"]
__version__ = "0.2.0"
