import numpy as np
import pytest
from c50py import C5Classifier

def test_sample_weights():
    # Case 1: Two identical points, one with weight 2, should be equivalent to 3 identical points
    X = np.array([[1, 1], [1, 1], [2, 2]])
    y = np.array([0, 0, 1])
    w = np.array([1, 2, 1]) # Effective counts: Class 0: 3, Class 1: 1
    
    clf = C5Classifier(min_samples_split=2, trials=1, random_state=42)
    clf.fit(X, y, sample_weight=w)
    
    # Check class distribution in root
    # We can't easily check internal state without private access, 
    # but we can check if it behaves rationally.
    # If we predict, it should work.
    assert clf.predict([[1, 1]])[0] == 0
    assert clf.predict([[2, 2]])[0] == 1

def test_missing_values_propagation():
    # Feature 0 is the split. 
    # Value < 5 -> Class 0
    # Value > 5 -> Class 1
    # Missing -> Distributed
    X = np.array([
        [2.0], [3.0], [4.0], # Class 0
        [6.0], [7.0], [8.0], # Class 1
        [np.nan]             # Missing
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 0]) # The missing one is Class 0, but let's see how it's handled
    
    # If we train, the missing value should be distributed.
    # If we predict on missing, it should give a probability.
    
    clf = C5Classifier(min_samples_split=2, trials=1, random_state=42)
    clf.fit(X, y)
    
    # Predict on knowns
    assert clf.predict([[2.0]])[0] == 0
    assert clf.predict([[8.0]])[0] == 1
    
    # Predict on missing
    # Should be roughly 50/50 if split was even?
    # In training, 3 vs 3 knowns.
    probs = clf.predict_proba([[np.nan]])[0]
    assert np.allclose(probs, [0.5, 0.5], atol=0.2) # Tolerant check

def test_boosting_weights():
    # 1D alternating problem: 0, 1, 0, 1
    # Stumps can't solve this in one go, but boosting stumps can.
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])
    
    # Stump 1: x <= 1.5 -> 0, x > 1.5 -> 1 (error on 3)
    # Stump 2: Focus on 3 (x=3, y=0). Split x > 2.5?
    clf = C5Classifier(trials=10, max_depth=1, random_state=42)
    clf.fit(X, y)
    
    assert clf.score(X, y) == 1.0
