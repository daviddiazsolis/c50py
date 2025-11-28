import numpy as np
from c5py import C5Classifier, C5Regressor

def test_classifier_smoke():
    X = np.array([[1,'A'],[2,'A'],[3,'B'],[4,'B']], dtype=object)
    y = np.array([0,0,1,1])
    clf = C5Classifier(categorical_features=[1], feature_names=['num','cat'], trials=1, min_samples_leaf=1)
    clf.fit(X,y)
    _ = clf.predict(X)
    _ = clf.export_rules(feature_names=['num','cat'], class_names=['no','yes'])

def test_regressor_smoke():
    X = np.array([[1.0,'A'],[2.0,'A'],[3.0,'B'],[4.0,'B']], dtype=object)
    y = np.array([1.0, 1.5, 2.0, 2.5])
    regr = C5Regressor(categorical_features=[1], feature_names=['num','cat'], min_samples_leaf=1)
    regr.fit(X,y)
    _ = regr.predict(X)
    _ = regr.export_rules(feature_names=['num','cat'])
