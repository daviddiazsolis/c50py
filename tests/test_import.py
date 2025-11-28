def test_import():
    from c50py import C5Classifier, C5Regressor
    assert C5Classifier and C5Regressor
