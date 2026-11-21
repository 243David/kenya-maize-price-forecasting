from src.explainability import explain

def test_explainability_importable():
    assert callable(explain)

def test_explainability_scope():
    from src.advanced_pipeline import FEATURES
    assert 'price' in FEATURES and 'month_sin' in FEATURES
