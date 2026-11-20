from src.advanced_pipeline import FEATURES,HORIZONS,metrics

def test_horizons():
    assert HORIZONS==(1,2,3)
    assert 'price' in FEATURES

def test_metrics():
    assert metrics([1,2],[1,3])['mae']==.5
