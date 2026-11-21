import numpy as np
import pandas as pd
import pytest
from src.drift_monitor import assess, drift_status, population_stability_index


def test_identical_distributions_have_zero_psi():
    x = np.arange(100, dtype=float)
    assert population_stability_index(x, x) == pytest.approx(0, abs=1e-9)


def test_shift_detected():
    x = np.arange(1000, dtype=float)
    assert population_stability_index(x, x + 500) > 0.25


def test_empty_and_constant_reference():
    assert population_stability_index([], [1, 2]) is None
    assert population_stability_index([2, 2, 2], [3, 4]) is None
    assert drift_status(None) == 'not_evaluable'


def test_report_and_missing_feature():
    a = pd.DataFrame({'x': range(100)})
    b = pd.DataFrame({'x': range(100, 200)})
    report = assess(a, b, ['x'])
    assert report.iloc[0]['status'] == 'high'
    with pytest.raises(KeyError):
        assess(a, b, ['not_there'])
