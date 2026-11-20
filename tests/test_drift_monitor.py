import numpy as np
import pandas as pd
from src.drift_monitor import population_stability_index, assess

def test_identical_distributions():
    x=np.arange(100,dtype=float)
    assert abs(population_stability_index(x,x))<1e-9

def test_shift_detected():
    x=np.arange(1000,dtype=float)
    assert population_stability_index(x,x+500)>0.25

def test_constant_reference_not_evaluable():
    assert population_stability_index([2,2,2],[3,4]) is None

def test_report_flags_shift():
    a=pd.DataFrame({"x":range(100)})
    b=pd.DataFrame({"x":range(100,200)})
    assert assess(a,b,["x"]).iloc[0]["status"]=="high"
