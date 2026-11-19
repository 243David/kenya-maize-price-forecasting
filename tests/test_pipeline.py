import pandas as pd
from src.pipeline import prepare, FEATURES

def test_feature_history_and_target(tmp_path):
    dates=pd.date_range("2020-01-01",periods=16,freq="MS")
    path=tmp_path/"sample.csv"
    pd.DataFrame({"price_date":dates.strftime("%Y-%m-%d"),"mkt_name":["A"]*16,"c_maize_fao":range(10,26)}).to_csv(path,index=False)
    data=prepare(path)
    assert len(data)==3
    assert data.iloc[0].current_price==22
    assert data.iloc[0].target==23
    assert data.iloc[0].lag_12==10
    assert set(FEATURES).issubset(data.columns)
