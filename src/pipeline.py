"""Reproducible market-level one-month-ahead price forecasting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW = Path("data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv")
FEATURES = ["current_price","lag_1","lag_2","lag_3","lag_6","lag_12","rolling_3","rolling_6","rolling_12","month","year","market"]
NUMERIC = [x for x in FEATURES if x != "market"]

def prepare(path=RAW):
    df = pd.read_csv(path, low_memory=False)
    required = {"price_date","mkt_name","c_maize_fao"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df["date"] = pd.to_datetime(df.price_date, errors="raise")
    df["price"] = pd.to_numeric(df.c_maize_fao, errors="coerce")
    df = df.sort_values(["mkt_name","date"]).copy()
    if df.duplicated(["mkt_name","date"]).any():
        raise ValueError("Duplicate market-date rows")
    g = df.groupby("mkt_name", sort=False)["price"]
    df["current_price"] = df.price
    for lag in [1,2,3,6,12]:
        df[f"lag_{lag}"] = g.shift(lag)
    for window in [3,6,12]:
        df[f"rolling_{window}"] = g.transform(lambda s: s.rolling(window,min_periods=window).mean())
    df["target"] = g.shift(-1)
    df["target_date"] = df.groupby("mkt_name",sort=False)["date"].shift(-1)
    df = df.loc[df.target_date.eq(df.date + pd.offsets.MonthBegin(1))].copy()
    df["month"] = df.target_date.dt.month
    df["year"] = df.target_date.dt.year
    df["market"] = df.mkt_name.astype(str)
    return df.dropna(subset=["target","current_price","lag_12","rolling_12"])

def model_pipeline(model):
    prep = ColumnTransformer([
        ("num",SimpleImputer(strategy="median"),NUMERIC),
        ("market",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["market"])
    ])
    return Pipeline([("preprocess",prep),("model",model)])

def scores(y,pred):
    return {"MAE":round(float(mean_absolute_error(y,pred)),4),
            "RMSE":round(float(np.sqrt(mean_squared_error(y,pred))),4)}

def train(input_path=RAW):
    df = prepare(input_path)
    training = df[df.target_date < "2023-01-01"]
    validation = df[(df.target_date >= "2023-01-01") & (df.target_date < "2025-01-01")]
    test = df[df.target_date >= "2025-01-01"]
    if min(len(training),len(validation),len(test)) == 0:
        raise ValueError("Empty temporal split")
    models = {
        "ridge":model_pipeline(Ridge(alpha=10.0)),
        "random_forest":model_pipeline(RandomForestRegressor(n_estimators=80,max_depth=16,min_samples_leaf=3,n_jobs=-1,random_state=42)),
        "hist_gradient_boosting":model_pipeline(HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=24,learning_rate=0.07,l2_regularization=1.0,random_state=42))
    }
    results = {"splits":{name:{"rows":len(part),"first_target_date":str(part.target_date.min().date()),"last_target_date":str(part.target_date.max().date())} for name,part in [("train",training),("validation",validation),("test",test)]},"validation":{},"test":{},"selected_model":None,"target":"next-month c_maize_fao (estimated series)"}
    results["validation"]["persistence"] = scores(validation.target,validation.current_price)
    for name,model in models.items():
        model.fit(training[FEATURES],training.target)
        results["validation"][name] = scores(validation.target,model.predict(validation[FEATURES]))
    selected = min(results["validation"],key=lambda x:results["validation"][x]["MAE"])
    results["selected_model"] = selected
    Path("models").mkdir(exist_ok=True)
    if selected == "persistence":
        pred = test.current_price.to_numpy()
        artifact = {"pipeline":None,"features":FEATURES,"model_name":selected}
    else:
        fitted = models[selected]
        combined = pd.concat([training,validation])
        fitted.fit(combined[FEATURES],combined.target)
        pred = fitted.predict(test[FEATURES])
        artifact = {"pipeline":fitted,"features":FEATURES,"model_name":selected}
    joblib.dump(artifact,"models/forecast_model.joblib")
    Path("models/model_metadata.json").write_text(json.dumps({"selected_model":selected,"features":FEATURES,"training_target":"next-month c_maize_fao"},indent=2)+"\n")
    results["test"][selected] = scores(test.target,pred)
    results["test"]["persistence"] = scores(test.target,test.current_price)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    out = test[["mkt_name","date","target_date","current_price","target"]].copy()
    out["prediction"] = pred
    out.to_csv("reports/test_predictions.csv",index=False)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=RAW)
    args = parser.parse_args()
    print(json.dumps(train(args.input),indent=2))
