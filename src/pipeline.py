"""Reproducible one-month-ahead market-level maize price forecasting."""
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

RAW = Path('data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv')
MODEL_DIR = Path('models')
REPORT_DIR = Path('reports')
FEATURES = ['current_price','lag_1','lag_2','lag_3','lag_6','lag_12','rolling_3','rolling_6','rolling_12','month','year','market']
NUMERIC = [c for c in FEATURES if c != 'market']


def prepare(path=RAW):
    df = pd.read_csv(path, low_memory=False)
    required = {'price_date','mkt_name','c_maize_fao'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    df['date'] = pd.to_datetime(df['price_date'], errors='raise')
    df['price'] = pd.to_numeric(df['c_maize_fao'], errors='coerce')
    df = df.sort_values(['mkt_name','date']).copy()
    if df.duplicated(['mkt_name','date']).any():
        raise ValueError('Duplicate market-date rows; audit the source first')
    g = df.groupby('mkt_name', sort=False)['price']
    df['current_price'] = df['price']
    for lag in [1,2,3,6,12]:
        df[f'lag_{lag}'] = g.shift(lag)
    for window in [3,6,12]:
        df[f'rolling_{window}'] = g.transform(lambda s: s.rolling(window, min_periods=window).mean())
    df['target'] = g.shift(-1)
    df['target_date'] = df.groupby('mkt_name', sort=False)['date'].shift(-1)
    # Reject non-consecutive observations; never label a gap as a one-month forecast.
    valid_next = df['target_date'].eq(df['date'] + pd.offsets.MonthBegin(1))
    df = df.loc[valid_next].copy()
    df['month'] = df['target_date'].dt.month
    df['year'] = df['target_date'].dt.year
    df['market'] = df['mkt_name'].astype(str)
    df = df.dropna(subset=['target','current_price','lag_12','rolling_12'])
    return df


def model_pipeline(model):
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUMERIC),
        ('market', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['market'])
    ])
    return Pipeline([('preprocess',prep),('model',model)])


def scores(y, pred):
    return {'MAE':round(float(mean_absolute_error(y,pred)),4),
            'RMSE':round(float(np.sqrt(mean_squared_error(y,pred))),4)}


def train(input_path=RAW):
    df=prepare(input_path)
    train_df=df[df.target_date < '2023-01-01']
    valid_df=df[(df.target_date >= '2023-01-01') & (df.target_date < '2025-01-01')]
    test_df=df[df.target_date >= '2025-01-01']
    if min(len(train_df),len(valid_df),len(test_df)) == 0:
        raise ValueError('At least one temporal split is empty')
    models={
        'ridge':model_pipeline(Ridge(alpha=10.0)),
        'random_forest':model_pipeline(RandomForestRegressor(n_estimators=80,max_depth=16,min_samples_leaf=3,n_jobs=-1,random_state=42)),
        'hist_gradient_boosting':model_pipeline(HistGradientBoostingRegressor(max_iter=130,max_leaf_nodes=24,learning_rate=0.07,l2_regularization=1.0,random_state=42))
    }
    X_train,y_train=train_df[FEATURES],train_df.target
    X_val,y_val=valid_df[FEATURES],valid_df.target
    results={'splits':{k:{'rows':len(s),'first_target_date':str(s.target_date.min().date()),'last_target_date':str(s.target_date.max().date())} for k,s in [('train',train_df),('validation',valid_df),('test',test_df)]},'validation':{},'test':{},'selected_model':None,'target':'next-month c_maize_fao (estimated series)'}
    results['validation']['persistence']=scores(y_val,valid_df.current_price)
    for name,model in models.items():
        model.fit(X_train,y_train)
        results['validation'][name]=scores(y_val,model.predict(X_val))
    # A model is chosen using validation only. The persistence baseline may win.
    selected=min(results['validation'],key=lambda k:results['validation'][k]['MAE'])
    results['selected_model']=selected
    final_train=pd.concat([train_df,valid_df])
    if selected!='persistence':
        final_model=models[selected]
        final_model.fit(final_train[FEATURES],final_train.target)
        MODEL_DIR.mkdir(exist_ok=True,parents=True)
        joblib.dump({'pipeline':final_model,'features':FEATURES,'model_name':selected},MODEL_DIR/'forecast_model.joblib')
        pred=final_model.predict(test_df[FEATURES])
    else:
        MODEL_DIR.mkdir(exist_ok=True,parents=True)
        joblib.dump({'pipeline':None,'features':FEATURES,'model_name':'persistence'},MODEL_DIR/'forecast_model.joblib')
        pred=test_df.current_price.to_numpy()
    (MODEL_DIR/'model_metadata.json').write_text(json.dumps({'selected_model':selected,'prediction_rule':'next_price = current_price' if selected=='persistence' else 'serialized sklearn pipeline','features':FEATURES,'training_target':'next-month c_maize_fao'},indent=2)+'\n')
    results['test'][selected]=scores(test_df.target,pred)
    results['test']['persistence']=scores(test_df.target,test_df.current_price)
    REPORT_DIR.mkdir(exist_ok=True,parents=True)
    (REPORT_DIR/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    prediction_table=test_df[['mkt_name','date','target_date','current_price','target']].copy()
    prediction_table['prediction']=pred
    prediction_table.to_csv(REPORT_DIR/'test_predictions.csv',index=False)
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=RAW)
    args=parser.parse_args()
    print(json.dumps(train(args.input),indent=2))
