"""Advanced, leakage-aware multi-horizon experimental pipeline.

Run: python -m src.advanced_pipeline
The World Bank c_maize_fao field is an ESTIMATED series, not observed ground truth.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

RAW=Path('data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv')
OUT=Path('reports/advanced')
HORIZONS=(1,2,3)
FEATURES=['price','lag1','lag3','lag6','lag12','ma3','ma6','ma12','ret1','ret3','ret12','vol6','month_sin','month_cos']

def prepare(path=RAW):
    d=pd.read_csv(path,low_memory=False,usecols=['price_date','mkt_name','c_maize_fao','maize_fao'])
    d['date']=pd.to_datetime(d.price_date)
    d['price']=pd.to_numeric(d.c_maize_fao,errors='coerce')
    d=d.sort_values(['mkt_name','date']).reset_index(drop=True)
    if d.duplicated(['mkt_name','date']).any(): raise ValueError('Duplicate market-month')
    g=d.groupby('mkt_name',sort=False)['price']
    for n in (1,3,6,12): d[f'lag{n}']=g.shift(n)
    for n in (3,6,12): d[f'ma{n}']=g.transform(lambda s:s.rolling(n,min_periods=n).mean())
    for n in (1,3,12): d[f'ret{n}']=np.log(d.price/g.shift(n))
    d['vol6']=g.transform(lambda s:np.log(s/s.shift(1)).rolling(6,min_periods=6).std())
    d['month_sin']=np.sin(2*np.pi*d.date.dt.month/12)
    d['month_cos']=np.cos(2*np.pi*d.date.dt.month/12)
    for h in HORIZONS:
        d[f'target{h}']=g.shift(-h)
        d[f'date{h}']=d.groupby('mkt_name',sort=False).date.shift(-h)
        d.loc[d[f'date{h}']!=d.date+pd.DateOffset(months=h),f'target{h}']=np.nan
    return d.dropna(subset=FEATURES).copy()

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    return {'mae':round(float(np.mean(np.abs(y-p))),4),'rmse':round(float(np.sqrt(np.mean((y-p)**2))),4)}

def train(path=RAW):
    d=prepare(path); OUT.mkdir(parents=True,exist_ok=True)
    summary={'source_warning':'Target c_maize_fao is a model-estimated series; estimated targets may incorporate hindsight information.', 'horizons':{}, 'features':FEATURES}
    predictions=[]
    for h in HORIZONS:
        v=d.dropna(subset=[f'target{h}']).copy()
        yname=f'target{h}'; dt=f'date{h}'
        tr=v[v[dt]<'2023-01-01']; va=v[(v[dt]>='2023-01-01')&(v[dt]<'2025-01-01')]; te=v[v[dt]>='2025-01-01']
        if min(len(tr),len(va),len(te))==0:raise ValueError('Empty chronological split')
        # Forecast log change, not price level; only information from forecast origin is used.
        def target(z):return np.log(z[yname].to_numpy()/z.price.to_numpy())
        candidates={'persistence':None,
            'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=20)),
            'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=20,learning_rate=.05,l2_regularization=2,random_state=42),
            'lightgbm':LGBMRegressor(n_estimators=160,num_leaves=15,learning_rate=.035,max_depth=5,min_child_samples=100,reg_lambda=3,verbosity=-1,n_jobs=2,random_state=42)}
        valpred={}; valmetrics={}
        for name,model in candidates.items():
            if model is None: pv=va.price.to_numpy()
            else:
                model.fit(tr[FEATURES],target(tr))
                pv=va.price.to_numpy()*np.exp(np.clip(model.predict(va[FEATURES]),-.5,.5))
            valpred[name]=pv;valmetrics[name]=metrics(va[yname].to_numpy(),pv)
        winner=min(valmetrics,key=lambda k:valmetrics[k]['mae'])
        # Conservative split conformal intervals: calibration residuals from validation;
        # re-fit point model on train only, avoiding reusing calibration data for fitting.
        model=candidates[winner]
        if model is None: pt=te.price.to_numpy()
        else: pt=te.price.to_numpy()*np.exp(np.clip(model.predict(te[FEATURES]),-.5,.5))
        residual=np.abs(va[yname].to_numpy()-valpred[winner]); q=float(np.quantile(residual,.8,method='higher'))
        actual=te[yname].to_numpy(); lo=np.maximum(0,pt-q);hi=pt+q
        baseline=te.price.to_numpy()
        result={'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),
                'selected_model':winner,'validation':valmetrics,'test':metrics(actual,pt),
                'test_persistence':metrics(actual,baseline),'interval_nominal_coverage':.8,
                'interval_empirical_coverage':round(float(np.mean((actual>=lo)&(actual<=hi))),4),
                'interval_half_width':round(q,4),
                'observed_survey_origin_rows_test':int(te.maize_fao.notna().sum())}
        # Paired month-block bootstrap for difference in MAE vs persistence.
        diff=np.abs(actual-pt)-np.abs(actual-baseline)
        block=pd.DataFrame({'month':te[dt].dt.strftime('%Y-%m').to_numpy(),'diff':diff}).groupby('month')['diff'].mean().to_numpy()
        rng=np.random.default_rng(42+h)
        samples=rng.choice(block,size=(1000,len(block)),replace=True).mean(axis=1)
        result['mae_difference_vs_persistence_month_bootstrap_95ci']=[round(float(x),4) for x in np.quantile(samples,[.025,.975])]
        summary['horizons'][str(h)]=result
        predictions.append(pd.DataFrame({'market':te.mkt_name,'origin_date':te.date,'target_date':te[dt], 'horizon':h,'actual_estimated':actual,'prediction':pt,'lower_80':lo,'upper_80':hi,'baseline':baseline}))
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':print(json.dumps(train(),indent=2))
