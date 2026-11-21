"""Explain the experimental LightGBM candidate, NOT the selected persistence model.

Run: python -m src.explainability
SHAP values explain predicted log price changes. They do not imply causality.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from .advanced_pipeline import prepare, RAW, FEATURES, HORIZONS

OUT = Path('reports/advanced/explainability')

def explain(path=RAW, max_rows=300, random_state=42):
    import shap
    d = prepare(path)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {'scope': 'LightGBM experimental candidate; selected production baseline is persistence',
               'interpretation': 'SHAP values explain LightGBM predictions of log future/current estimated price ratios, not price levels, causal effects, or observed ground truth.',
               'horizons': {}}
    for h in HORIZONS:
        z = d.dropna(subset=[f'target{h}']).copy()
        tr = z[z[f'date{h}'] < '2023-01-01']
        va = z[(z[f'date{h}'] >= '2023-01-01') & (z[f'date{h}'] < '2025-01-01')]
        if tr.empty or va.empty: raise ValueError('Missing training or validation rows')
        y = np.log(tr[f'target{h}'].to_numpy() / tr.price.to_numpy())
        model = LGBMRegressor(n_estimators=160, num_leaves=15, learning_rate=.035,
                             max_depth=5, min_child_samples=100, reg_lambda=3,
                             verbosity=-1, n_jobs=2, random_state=42)
        model.fit(tr[FEATURES], y)
        X = va[FEATURES].sample(n=min(max_rows,len(va)),random_state=random_state).copy()
        explanation = shap.TreeExplainer(model)(X)
        values = np.asarray(explanation.values)
        if values.shape != X.shape: raise ValueError('Unexpected SHAP dimensions')
        importance = pd.DataFrame({'feature':FEATURES,'mean_absolute_shap_log_change':np.abs(values).mean(axis=0)})
        importance = importance.sort_values('mean_absolute_shap_log_change',ascending=False)
        importance.to_csv(OUT/f'feature_importance_h{h}.csv',index=False)
        plt.figure(figsize=(9,6))
        top = importance.head(12).iloc[::-1]
        plt.barh(top.feature,top.mean_absolute_shap_log_change)
        plt.xlabel('Mean absolute SHAP value (predicted log price change)')
        plt.title(f'LightGBM SHAP importance — {h}-month horizon')
        plt.tight_layout()
        plt.savefig(OUT/f'shap_importance_h{h}.png',dpi=160)
        plt.close()
        plt.figure(figsize=(10,6))
        shap.summary_plot(values,X,show=False,max_display=12)
        plt.title(f'LightGBM SHAP summary — {h}-month horizon')
        plt.tight_layout()
        plt.savefig(OUT/f'shap_summary_h{h}.png',dpi=160,bbox_inches='tight')
        plt.close()
        results['horizons'][str(h)]={'sample_rows':len(X),'top_features':importance.head(5).to_dict('records')}
    (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':print(json.dumps(explain(),indent=2))
