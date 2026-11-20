"""Advanced V2 API: validated, multi-horizon persistence forecasts.

The evaluated winner was persistence for each horizon. LightGBM is a research
comparator, NOT served here without a trained and validated serialized model.
"""
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_METRICS = Path(__file__).resolve().parents[1] / 'reports' / 'advanced' / 'metrics.json'
app = FastAPI(title='Kenya Maize Forecasting — Advanced V2', version='2.0.0')


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    market: str = Field(min_length=1, max_length=150)
    current_price: float = Field(gt=0, allow_inf_nan=False, description='Current model-estimated maize price in source units')
    origin_year: int = Field(ge=2007, le=2100)
    origin_month: int = Field(ge=1, le=12)
    horizon: int = Field(ge=1, le=3)


@lru_cache(maxsize=1)
def load_metrics():
    path = Path(os.environ.get('ADVANCED_METRICS_PATH', str(DEFAULT_METRICS)))
    if not path.is_file():
        raise FileNotFoundError(f'Advanced evaluation metrics missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    for horizon in ('1', '2', '3'):
        item = data.get('horizons', {}).get(horizon)
        if not item or item.get('selected_model') != 'persistence' or not isinstance(item.get('interval_half_width'), (int, float)):
            raise ValueError(f'Missing or unsupported evaluated model for horizon {horizon}')
        if item['interval_half_width'] < 0:
            raise ValueError('Negative interval width')
    return data


def target_month(year, month, horizon):
    absolute = year * 12 + month - 1 + horizon
    return absolute // 12, absolute % 12 + 1


@app.get('/health')
def health():
    try:
        load_metrics()
        return {'status': 'ok', 'version': '2.0.0', 'ready': True}
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get('/models')
def models():
    try:
        data = load_metrics()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {'production_model': 'persistence',
            'horizons': {h: {'selected_model': m['selected_model'],
                             'test_mae': m['test']['mae'],
                             'test_rmse': m['test']['rmse'],
                             'interval_nominal_coverage': m['interval_nominal_coverage'],
                             'interval_empirical_coverage': m['interval_empirical_coverage']}
                         for h, m in data['horizons'].items()},
            'research_comparators': ['ridge', 'hist_gradient_boosting', 'lightgbm'],
            'source_warning': data.get('source_warning')}


@app.post('/predict')
def predict(req: ForecastRequest):
    try:
        m = load_metrics()['horizons'][str(req.horizon)]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    year, month = target_month(req.origin_year, req.origin_month, req.horizon)
    q = float(m['interval_half_width'])
    return {'market': req.market.strip(),
            'origin_date': f'{req.origin_year:04d}-{req.origin_month:02d}',
            'target_date': f'{year:04d}-{month:02d}',
            'horizon_months': req.horizon,
            'predicted_price': round(req.current_price, 4),
            'interval_80': {'lower': round(max(0., req.current_price - q), 4),
                            'upper': round(req.current_price + q, 4)},
            'model': 'persistence',
            'interval_method': 'validation absolute residual quantile; pooled across markets',
            'estimated_test_coverage': m['interval_empirical_coverage'],
            'warning': 'Forecast is for an estimated price series, not a directly observed market quote. Intervals are empirical and not guaranteed.'}
