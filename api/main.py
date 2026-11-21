"""Prediction service for an already trained next-month forecast model."""
from functools import lru_cache
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title='Kenya Maize Price Forecast API',version='1.0.0')
MODEL_PATH=Path('models/forecast_model.joblib')
METADATA_PATH=Path('models/model_metadata.json')

class ForecastRequest(BaseModel):
    market: str = Field(min_length=1)
    forecast_year: int = Field(ge=2008,le=2100)
    forecast_month: int = Field(ge=1,le=12)
    current_price: float = Field(gt=0)
    lag_1: float = Field(gt=0)
    lag_2: float = Field(gt=0)
    lag_3: float = Field(gt=0)
    lag_6: float = Field(gt=0)
    lag_12: float = Field(gt=0)
    rolling_3: float = Field(gt=0)
    rolling_6: float = Field(gt=0)
    rolling_12: float = Field(gt=0)

@lru_cache(maxsize=1)
def artifact():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        metadata=json.loads(METADATA_PATH.read_text())
        if metadata['selected_model']=='persistence':
            return {'pipeline':None,'features':metadata['features'],'model_name':'persistence'}
    raise FileNotFoundError('Model not trained; run python -m src.pipeline')

@app.get('/health')
def health():
    return {'status':'ok','model_available':MODEL_PATH.exists() or METADATA_PATH.exists()}

@app.post('/predict')
def predict(req:ForecastRequest):
    try:
        model=artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    row=req.model_dump()
    row['year']=row.pop('forecast_year')
    row['month']=row.pop('forecast_month')
    value=float(row['current_price']) if model['pipeline'] is None else float(model['pipeline'].predict(pd.DataFrame([row])[model['features']])[0])
    return {'market':req.market,'forecast_year':req.forecast_year,'forecast_month':req.forecast_month,'predicted_price':round(value,3),'model':model['model_name'],'note':'Forecast of the World Bank model-estimated maize price series; not a direct observed market quote.'}
