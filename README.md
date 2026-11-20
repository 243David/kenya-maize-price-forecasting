# Forecasting Maize Prices Across Kenyan Markets

**DSA 8401 Applied Machine Learning | Strathmore University**  
**Student:** RUHANAMIRINDI BULINDI David | **Student number:** 224688

## Overview
A reproducible one-month-ahead forecasting study using the World Bank Kenya monthly market food-price estimates, 2007–2026. The target is next month's `c_maize_fao`, a **model-estimated price series**, not a directly observed transaction price. The CSV has 55,224 rows and 234 markets; only 2,997 entries in the directly observed `maize_fao` field are populated.

## Results
Validation uses 2023–2024, with training through 2022 and a held-out test set from January 2025 to August 2026.

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Persistence | **1.5558** | **2.5873** |
| Ridge regression | 1.7362 | 2.5896 |
| Random Forest | 2.7840 | 3.9042 |
| Histogram Gradient Boosting | 3.2021 | 4.0426 |

The lowest-validation-MAE model was persistence (next month's price = current price). Its untouched test MAE was **1.5347** and RMSE **2.3291**, based on 4,680 market-month predictions. More complex methods did not improve on the baseline in this experiment.

## Architecture
```
Raw CSV -> data audit -> market-level lags and rolling features
        -> chronological train / validation / test
        -> compare persistence, Ridge, Random Forest, Gradient Boosting
        -> select by validation MAE -> saved artifact -> FastAPI -> Docker
```

## Reproduce
Use Python 3.11+. Obtain `KEN_RTFP_mkt_2007_2026-08-24.csv` from the World Bank dataset and place it under `data/raw/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_audit
python -m src.eda
python -m src.pipeline
pip install pytest httpx
python -m pytest -q
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` to try the API. Alternatively run `docker compose up --build`.

## Live deployment

- **API documentation:** https://kenya-maize-price-forecasting.onrender.com/docs
- **Health check:** https://kenya-maize-price-forecasting.onrender.com/health
- **Prediction endpoint:** `POST https://kenya-maize-price-forecasting.onrender.com/predict`
- **Hosting:** Render Docker Web Service (free instance; cold starts after inactivity).
- **Verified 20 November 2026:** Render reported deployment **Live**; `/health` returned HTTP 200. Swagger UI `POST /predict` returned HTTP 200 and `predicted_price: 60` for an illustrative Nairobi request with `current_price: 60`. An otherwise identical request with `current_price: -60` returned HTTP 422 with validation message `Input should be greater than 0`.
- The deployed service uses the selected persistence rule from `models/model_metadata.json` because the serialized joblib artifact is not in the public GitHub repository. A fresh full-pipeline reproduction still requires downloading the raw data and running the training scripts.

## Important limitations
The estimated historical price series may have been revised using future information; chronological evaluation alone cannot remove that upstream hindsight risk. These results do not establish real-time accuracy on independently observed market prices. The API is deployed on Render as a live demonstration service. This is not evidence of real-time forecasting accuracy.

See `reports/final_report.md`, `reports/metrics.json` and `AI_USE.md` for the report, detailed metrics and AI disclosure.
