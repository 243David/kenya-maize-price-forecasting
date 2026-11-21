# Forecasting Maize Prices Across Kenyan Markets

**DSA 8401 Applied Machine Learning | Strathmore University**  
**Student:** RUHANAMIRINDI BULINDI David | **Student number:** 224688

## Live Application — Advanced V2

The Kenya Maize Price Forecasting project has been extended into an interactive forecasting and model-evaluation system, deployed online for demonstration and academic assessment.

### Access the application

**[Open the Interactive Forecasting Dashboard](https://kenya-maize-dashboard.onrender.com)**

**[Open the Advanced V2 Forecasting API](https://kenya-maize-price-forecasting.onrender.com/docs)**

The dashboard allows users to explore maize price forecasts and evaluate the performance of the selected forecasting model.

### Advanced V2 features

- **Multi-horizon forecasting:** Generate maize price forecasts for 1, 2, and 3 months ahead.
- **Interactive dashboard:** Configure forecast inputs, visualize results, and download forecast data as CSV.
- **Prediction intervals:** Display empirical 80% prediction intervals to communicate forecasting uncertainty.
- **Model evaluation:** Compare Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and prediction interval coverage across forecasting horizons.
- **Machine learning experiments:** Evaluate persistence against regression-based alternatives, including Ridge, histogram gradient boosting, and LightGBM.
- **Explainability:** Include SHAP-based analysis within the advanced research workflow.
- **Drift monitoring:** Provide offline monitoring capabilities to investigate changes in data distributions and forecasting performance.
- **REST API:** Access forecasting functionality through FastAPI endpoints for health checks, model information, and predictions.

### Production model

The persistence model was selected for all three forecasting horizons based on the evaluation results.

| Forecast horizon | Test MAE | Test RMSE | Empirical 80% coverage |
|---|---:|---:|---:|
| 1 month | 1.5347 | 2.3291 | 79.21% |
| 2 months | 2.0446 | 3.0071 | 81.39% |
| 3 months | 2.4620 | 3.4773 | 84.55% |

Although more complex machine learning methods were investigated, persistence remained the selected production model. This demonstrates the importance of evaluating models against a strong baseline rather than assuming that greater complexity guarantees better forecasting accuracy.

**Research limitation:** The forecasting target is the World Bank's model-estimated maize price series (`c_maize_fao`), rather than directly observed market transaction prices. Historical estimates may incorporate hindsight information. The reported performance should therefore not be interpreted as validated real-time market forecasting accuracy.

*Note: The free hosted applications may take approximately one minute to respond after a period of inactivity.*

## What the project does

This project develops and evaluates a reproducible maize price forecasting system using the World Bank's model-estimated maize price series across 234 Kenyan markets.

The original experiment focused on one-month-ahead forecasting and compared persistence against Ridge Regression, Random Forest, and Histogram Gradient Boosting using chronological training, validation, and testing periods.

The project was subsequently extended into **Advanced V2**, supporting forecasts for 1, 2, and 3 months ahead, empirical prediction intervals, model evaluation, explainability research, drift monitoring, and an interactive web dashboard.

The system is deployed through FastAPI and Streamlit, allowing users to generate forecasts, explore uncertainty, compare forecasting performance, and download results.

**Key finding:** Persistence outperformed the evaluated machine learning alternatives during model selection and was retained as the production forecasting approach. This finding highlights the importance of benchmarking complex models against simple, reliable alternatives.

## Data and important caveat

The source is the World Bank's Kenya *Monthly Food Price Estimates by Product and Market*, 2007–2026, distributed as `KEN_RTFP_mkt_2007_2026-08-24.csv`. The original ZIP is available at the World Bank source; place the extracted CSV under `data/raw/`.

There are **55,224 market-month records**, **234 markets**, **2,997 non-missing directly observed maize prices** (`maize_fao`) and **55,224 model-estimated maize-price entries** (`c_maize_fao`). Our prediction target is **next month's `c_maize_fao`**, not a directly observed market transaction. The dataset's historical estimates may have been retrospectively revised and may contain information not available in real time. Therefore, scores measure predictability of this published historical estimated series and should **not** be interpreted as demonstrated live forecasting performance.

## Architecture

World Bank Maize Price Dataset
             |
             v
Data Quality Assessment and EDA
             |
             v
Feature Engineering and Preprocessing
             |
             v
Chronological Train / Validation / Test
             |
             v
Model Development and Comparison
(Persistence, Ridge, RF, HGB, LightGBM)
             |
             v
Select Best Model by Validation MAE
             |
             v
1-, 2-, and 3-Month Forecasting
             |
             v
80% Prediction Intervals
             |
             v
Model Evaluation + Explainability
             |
             v
Drift Monitoring
             |
             v
FastAPI Advanced V2
             |
             v
Interactive Streamlit Dashboard
             |
             v
Forecast Visualization + CSV Export

## Setup and reproduce

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
# Extract KEN_RTFP_mkt_2007_2026-08-24.csv into data/raw/
python -m src.data_audit
python -m src.eda
python -m src.pipeline
python -m pytest -q
uvicorn api.main:app --reload
```

To run the tests, also install `pytest` and `httpx`: `pip install pytest httpx`.

API docs: `http://localhost:8000/docs`; health: `http://localhost:8000/health`.

Docker (after cloning):

```bash
docker compose up --build
```

### Example API request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"market":"Adele Center","forecast_year":2026,"forecast_month":9,"current_price":60,"lag_1":59,"lag_2":58,"lag_3":57,"lag_6":54,"lag_12":48,"rolling_3":59,"rolling_6":57.5,"rolling_12":54}'
```

This returns `predicted_price: 60.0` under the selected persistence model. **These request values are illustrative, not real September 2026 observations.**

## Experiment results

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Persistence | **1.5558** | **2.5873** |
| Ridge | 1.7362 | 2.5896 |
| Random Forest | 2.7840 | 3.9042 |
| Histogram Gradient Boosting | 3.2021 | 4.0426 |

**Held-out test (2025-01 to 2026-08):** selected persistence model MAE **1.5347**, RMSE **2.3291**, over 4,680 market-month forecasts. Scores are in the source dataset's price units. These are results from the current one-split experiment, not uncertainty bounds.

## Project files

- `src/data_audit.py`: data-quality audit
- `src/eda.py`: three exploratory plots and a market summary
- `src/pipeline.py`: features, chronological split, model comparison, selection and test predictions
- `api/main.py`: FastAPI service with input validation
- `models/model_metadata.json`: deployable selected-baseline model definition
- `reports/metrics.json`: validation/test metrics
- `reports/final_report.md`: full interpretation, decisions and limitations
- `Dockerfile`, `docker-compose.yml`: container deployment
- `tests/`: automated checks
- `AI_USE.md`: AI assistance disclosure
  
### Advanced V2 components

- `api_v2/main.py`: Advanced FastAPI service with forecasting, model metadata, and health endpoints.
- `dashboard/app.py`: Interactive Streamlit forecasting dashboard.
- `requirements-api-v2.txt`: Dependencies for the Advanced V2 API.
- `requirements-dashboard.txt`: Dependencies for the Streamlit dashboard.
- `Dockerfile.v2`: Container configuration for the Advanced V2 API.
- `reports/advanced/metrics.json`: Advanced model evaluation results.
- `API_V2_README.md`: Advanced API documentation.
- `ADVANCED_README.md`: Advanced project documentation, where available.

The `advanced-v2` branch is retained for continued development, while `main` contains the integrated project.

Raw source data are omitted from the public repository to avoid duplicating the external dataset; the downloadable project bundle provided separately includes them for reproduction. The compact selected-model metadata is versioned; a full `joblib` artifact is regenerated by the training script. Never load untrusted joblib files.

## Limitations

A static, retrospectively estimated series is not the same as real-time market quotes. One holdout split does not demonstrate robustness across economic regimes. Market identifiers, feature histories and price units must be checked carefully before operational use. The Advanced V2 API and interactive dashboard are publicly deployed as academic demonstration services. Availability depends on the hosting platform, and the system is not intended for operational trading or commercial price decisions.
