# Forecasting Maize Prices Across Kenyan Markets

**DSA 8401 — Applied Machine Learning | Strathmore University**  
**Student:** David Bulindi | **Student number:** 224688

## Research question
Can models using historical market-level prices and calendar information forecast next month's estimated maize price more accurately than a last-price persistence baseline?

## Dataset
World Bank Microdata Library, *Monthly Food Price Estimates by Product and Market*, Kenya, version 2026-08-24: https://microdata.worldbank.org/catalog/6167

The source CSV is `KEN_RTFP_mkt_2007_2026-08-24.csv` (55,224 rows, 46 columns). **Important:** `c_maize_fao` is a model-derived price estimate; `maize_fao` contains relatively few directly observed prices. Forecast results must not be presented as validation against 55,224 independently measured market prices.

## Current status
Repository initialized with a reproducible data audit in `src/data_audit.py`. Modelling, API, Docker and report are **not yet completed**.

## Run the audit
1. Download the 2026-08-24 Kenya market CSV from the dataset source.
2. Place it at `data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv`.
3. Install Python 3.10+ and run `pip install -r requirements.txt`.
4. Run `python -m src.data_audit --input data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv`.
5. Read the generated `reports/data_audit.json`.

## Planned evaluation
Chronological training/validation/test splits, a last-price persistence baseline, and regression models evaluated using MAE and RMSE. Historical features will be shifted to avoid look-ahead leakage. The World Bank's model-estimated historical values may incorporate information unavailable in real time; this limitation requires explicit investigation before interpreting forecasting scores.

## Intended system
Raw CSV → validation → historical feature engineering → model training → temporal evaluation → serialized model → FastAPI → Docker.

See `AI_USE.md` for assistance disclosure. No accuracy or deployment claims are made before execution and testing.
