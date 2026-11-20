# Advanced research extension — Kenya maize-price forecasting

Run `python -m src.advanced_pipeline` from repository root after installing `requirements.txt` and placing the CSV at `data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv`.

## Implemented in this extension
- Direct 1-, 2-, 3-month market-level forecast evaluation
- Log-return targets; leakage-safe lagged and rolling features
- Chronological train (through 2022), validation (2023–2024), test (2025–Aug 2026)
- Persistence, ridge, histogram gradient boosting, LightGBM; selection by validation MAE
- Holdout-residual 80% intervals (empirical coverage is reported; not guaranteed)
- Month-block bootstrap intervals for MAE differences vs persistence
- Saved test predictions and machine-readable metrics

## Not yet implemented
SHAP, 32-feature parity, formal PSI drift monitoring, a multi-horizon production API, and cloud deployment. This is an honest research extension, not a claim that the example deck's results are replicated.

**Crucial data limitation:** `c_maize_fao` is a model-estimated series. Evaluation against it is not validation against independently collected market prices, and upstream hindsight leakage is possible. A stronger next phase requires real survey data and source-vintage validation.

## LightGBM validation MAE (2023–2024)
- 1-month: 1.7398 (persistence 1.5558)
- 2-month: 2.3995 (persistence 2.3085)
- 3-month: 3.2425 (persistence 3.0420)

LightGBM was not selected because persistence was more accurate on validation for all horizons. Four local tests passed.

## Version 2: feature drift monitoring

Run `python -m src.drift_monitor` from the repository root after placing the source CSV in `data/raw/`. Outputs: `reports/advanced/drift/feature_drift.csv` and `summary.json`.

Compares 2023–2024 forecast-origin features with 2025 onward using reference-decile Population Stability Index. Heuristic thresholds: PSI >= 0.10 moderate; PSI >= 0.25 high. This is an offline diagnostic, not continuous production monitoring. It does not measure forecasting accuracy. Test with `python -m pytest tests/test_drift_monitor.py -q`.
