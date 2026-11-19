# Forecasting Maize Prices Across Kenyan Markets Using Machine Learning

**DSA 8401 Applied Machine Learning — Individual Capstone**  
**RUHANAMIRINDI BULINDI David | 224688 | Strathmore University**

## Executive summary

This study develops a reproducible system for forecasting next-month maize prices across Kenyan markets. The source contains 55,224 monthly market records spanning January 2007 to August 2026. Crucially, the target (`c_maize_fao`) is a **model-estimated price series**, not a set of 55,224 direct market quotes; only 2,997 entries are non-missing in the direct observation field (`maize_fao`). A chronological evaluation compared persistence with Ridge regression, Random Forest and histogram gradient boosting. Persistence performed best on validation and achieved test MAE **1.5347** and RMSE **2.3291**. This is a useful negative finding: added model complexity did not improve validation accuracy.

## 1. Problem and motivation

Maize prices matter for households, farmers and food-market decisions in Kenya. The research question is whether past market-level price information can predict next month's **published estimated maize price** more accurately than carrying forward the current month's value. The unit of analysis is a market-month; the horizon is one month.

## 2. Data and quality

The dataset has 55,224 rows, 46 columns, 234 markets and dates from 2007-01 to 2026-08. No duplicate market/date pairs were found. Direct observed maize prices are sparse (2,997 non-missing); the estimated maize series is complete. The latter was selected for panel coverage, while acknowledging that it may be smoothed or retrospectively revised. The audit script records missingness and duplicate keys. Forecast labels are shifted one month ahead within each market; rows with inadequate history are excluded.

## 3. Exploratory analysis

The EDA script generates overall mean/median estimated-price trends, calendar-month average prices and comparisons of high- and low-price markets. Calendar-month averages pool different years and should not be interpreted as causal seasonal effects. Market differences may reflect both economic conditions and estimation methodology. The local downloadable project bundle includes the rendered figures.

## 4. Features and leakage prevention

Features include the current price, lags at 1, 2, 3, 6 and 12 months, rolling averages over 3, 6 and 12 months, target month/year and market identity. Each feature uses information dated no later than the current month. Training targets end in December 2022 (**41,886** rows); validation targets run January 2023–December 2024 (**5,616**); test targets cover January 2025–August 2026 (**4,680**). Chronological splitting avoids direct mixing of future labels into training. However, if the upstream World Bank estimates use hindsight, this split cannot eliminate that source-level risk.

## 5. Model choice and alternatives

Persistence is a demanding baseline for monthly prices because recent values are often informative. Ridge regression provides a regularized linear comparator. Random Forest and histogram gradient boosting test whether nonlinear relationships and interactions improve accuracy. Numerical features are imputed using training-fitted medians; market names are one-hot encoded with unknown-market handling. Deep sequence models such as LSTMs and transformers were excluded because their complexity was not justified before demonstrating improvements over strong simpler methods. Unsupervised clustering does not directly answer a supervised numeric forecasting question.

## 6. Results

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Persistence | **1.5558** | **2.5873** |
| Ridge | 1.7362 | 2.5896 |
| Random Forest | 2.7840 | 3.9042 |
| Histogram Gradient Boosting | 3.2021 | 4.0426 |

Selection used **validation MAE**, not test results. Persistence was the winner. On the held-out test period it achieved MAE **1.5347** and RMSE **2.3291**. MAE is average absolute forecast error in the dataset's price units; RMSE gives more weight to larger errors. The selected model's prediction rule is fully interpretable: next-month forecast equals current-month estimated price. SHAP analysis of this deterministic baseline would add no meaningful explanation.

## 7. End-to-end architecture and deployment

The scripted workflow comprises CSV audit, exploratory figures, lag/rolling feature creation, chronological model training, validation selection, saved artifacts and test metrics. The FastAPI service exposes `/health` and `/predict`, accepts user-supplied historical features and returns a predicted price. Docker and Compose files allow local container deployment. The API is not connected to live market feeds and is not claimed to be a publicly hosted production system.

## 8. Limitations and next steps

The central limitation is provenance: the target is a historical model-estimated series that may have been revised using later information. This analysis does not prove that forecasts would have been accurate in real time or against independently observed market prices. The experiment uses one chronological holdout, fixed hyperparameters and no rolling-origin backtesting. A stronger next study would evaluate timestamped observed prices, several forecast origins and forecast-time-available predictors such as rainfall, transport disruptions and exchange rates.

## Source

World Bank Microdata Library, Kenya monthly food-price estimates by product and market, extract dated 2026-08-24. Dataset supplied for the capstone. See `AI_USE.md` for the disclosure of AI contributions.
