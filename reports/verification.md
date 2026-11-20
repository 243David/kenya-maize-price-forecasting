# Verification record — updated 20 November 2026

## Local checks
- `python -m pytest -q`: **2 passed**.
- FastAPI `GET /health`: HTTP 200, `model_available: true`.
- FastAPI `POST /predict`: HTTP 200 for an illustrative market-price request; the selected persistence model returned 60.0 when `current_price` was 60.0.
- Presentation prepared as a 13-slide PowerPoint, supplied in the downloadable project bundle.

## Cloud deployment verified (20 November 2026)
- Render Docker Web Service deployment displayed **Deploy succeeded | Live** with duration 50.8 seconds.
- Public service URL: https://kenya-maize-price-forecasting.onrender.com
- Live `GET /health` returned HTTP 200 in Render logs.
- User executed `POST /predict` via Swagger UI: HTTP 200, market `Nairobi`, forecast December 2026, `current_price: 60`, `predicted_price: 60`, model `persistence`.
- User executed an invalid request with `current_price: -60`: HTTP 422, validation message `Input should be greater than 0`.
- These are functional deployment checks with illustrative input, not accuracy tests on real market observations.

## Not yet verified
- Docker image successfully built on Render; a separate local `docker compose up --build` was **not run** because Docker was unavailable in the local execution environment.
- GitHub Actions / fresh-environment installation were not run.
- The large raw CSV and serialized joblib artifact are included in the downloadable ZIP but are not committed to the public repository.
- The API is live on Render, but is a demonstration using the persistence rule from committed model metadata rather than a published serialized joblib artifact.

## Reproduction guidance
After cloning, download the World Bank CSV to `data/raw/`, install requirements plus `pytest httpx`, run `python -m src.data_audit`, `python -m src.eda`, `python -m src.pipeline`, and `python -m pytest -q`. Then launch `uvicorn api.main:app --reload` or build the Docker image locally.
