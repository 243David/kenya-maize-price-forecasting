# Verification record — 19 November 2026

## Local checks
- `python -m pytest -q`: **2 passed**.
- FastAPI `GET /health`: HTTP 200, `model_available: true`.
- FastAPI `POST /predict`: HTTP 200 for an illustrative market-price request; the selected persistence model returned 60.0 when `current_price` was 60.0.
- Presentation prepared as a 13-slide PowerPoint, supplied in the downloadable project bundle.

## Not yet verified
- Docker image build and `docker compose up --build` were **not run**, because Docker is not installed in the execution environment.
- GitHub Actions / fresh-environment installation were not run.
- The large raw CSV and serialized joblib artifact are included in the downloadable ZIP but are not committed to the public repository.
- The API is a local demonstration, not a live hosted service.

## Reproduction guidance
After cloning, download the World Bank CSV to `data/raw/`, install requirements plus `pytest httpx`, run `python -m src.data_audit`, `python -m src.eda`, `python -m src.pipeline`, and `python -m pytest -q`. Then launch `uvicorn api.main:app --reload` or build the Docker image locally.
