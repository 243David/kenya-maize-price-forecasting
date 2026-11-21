# Advanced V2 API (isolated)

This is a separate experimental service. It does **not** replace the submitted `api.main` or original Render service.

## Run locally

```bash
pip install -r requirements-api-v2.txt
uvicorn api_v2.main:app --reload --port 8001
```

Open http://127.0.0.1:8001/docs. Example:

```bash
curl -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' -d '{"market":"Example market","current_price":60,"origin_year":2026,"origin_month":11,"horizon":2}'
```

## Docker

```bash
docker build -f Dockerfile.v2 -t maize-advanced-v2 .
docker run --rm -p 8001:8000 maize-advanced-v2
```

The API uses the **validated winner (persistence)** for all horizons. The evaluated LightGBM models remain research comparators, and are not falsely advertised as deployed. The intervals use absolute validation residual quantiles and have been evaluated on the held-out test period. The target is the World Bank's **estimated** `c_maize_fao` series, not directly observed market prices. The JSON metrics file is mandatory: if missing, the service returns HTTP 503 instead of fabricating estimates.

For a new Render service, use `Dockerfile.v2` and a separate service name. Do not change the original Render service settings.
