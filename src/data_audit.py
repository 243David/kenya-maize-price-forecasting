"""Reproducible audit of the World Bank Kenya market-price CSV.

Usage:
    python -m src.data_audit --input data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"mkt_name", "price_date", "maize_fao", "c_maize_fao"}


def audit(input_path: Path) -> dict:
    data = pd.read_csv(input_path, low_memory=False)
    missing_columns = sorted(REQUIRED.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dates = pd.to_datetime(data["price_date"], errors="coerce")
    observed = pd.to_numeric(data["maize_fao"], errors="coerce")
    estimated = pd.to_numeric(data["c_maize_fao"], errors="coerce")
    duplicated = data.assign(_date=dates).duplicated(["mkt_name", "_date"]).sum()
    by_market = (
        data.assign(_date=dates)
        .groupby("mkt_name", dropna=False)["_date"]
        .nunique()
    )

    return {
        "source_file": input_path.name,
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "market_count": int(data["mkt_name"].nunique(dropna=True)),
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_market_months": int(duplicated),
        "observed_maize_count": int(observed.notna().sum()),
        "estimated_maize_count": int(estimated.notna().sum()),
        "observed_maize_markets": int(data.loc[observed.notna(), "mkt_name"].nunique()),
        "min_months_per_market": int(by_market.min()),
        "max_months_per_market": int(by_market.max()),
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "note": "c_maize_fao contains model-generated estimates, not independent observed prices.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    result = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "missing_by_column"}, indent=2))
    print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
