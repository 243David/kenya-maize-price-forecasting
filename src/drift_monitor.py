"""Offline feature-drift diagnostics for the advanced maize forecasting experiment.

Run: python -m src.drift_monitor
Reference: 2023–2024 forecast origins; monitoring: 2025 onward.
This is descriptive monitoring, not proof of model degradation or causal change.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.advanced_pipeline import FEATURES, RAW, prepare

OUT = Path('reports/advanced/drift')


def population_stability_index(reference, current, bins=10):
    """PSI with quantile bins learned exclusively from the reference sample.

    Return None when a comparison cannot be made (empty or constant reference).
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    epsilon = 1e-6
    return float(np.sum((pb - pa) * np.log((pb + epsilon) / (pa + epsilon))))


def drift_status(psi):
    if psi is None:
        return 'not_evaluable'
    if psi >= 0.25:
        return 'high'
    if psi >= 0.10:
        return 'moderate'
    return 'low'


def assess(reference, current, features=FEATURES):
    if reference.empty or current.empty:
        raise ValueError('Both reference and monitoring windows must contain data')
    rows = []
    for feature in features:
        if feature not in reference or feature not in current:
            raise KeyError(f'Missing feature: {feature}')
        a = pd.to_numeric(reference[feature], errors='coerce')
        b = pd.to_numeric(current[feature], errors='coerce')
        psi = population_stability_index(a, b)
        rows.append({'feature': feature, 'psi': round(psi, 5) if psi is not None else None,
                     'status': drift_status(psi),
                     'reference_mean': float(a.mean()) if a.notna().any() else None,
                     'monitor_mean': float(b.mean()) if b.notna().any() else None,
                     'reference_missing_rate': round(float(a.isna().mean()), 5),
                     'monitor_missing_rate': round(float(b.isna().mean()), 5)})
    return pd.DataFrame(rows).sort_values('psi', ascending=False, na_position='last')


def run(path=RAW, out=OUT):
    d = prepare(path)
    ref = d[(d.date >= '2023-01-01') & (d.date < '2025-01-01')]
    cur = d[d.date >= '2025-01-01']
    report = assess(ref, cur)
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / 'feature_drift.csv', index=False)
    summary = {
        'reference_period': '2023-01 through 2024-12',
        'monitor_period': '2025-01 onward',
        'reference_rows': int(len(ref)), 'monitor_rows': int(len(cur)),
        'method': 'Population Stability Index (PSI), reference-decile bins',
        'thresholds': {'moderate': 0.10, 'high': 0.25},
        'limitations': [
            'PSI measures distribution shift, not predictive performance or causality.',
            'Reference and monitoring periods may differ in market coverage.',
            'The estimated maize-price series may reflect upstream estimation changes.',
            'Thresholds are heuristic; alerting requires operational validation.'
        ],
        'high_features': report.loc[report.status == 'high', 'feature'].tolist(),
        'moderate_features': report.loc[report.status == 'moderate', 'feature'].tolist(),
        'feature_count': int(len(report))
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
