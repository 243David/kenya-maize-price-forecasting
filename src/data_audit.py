"""Inspect completeness and provenance of maize-price columns."""
import json
from pathlib import Path
import pandas as pd

def audit(path):
    df=pd.read_csv(path,low_memory=False)
    date=pd.to_datetime(df.price_date)
    return {'rows':len(df),'columns':len(df.columns),'markets':df.mkt_name.nunique(),
            'first_date':str(date.min().date()),'last_date':str(date.max().date()),
            'duplicate_market_months':int(df.duplicated(['mkt_name','price_date']).sum()),
            'direct_maize_observations':int(df.maize_fao.notna().sum()),
            'estimated_maize_prices':int(df.c_maize_fao.notna().sum()),
            'missing_by_column':df.isna().sum().astype(int).to_dict()}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('data/raw/KEN_RTFP_mkt_2007_2026-08-24.csv'))
    args=parser.parse_args()
    result=audit(args.input)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='missing_by_column'},indent=2))
