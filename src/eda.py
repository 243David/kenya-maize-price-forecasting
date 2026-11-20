"""Produce figures and descriptive tables from the source data."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pipeline import RAW

def main():
    df=pd.read_csv(RAW,low_memory=False)
    df['date']=pd.to_datetime(df.price_date)
    figdir=Path('reports/figures');figdir.mkdir(parents=True,exist_ok=True)
    series=df.groupby('date').c_maize_fao.agg(['mean','median'])
    ax=series.plot(figsize=(10,4),title='Kenya market-level estimated maize prices, 2007–2026')
    ax.set(ylabel='Source dataset price units',xlabel='Month')
    plt.tight_layout();plt.savefig(figdir/'price_trend.png',dpi=160);plt.close()
    monthly=df.groupby(df.date.dt.month).c_maize_fao.mean()
    ax=monthly.plot.bar(figsize=(9,4),title='Average estimated maize price by calendar month')
    ax.set(xlabel='Month',ylabel='Mean estimated price')
    plt.tight_layout();plt.savefig(figdir/'seasonality.png',dpi=160);plt.close()
    market=df.groupby('mkt_name').c_maize_fao.mean().sort_values()
    selected=pd.concat([market.head(10),market.tail(10)])
    ax=selected.plot.barh(figsize=(9,7),title='Ten lowest and highest mean-price markets')
    ax.set(xlabel='Mean estimated maize price')
    plt.tight_layout();plt.savefig(figdir/'market_comparison.png',dpi=160);plt.close()
    df.groupby('mkt_name').agg(mean_price=('c_maize_fao','mean'),observed_prices=('maize_fao','count')).to_csv('reports/market_summary.csv')
    print('EDA figures and market summary saved in reports/')

if __name__=='__main__':main()
