"""Generador de ventas semanales sintéticas para múltiples SKUs.
Guarda un CSV con columnas: date, sku, sales
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def generate_sales(start_date='2020-01-01', weeks=260, n_skus=5, out='data/sales_sample.csv'):
    dates = pd.date_range(start=start_date, periods=weeks, freq='W')
    rows = []
    rng = np.random.default_rng(42)
    for sku in range(1, n_skus + 1):
        base = rng.integers(20, 100)
        trend = np.linspace(0, rng.integers(0, 50), weeks)
        season = 10 * np.sin(2 * np.pi * (np.arange(weeks) / 52) + sku)
        noise = rng.normal(scale=base * 0.15, size=weeks)
        sales = np.maximum(0, base + trend + season + noise).round().astype(int)
        for d, s in zip(dates, sales):
            rows.append({'date': d.date(), 'sku': f'SKU_{sku}', 'sales': int(s)})

    df = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Generated {len(df)} rows to {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/sales_sample.csv')
    p.add_argument('--weeks', type=int, default=260)
    p.add_argument('--n_skus', type=int, default=5)
    args = p.parse_args()
    generate_sales(weeks=args.weeks, n_skus=args.n_skus, out=args.out)


if __name__ == '__main__':
    main()
