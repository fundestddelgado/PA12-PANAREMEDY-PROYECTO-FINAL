"""Módulo Data Manager: generación, limpieza y resampling de datos."""
from pathlib import Path
import pandas as pd
import numpy as np


def generar_dataset_simulado(filepath: str = 'data/sales_sample.csv', start='2020-01-01', days=365*2, sku='SKU_1'):
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return str(p)

    dates = pd.date_range(start=start, periods=days, freq='D')
    rng = np.random.default_rng(123)
    base = 30
    trend = np.linspace(0, 20, days)
    season = 10 * np.sin(2 * np.pi * (np.arange(days) / 365))
    noise = rng.normal(scale=base * 0.2, size=days)
    sales = np.maximum(0, base + trend + season + noise).round().astype(int)
    df = pd.DataFrame({'date': dates, 'sku': sku, 'sales': sales})
    df.to_csv(p, index=False)
    return str(p)


def cargar_y_limpiar_datos(filepath: str):
    # Read without forcing parse_dates to allow files with different date column names
    df = pd.read_csv(filepath, low_memory=False)

    # try common date column names first
    candidates = ['date', 'fecha', 'ds', 'datetime', 'timestamp', 'date_time', 'order_date', 'sale_date']
    date_col = None
    for c in candidates:
        if c in df.columns:
            date_col = c
            break

    # if not found, try to infer a column with datetime-like values
    if date_col is None:
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                # try parsing a sample
                parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                non_na_ratio = parsed.notna().mean()
                if non_na_ratio > 0.5:
                    date_col = c
                    df[c] = parsed
                    break

    # if we found a date column, convert it and sort
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        df = df.sort_values(date_col)
        # standardize name to 'date'
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
    else:
        # fallback: keep original order and add a date column as NaT
        df['date'] = pd.NaT

    if 'sales' in df.columns:
        if df['sales'].isna().any():
            # fill with 0 where too many missing, otherwise use mean
            if df['sales'].isna().sum() / len(df) > 0.2:
                df['sales'] = df['sales'].fillna(0)
            else:
                df['sales'] = df['sales'].fillna(int(df['sales'].mean()))
    return df


def resample_datos(df: pd.DataFrame, frecuencia: str = 'W') -> pd.DataFrame:
    # expects df with columns ['date','sku','sales']
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    if 'sku' in df.columns:
        res = df.set_index('date').groupby('sku')['sales'].resample(frecuencia).sum().reset_index()
    else:
        res = df.set_index('date')['sales'].resample(frecuencia).sum().reset_index()
    return res


def get_estadisticas_basicas(df: pd.DataFrame):
    # compute stats for sales column aggregated
    s = df['sales']
    return {
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': int(s.min()),
        'max': int(s.max())
    }
