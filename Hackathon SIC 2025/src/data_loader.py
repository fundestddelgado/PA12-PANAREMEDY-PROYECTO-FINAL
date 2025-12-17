"""Carga y preprocesamiento minimal para series semanales por SKU."""
from pathlib import Path
import pandas as pd


def load_sales(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    # aggregate weekly per sku
    weekly = df.groupby(['sku', 'week'])['sales'].sum().reset_index()
    weekly = weekly.sort_values(['sku', 'week'])
    return weekly


def get_series_for_sku(weekly_df, sku):
    s = weekly_df[weekly_df['sku'] == sku].set_index('week')['sales'].asfreq('W-MON')
    s.index = pd.to_datetime(s.index)
    return s
