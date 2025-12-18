"""Modelos estadísticos: Prophet y ARIMA, más métricas de evaluación."""
from typing import Tuple
import pandas as pd
import numpy as np


def entrenar_prophet(df_semanal: pd.DataFrame, semanas_a_predecir: int = 12) -> Tuple[pd.DataFrame, object]:
    from prophet import Prophet

    df = df_semanal.copy()
    # accept several common column name pairs: (date,sales), (fecha,ventas) or (ds,y)
    date_candidates = ['ds', 'date', 'fecha']
    sales_candidates = ['y', 'sales', 'ventas']

    date_col = None
    sales_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    for c in sales_candidates:
        if c in df.columns:
            sales_col = c
            break

    if date_col is None or sales_col is None:
        raise ValueError('Formato de DataFrame no reconocido: se requiere columna de fecha y ventas')

    if date_col != 'ds' or sales_col != 'y':
        df = df.rename(columns={date_col: 'ds', sales_col: 'y'})

    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df[['ds', 'y']].dropna()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=semanas_a_predecir, freq='W')
    forecast = m.predict(future)
    return forecast, m


def entrenar_arima(df_semanal: pd.DataFrame, pasos_futuros: int = 12) -> pd.Series:
    from statsmodels.tsa.arima.model import ARIMA

    df = df_semanal.copy()
    # detect common date/sales column names
    date_candidates = ['date', 'ds', 'fecha']
    sales_candidates = ['sales', 'y', 'ventas']

    date_col = None
    sales_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    for c in sales_candidates:
        if c in df.columns:
            sales_col = c
            break

    if date_col is None or sales_col is None:
        raise ValueError('Formato de DataFrame no reconocido')

    s = df.set_index(date_col)[sales_col]
    # ensure datetime index and weekly frequency
    s.index = pd.to_datetime(s.index, errors='coerce')
    s = s.asfreq('W')

    s = s.fillna(method='ffill').fillna(0)
    model = ARIMA(s, order=(5, 1, 0)).fit()
    pred = model.forecast(steps=pasos_futuros)
    pred.index = pd.date_range(s.index[-1] + pd.Timedelta(weeks=1), periods=pasos_futuros, freq='W')
    return pred


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray):
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    return {'MAE': float(mae), 'RMSE': float(rmse)}
