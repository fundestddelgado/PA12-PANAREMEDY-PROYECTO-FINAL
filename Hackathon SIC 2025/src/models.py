"""Modelos: ARIMA (statsmodels), Prophet, LSTM (simple Keras).
Cada función devuelve una dict con keys: 'model' y 'predict(steps)'.
"""
from typing import Tuple
import numpy as np
import pandas as pd


def fit_arima(series: pd.Series):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=(1, 1, 1)).fit()

    def predict(steps=12):
        pred = model.forecast(steps)
        return pd.Series(pred, index=pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W'))

    return {'model': model, 'predict': predict}


def fit_prophet(series: pd.Series):
    from prophet import Prophet

    df = series.reset_index()
    df.columns = ['ds', 'y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)

    def predict(steps=12):
        future = m.make_future_dataframe(periods=steps, freq='W')
        fut = m.predict(future)
        res = fut[['ds', 'yhat']].set_index('ds').yhat[-steps:]
        return pd.Series(res.values, index=pd.to_datetime(res.index))

    return {'model': m, 'predict': predict}


def fit_lstm(series: pd.Series, lookback=8, epochs=20, verbose=0):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    arr = series.values.astype(float)

    def make_xy(a):
        X, y = [], []
        for i in range(len(a) - lookback):
            X.append(a[i:i+lookback])
            y.append(a[i+lookback])
        return np.array(X)[..., np.newaxis], np.array(y)

    X, y = make_xy(arr)
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    if len(X) > 0:
        model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(steps=12):
        last = arr[-lookback:].tolist()
        preds = []
        for _ in range(steps):
            x = np.array(last[-lookback:])[np.newaxis, ..., np.newaxis]
            p = model.predict(x, verbose=0)[0, 0]
            preds.append(max(0, p))
            last.append(p)
        idx = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
        return pd.Series(np.round(preds).astype(int), index=idx)

    return {'model': model, 'predict': predict}
