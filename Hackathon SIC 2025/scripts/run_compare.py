"""Script para comparar Prophet, ARIMA y LSTM en un SKU específico.
Ejemplo:
  python scripts/run_compare.py --csv data/sales_sample.csv --sku SKU_1 --weeks 12
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.data_manager import cargar_y_limpiar_datos, resample_datos
from src.stat_models import entrenar_prophet, entrenar_arima, calcular_metricas
from src.dl_models import escalar_datos, crear_secuencias, construir_y_entrenar_lstm, predecir_lstm
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--sku', required=True)
    p.add_argument('--weeks', type=int, default=12)
    args = p.parse_args()

    df = cargar_y_limpiar_datos(args.csv)
    df_sku = df[df['sku'] == args.sku]
    res = resample_datos(df_sku, frecuencia='W')
    # ensure flat columns (resample may return MultiIndex); reset_index to get date column
    res = res.reset_index()
    # normalize column names: find date and sales
    # after reset_index possible columns: ['sku','date','sales'] or ['date','sales']
    if 'date' not in res.columns:
        # try to detect datetime column
        for c in res.columns:
            if pd.api.types.is_datetime64_any_dtype(res[c]):
                res = res.rename(columns={c: 'date'})
                break
    if 'sales' not in res.columns:
        res = res.rename(columns={res.columns[-1]: 'sales'})

    # train/test split: last N weeks as test
    N = args.weeks
    series = res.set_index('date')['sales'].asfreq('W').fillna(method='ffill')
    train = series[:-N]
    test = series[-N:]

    print('Training Prophet...')
    df_train = train.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    try:
        forecast, m = entrenar_prophet(df_train, semanas_a_predecir=N)
        pred_prophet = forecast[['ds', 'yhat']].set_index('ds').yhat[-N:]
    except Exception as e:
        print('Prophet not available or failed:', e)
        pred_prophet = pd.Series([train.mean()] * N, index=test.index)

    print('Training ARIMA...')
    try:
        pred_arima = entrenar_arima(train.reset_index().rename(columns={'date':'date','sales':'sales'}), pasos_futuros=N)
    except Exception as e:
        print('ARIMA not available or failed:', e)
        pred_arima = pd.Series([train.mean()] * N, index=test.index)

    print('Training LSTM...')
    try:
        scaled, scaler = escalar_datos(train.values)
        look_back = 8
        X, y = crear_secuencias(scaled, look_back=look_back)
        if len(X) > 0:
            model = construir_y_entrenar_lstm(X, y, epochs=20, verbose=0)
            ult = scaled[-look_back:].flatten()
            pred_lstm = predecir_lstm(model, ult, scaler, pasos=N)
            pred_lstm = pd.Series(pred_lstm, index=test.index)
        else:
            pred_lstm = pd.Series([train.mean()]*N, index=test.index)
    except Exception as e:
        print('LSTM not available or failed:', e)
        pred_lstm = pd.Series([train.mean()] * N, index=test.index)

    # metrics
    m_prop = calcular_metricas(test.values, pred_prophet.values)
    m_arima = calcular_metricas(test.values, pred_arima.values)
    m_lstm = calcular_metricas(test.values, pred_lstm.values)

    print('Metrics:')
    print('Prophet:', m_prop)
    print('ARIMA:', m_arima)
    print('LSTM:', m_lstm)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train.values, label='train')
    plt.plot(test.index, test.values, label='test')
    plt.plot(pred_prophet.index, pred_prophet.values, label='prophet')
    plt.plot(pred_arima.index, pred_arima.values, label='arima')
    plt.plot(pred_lstm.index, pred_lstm.values, label='lstm')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print('Saved model_comparison.png')


if __name__ == '__main__':
    main()
