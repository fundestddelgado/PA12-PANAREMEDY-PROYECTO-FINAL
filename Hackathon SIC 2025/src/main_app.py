"""Aplicación Streamlit que integra los módulos y presenta dashboard de inventario."""
# Ensure project root is on sys.path so `src` package imports work when running
# the script via `streamlit run src/main_app.py` (Streamlit may change import context).
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_manager import generar_dataset_simulado, cargar_y_limpiar_datos, resample_datos, get_estadisticas_basicas
from src.stat_models import entrenar_prophet, entrenar_arima, calcular_metricas
from src.dl_models import escalar_datos, crear_secuencias, construir_y_entrenar_lstm, predecir_lstm
from src.inventory_logic import calcular_punto_reorden, calcular_cantidad_a_pedir
import numpy as np


def main():
    st.set_page_config(page_title='Inventory Suggestion App', layout='wide')
    st.title('Motor de Sugerencias - Optimización de Inventario')

    st.sidebar.header('Datos')
    uploaded = st.sidebar.file_uploader('CSV (date,sku,sales)', type=['csv'])
    default_file = 'data/sales_sample.csv'
    if uploaded is None:
        generar_dataset_simulado(default_file)
        df = cargar_y_limpiar_datos(default_file)
        st.sidebar.info('Usando dataset simulado: data/sales_sample.csv')
    else:
        df = cargar_y_limpiar_datos(uploaded)

    # detect sku-like column; if missing, create a default 'sku' column
    sku_col_candidates = ['sku', 'product', 'product_id', 'item', 'id']
    sku_col = next((c for c in sku_col_candidates if c in df.columns), None)
    if sku_col is None:
        df['sku'] = 'ALL'
        sku_col = 'sku'

    skus = df[sku_col].unique().tolist()
    sku = st.sidebar.selectbox('SKU', skus)
    freq = st.sidebar.selectbox('Frecuencia', ['W', 'M'], index=0)
    semanas_pred = st.sidebar.slider('Semanas a predecir', 4, 52, 12)
    lead_time = st.sidebar.number_input('Lead time (semanas)', min_value=1, max_value=12, value=2)
    objetivo_max = st.sidebar.number_input('Objetivo máximo stock', min_value=1, value=200)

    df_sku = df[df[sku_col] == sku]
    res = resample_datos(df_sku, frecuencia=freq)
    # normalize columns: ensure we have 'date' (datetime) and 'sales'
    res = res.reset_index(drop=True)
    # if res has a datetime-typed column already, use it
    date_col = None
    for c in res.columns:
        if pd.api.types.is_datetime64_any_dtype(res[c]):
            date_col = c
            break
    if date_col is None:
        # try to find a column that can be parsed as datetime
        for c in res.columns:
            if c == 'sku':
                continue
            try:
                parsed = pd.to_datetime(res[c])
                res[c] = parsed
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        # fallback: create date from index if possible
        try:
            res['date'] = pd.to_datetime(res.index)
            date_col = 'date'
        except Exception:
            res['date'] = pd.NaT
            date_col = 'date'

    # rename detected date column to 'date' and last column to 'sales' if needed
    if date_col != 'date':
        res = res.rename(columns={date_col: 'date'})
    if 'sales' not in res.columns:
        res = res.rename(columns={res.columns[-1]: 'sales'})

    # ensure unique column names (append suffix for duplicates) to avoid duplicate-key errors
    new_cols = []
    counts = {}
    for c in res.columns:
        counts[c] = counts.get(c, 0) + 1
        if counts[c] == 1:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}_{counts[c]}")
    res.columns = new_cols

    # pick the first available date and sales columns
    date_col = next((c for c in res.columns if c.startswith('date')), None)
    sales_col = next((c for c in res.columns if c.startswith('sales')), None)
    if date_col is None:
        # fallback to first non-sku column
        date_col = next((c for c in res.columns if c != 'sku'), res.columns[0])
    if sales_col is None:
        sales_col = res.columns[-1]

    st.subheader('Estadísticas básicas')
    stats = get_estadisticas_basicas(res)
    st.write(stats)

    st.subheader('Serie histórica')
    fig = go.Figure()
    # use the selected date and sales columns (safe against duplicate names)
    date_series = pd.to_datetime(res[date_col], errors='coerce')
    fig.add_trace(go.Scatter(x=date_series, y=res[sales_col], name='histórico'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Predecir con Prophet'):
            # preparar df semanal en formato ds,y
            df_prop = res.rename(columns={'date': 'ds', 'sales': 'y'})[['ds', 'y']]
            forecast, model = entrenar_prophet(df_prop, semanas_a_predecir=semanas_pred)
            pred = forecast[['ds', 'yhat']].set_index('ds').yhat[-semanas_pred:]
            fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name='prophet'))
            st.plotly_chart(fig, use_container_width=True)

            # inventory logic
            mean_pred = float(pred.mean())
            stock_seg = float(pred.std()) * 1.65
            punto = calcular_punto_reorden(mean_pred, lead_time, stock_seg)
            stock_actual = int(st.number_input('Stock actual', min_value=0, value=50, key='prophet_stock'))
            qty = calcular_cantidad_a_pedir(stock_actual, punto, objetivo_max)
            color = '✅' if qty == 0 else '🔴'
            st.markdown(f'**Punto de reorden:** {punto:.0f} — {color} **Sugerencia:** {qty} unidades')

        # ARIMA button
        if st.button('Predecir con ARIMA'):
            # preparar df semanal con columnas date/sales
            df_arima = res.rename(columns={'date': 'date', 'sales': 'sales'})[['date', 'sales']]
            try:
                pred_arima = entrenar_arima(df_arima, pasos_futuros=semanas_pred, frecuencia=freq)
                # add to plot
                fig.add_trace(go.Scatter(x=pred_arima.index, y=pred_arima.values, name='arima'))
                st.plotly_chart(fig, use_container_width=True)

                mean_pred = float(pred_arima.mean())
                stock_seg = float(pred_arima.std()) * 1.65
                punto = calcular_punto_reorden(mean_pred, lead_time, stock_seg)
                stock_actual = int(st.number_input('Stock actual', min_value=0, value=50, key='arima_stock'))
                qty = calcular_cantidad_a_pedir(stock_actual, punto, objetivo_max)
                color = '✅' if qty == 0 else '🔴'
                st.markdown(f'**Punto de reorden:** {punto:.0f} — {color} **Sugerencia:** {qty} unidades')
            except Exception as e:
                st.error(f'ARIMA falló: {e}')

    with col2:
        if st.button('Predecir con LSTM'):
            # use selected frecuencia (W or M)
            series = res.set_index('date')['sales'].asfreq(freq).fillna(method='ffill')
            scaled, scaler = escalar_datos(series.values)
            look_back = 8
            X, y = crear_secuencias(scaled, look_back=look_back)
            if len(X) == 0:
                st.error('No hay suficientes datos para entrenar LSTM')
            else:
                model = construir_y_entrenar_lstm(X, y, epochs=20, verbose=0)
                ult = scaled[-look_back:].flatten()
                preds = predecir_lstm(model, ult, scaler, pasos=semanas_pred)
                # build future index using selected frecuencia
                if freq == 'W':
                    start = pd.to_datetime(series.index[-1]) + pd.Timedelta(weeks=1)
                else:
                    # for monthly, add one month
                    start = pd.to_datetime(series.index[-1]) + pd.offsets.MonthBegin(1)
                idx = pd.date_range(start, periods=semanas_pred, freq=freq)
                fig.add_trace(go.Scatter(x=idx, y=preds, name='lstm'))
                st.plotly_chart(fig, use_container_width=True)

                mean_pred = float(np.mean(preds))
                stock_seg = float(np.std(preds)) * 1.65
                punto = calcular_punto_reorden(mean_pred, lead_time, stock_seg)
                stock_actual = int(st.number_input('Stock actual', min_value=0, value=50, key='lstm_stock'))
                qty = calcular_cantidad_a_pedir(stock_actual, punto, objetivo_max)
                color = '✅' if qty == 0 else '🔴'
                st.markdown(f'**Punto de reorden:** {punto:.0f} — {color} **Sugerencia:** {qty} unidades')


if __name__ == '__main__':
    main()
