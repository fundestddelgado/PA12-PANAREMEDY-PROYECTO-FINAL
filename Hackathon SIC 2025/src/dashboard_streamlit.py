"""App Streamlit simple para mostrar pronósticos y sugerencia de pedido."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import load_sales, get_series_for_sku
from src import models as mdl
from src.reorder import safety_stock, reorder_point, suggest_order


st.set_page_config(layout='wide', page_title='Sugerencias Inventario')

st.title('Motor de Sugerencias - Optimización de Inventario')

uploaded = st.file_uploader('Carga CSV de ventas (date,sku,sales)', type=['csv'])
if uploaded is None:
    st.info('Carga un CSV o ejecuta `python data/generate_sales.py` para generar uno de ejemplo.')
    st.stop()

weekly = load_sales(uploaded)
skus = weekly['sku'].unique().tolist()
sku = st.selectbox('Seleccione SKU', skus)
model_choice = st.selectbox('Modelo', ['Prophet', 'ARIMA', 'LSTM'])
steps = st.number_input('Semanas a predecir', min_value=4, max_value=52, value=12)

series = get_series_for_sku(weekly, sku)
st.subheader(f'Serie histórica: {sku}')
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name='histórico'))

if st.button('Generar pronóstico'):
    if model_choice == 'Prophet':
        m = mdl.fit_prophet(series)
    elif model_choice == 'ARIMA':
        m = mdl.fit_arima(series)
    else:
        m = mdl.fit_lstm(series, epochs=30)

    pred = m['predict'](steps=steps)
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name='forecast'))
    st.plotly_chart(fig, use_container_width=True)

    # simple reorder suggestion: forecasted weekly mean and std
    mean_forecast = pred.mean()
    std_forecast = pred.std()
    lead = st.number_input('Lead time (semanas)', min_value=1, max_value=12, value=2)
    z = st.number_input('Z-score (nivel servicio)', min_value=0.0, max_value=3.0, value=1.65)

    rp = reorder_point(mean_forecast, std_forecast, lead_time_weeks=lead, z_score=z)
    st.markdown(f'**Reorder point (units):** {rp:.0f}')

    stock = st.number_input('Stock actual', min_value=0, value=50)
    order_qty = suggest_order(stock, rp, mean_forecast * lead)
    st.markdown(f'**Sugerencia de pedido:** {order_qty} unidades')
