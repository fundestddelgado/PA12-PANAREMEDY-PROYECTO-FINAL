import pandas as pd
from stat_models import entrenar_prophet, entrenar_arima, calcular_metricas

# Dataset de prueba
data = {
    "fecha": pd.date_range(start="2023-01-01", periods=20, freq="W"),
    "ventas": [100, 110, 120, 130, 125, 140, 150, 160, 155, 165,170, 180, 175, 185, 190, 200, 195, 205, 210, 220]
}

df = pd.DataFrame(data)

# Prophet
forecast, model = entrenar_prophet(df, semanas_a_predecir=4)
print("Prophet OK")

# ARIMA
pred_arima = entrenar_arima(df, pasos_futuros=4)
print("ARIMA OK:", pred_arima)

# Métricas
y_real = df['ventas'][-4:]
y_pred = pred_arima[:4]
metricas = calcular_metricas(y_real, y_pred)

print("Métricas:", metricas)
