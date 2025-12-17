"""Modelos Deep Learning: LSTM pipeline (escalado, secuencias, entrenamiento y predicción)."""
import numpy as np
import pandas as pd


def escalar_datos(dataset: np.ndarray):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = np.array(dataset).reshape(-1, 1).astype(float)
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def crear_secuencias(dataset: np.ndarray, look_back: int = 4):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        y.append(dataset[i + look_back])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def construir_y_entrenar_lstm(X_train, y_train, epochs: int = 25, verbose: int = 0):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        layers.LSTM(32, activation='tanh'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)
    return model


def predecir_lstm(modelo, ultimos_datos, scaler, pasos=12):
    # ultimos_datos: array shape (look_back,)
    preds = []
    seq = list(ultimos_datos)
    for _ in range(pasos):
        x = np.array(seq[-len(ultimos_datos):]).reshape(1, len(ultimos_datos), 1)
        p = modelo.predict(x, verbose=0)[0, 0]
        preds.append(p)
        seq.append(p)
    arr = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(arr)
    return inv.flatten()
