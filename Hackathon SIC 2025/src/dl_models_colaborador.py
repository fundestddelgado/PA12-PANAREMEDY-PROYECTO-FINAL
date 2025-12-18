import numpy as np


"""Funciones reutilizables del compañero: escalado, secuencias, entrenamiento y predicción LSTM.
Este módulo evita importar TensorFlow/keras al nivel del módulo (importaciones internas) para permitir pruebas ligeras sin TF.
"""


def escalar_datos(dataset):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = np.array(dataset).reshape(-1, 1).astype(float)
    dataset_escalado = scaler.fit_transform(dataset)
    return dataset_escalado, scaler


def crear_secuencias(dataset, look_back=4):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


def construir_y_entrenar_lstm(X_train, y_train, epochs=50, batch_size=16, verbose=0):
    # Importar TF/Keras dentro de la función para evitar requisitos al importar el módulo
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model


def predecir_lstm(modelo, ultimos_datos, scaler):
    ultimos_datos = np.array(ultimos_datos).reshape(1, len(ultimos_datos), 1)
    prediccion_escalada = modelo.predict(ultimos_datos, verbose=0)
    prediccion_real = scaler.inverse_transform(prediccion_escalada)
    return float(prediccion_real[0][0])
