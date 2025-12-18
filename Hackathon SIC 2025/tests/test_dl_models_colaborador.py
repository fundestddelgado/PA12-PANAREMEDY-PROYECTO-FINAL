import numpy as np
from src.dl_models_colaborador import escalar_datos, crear_secuencias


def test_escalar_y_secuencias():
    ventas = np.array([100, 120, 130, 140, 150, 160, 170, 180])
    escaladas, scaler = escalar_datos(ventas)
    assert escaladas.shape == (8, 1)
    X, y = crear_secuencias(escaladas, look_back=4)
    # crear_secuencias expected to return X shape (n_samples, look_back) and y shape (n_samples,)
    assert X.shape[1] == 4
    assert X.shape[0] == y.shape[0]
    # valores escalados deben estar entre 0 y 1
    assert np.nanmin(escaladas) >= 0.0
    assert np.nanmax(escaladas) <= 1.0
