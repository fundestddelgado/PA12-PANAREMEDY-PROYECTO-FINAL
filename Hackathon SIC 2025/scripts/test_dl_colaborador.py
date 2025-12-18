"""Script de prueba ligero para `dl_models_colaborador.py`.
Ejecuta escalado y creación de secuencias (no requiere TensorFlow para esta prueba).
"""
import os
import sys
from importlib.machinery import SourceFileLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_path = os.path.join(repo_root, 'src', 'dl_models_colaborador.py')

dl = SourceFileLoader('dl_models_colaborador', module_path).load_module()

import numpy as np

ventas = np.array([100, 120, 130, 140, 150, 160, 170, 180])
print('Ventas:', ventas)

ventas_escaladas, scaler = dl.escalar_datos(ventas)
print('Ventas escaladas shape:', ventas_escaladas.shape)

X, y = dl.crear_secuencias(ventas_escaladas, look_back=4)
print('X shape:', X.shape)
print('y shape:', y.shape)

print('Prueba ligera completada exitosamente')
