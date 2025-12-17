
Motor de Sugerencias para Optimización de Inventario

Proyecto 12 — Motor de sugerencias para optimización de inventario en tiendas pequeñas.

Descripción
Muchos negocios pequeños no saben cuánto stock pedir. Este proyecto genera pronósticos semanales de demanda y sugiere niveles de reorden para evitar quiebres o exceso.

Contenido
- `data/generate_sales.py`: script para generar ventas sintéticas (CSV semanal).
- `requirements.txt`: dependencias necesarias.
- `src/data_manager.py`: funciones de generación, carga y preprocesamiento.
- `src/stat_models.py`: implementaciones Prophet y ARIMA.
- `src/dl_models.py`: LSTM (escalado, secuencias, entrenamiento y predicción).
- `src/inventory_logic.py`: cálculo de punto de reorden y cantidad a pedir.
- `src/main_app.py`: app Streamlit para visualizar pronósticos y sugerencias.
- `scripts/run_compare.py`: script para comparar modelos y calcular métricas.
- `notebooks/compare_models.ipynb`: notebook interactivo con comparación.

Instalación (recomendado en virtualenv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notas sobre dependencias (Windows)
- `prophet` puede requerir compilación adicional en Windows; si hay problemas, use WSL o instale con `conda`:

```powershell
conda install -c conda-forge prophet
```

- `tensorflow` puede instalarse con `pip install tensorflow` en Windows 64-bit; para máquinas con GPU consulte la guía oficial.

Comandos útiles

```powershell
# Generar dataset de ejemplo
python data/generate_sales.py --out data/sales_sample.csv

# Ejecutar el dashboard Streamlit
streamlit run src/main_app.py

# Ejecutar comparación automática (script)
python scripts/run_compare.py --csv data/sales_sample.csv --sku SKU_1 --weeks 12
```

Notebook de comparación

Se incluye `notebooks/compare_models.ipynb` con un flujo interactivo que carga los datos, entrena Prophet / ARIMA / LSTM y calcula MAE/RMSE. Ejecute con Jupyter Lab/Notebook:

```powershell
jupyter lab
```

Interpretación rápida
- Si la predicción muestra alta varianza, ajuste stock de seguridad o reduzca el horizonte de pedido.
- Compare MAE/RMSE entre modelos y prefiera el modelo con menor error y que sea estable en picos estacionales.
