#  PanaRemedy v.2.0 - Análisis de Medicamentos Avanzado con IA's
---
## PanaRemedy v2 — Visión general

PanaRemedy es una colección de utilidades y scripts en Python para limpiar, analizar y visualizar datos de medicamentos y farmacias. El proyecto incluye un pipeline de limpieza (ETL), scripts de análisis y modelos sencillos (regresión y clustering), y una interfaz gráfica (Tkinter) para explorar visualizaciones: histogramas, violines/cajas, mapas de calor, clustering y pronósticos.

### Planteamiento
- Muchas organizaciones tienen datos heterogéneos de precios y existencias (encodings distintos, delimitadores mixtos, columnas renombradas). Este proyecto plantea un flujo reproducible para transformar esos datos a un esquema canónico listo para análisis y ML.

### Objetivos
- Normalizar y limpiar datasets de medicamentos para análisis comparables.
- Proveer visualizaciones exploratorias y herramientas de diagnóstico de datos.
- Entregar scripts reproducibles de ML (regresión ligera y clustering) sobre el esquema canónico.

### Resultado del proyecto (resumen)
- Cleaner robusto que detecta encoding/delimitador y exporta CSV limpios y reportes JSON (`Proyecto/ml/cleaner.py`).
- Tabla canónica en formato "long" para ML (`long_monthly_all.csv` y `long_monthly_all_dedup.csv`).
- Visualizaciones reproducibles (histogramas, violin, heatmaps) en `Proyecto/viz` y `run_tests/output`.
- Interfaz (`Proyecto/interfaz.py`) que facilita generar y explorar todas las gráficas y ejecutar smoke-tests de ML.

### Tecnologías utilizadas
- Python 3.10+
- pandas, numpy, matplotlib, scikit-learn, pillow
- Tkinter para la GUI

### Estructura clave del repo
- `Proyecto/ml/` — cleaner, scripts ML y utilidades
- `Proyecto/viz/` — funciones de visualización reutilizables
- `Proyecto/data/cleaned/` — salida de limpieza (CSV listos)
- `Proyecto/interfaz.py` — GUI
- `run_tests/` — scripts de diagnóstico y generación de ejemplos

---
A continuación se incluyen instrucciones prácticas para ejecutar la interfaz y reproducir las visualizaciones (si ya conoces el proyecto puedes saltar a la sección "Ejecutar la interfaz").

## Ejecutar la interfaz (GUI) y asegurarse de que se muestren todas las visualizaciones

Estas instrucciones asumen que estás en Windows/PowerShell y que ya instalaste las dependencias con `requirements.txt`.

- 1) Preparar entorno (si aún no lo hiciste)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- 2) Generar/validar datos limpios

El GUI espera los datasets limpios en `Proyecto/data/cleaned/` (ej. `cleaned_data-v1.csv` y `long_monthly_all_dedup.csv`). Si esos archivos no están en el repositorio (o están en `.gitignore`), genera los datos con el cleaner:

```powershell
# ejecuta el cleaner para procesar los CSV de entrada
python Proyecto/ml/cleaner.py --input Proyecto/data/medicamentos.csv --output Proyecto/data/cleaned/

# (opcional) reconstruir la tabla 'long' y deduplicar
python Proyecto/ml/build_long.py
python Proyecto/ml/dedup_long.py
```

Si tu **equipo** no tiene los CSV de entrada, comparte `Proyecto/data/cleaned/*.csv` o indícales que ejecuten el cleaner localmente.

- 3) (Opcional) Ejecutar scripts ML/Smoke para preparar artefactos

```powershell
# crea los CSV mapeados para ML y ejecuta clustering/regresión de prueba
python Proyecto/ml/run_ml_smoke.py
```

- 4) Ejecutar la interfaz (GUI)

```powershell
python Proyecto/interfaz.py
```

La GUI incluye pestañas y botones para:
- Histograma: genera histogramas por precio.
- Violin/Caja: distribución por presentación/medicamento.
- Heatmap Precio vs Existencias: global o por medicamento (ingresa nombre parcial y pulsa el botón).
- Heatmap Rangos: matriz cruzada por rango de precio / rango de existencias.
- Predicción (regresión lineal y MA): calcula pronóstico y muestra gráfico.
- Clustering: ejecuta el clustering sobre las columnas mapeadas y muestra resultados básicos.

Problemas comunes y soluciones
- Si ves warnings de pandas como "SettingWithCopyWarning": actualizar `Proyecto/ml/heatmap.py` ya corregido para usar copias y `.loc`.
- Si la GUI muestra "Módulo heatmap no disponible": asegúrate de ejecutar desde la raíz del repo (`PanaRemedy`) para que `Proyecto` sea importable, o ejecutar con `python -m Proyecto.interfaz`.
- Si faltan datos o las gráficas aparecen vacías: revisa que `Proyecto/data/cleaned/long_monthly_all_dedup.csv` exista y tenga columnas esperadas (`CODIGO`, `DESCRIPCION_LIMPIA`, `precio_unitario`, `total_existencias`, `monto_existencias`, `snapshot_date`).
- Para depurar errores del heatmap desde la GUI: el handler guarda trazas en `run_tests/heatmap_error.log` cuando ocurre una excepción.

Compartir con el equipo
- Recomendado: subir `Proyecto/data/cleaned/*.csv` y `run_tests/` al repo si no contienen datos sensibles.
- Si prefieres no subir datos, comparte `run_tests/` y añade instrucciones a tus compañeros para ejecutar el `cleaner.py` y generar los CSV antes de ejecutar la GUI.

Comandos útiles (PowerShell)
```powershell
# comprobar si archivo está trackeado
git ls-files --error-unmatch run_tests/inspect_paracetamol.py

# subir run_tests al repo
git add run_tests
git commit -m "test: add run_tests diagnostics and plotting utilities"
git push origin main

# ignorar un archivo específico localmente
git rm --cached run_tests/inspect_paracetamol.py
Add-Content -Path .gitignore -Value 'run_tests/inspect_paracetamol.py'
git add .gitignore
git commit -m "chore: ignore diagnostic run_tests/inspect_paracetamol.py"
git push origin main
```

---
Si quieres, puedo:
- añadir un README más detallado en `run_tests/README.md` con ejemplos de salida; o
- ejecutar la GUI aquí y verificar junto a ti los botones (heatmap, hist, clustering). ¿Cuál prefieres?
