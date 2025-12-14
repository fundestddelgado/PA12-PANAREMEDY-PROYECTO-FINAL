# Dataset limpio (cleaned_data-v1.csv)

Resumen:

Cómo regenerar:
1. Asegúrate de tener `Proyecto/data/medicamentos.csv` en la carpeta `Proyecto/data/`.
2. Activa el virtualenv y ejecuta:

```powershell
& "C:/Users/ianar/Documents/SIC Samsung/PanaRemedy/venv/Scripts/Activate.ps1"
python Proyecto/ml/export_cleaned.py
```

Notas:
# Dataset notes
Este directorio contiene los datos de inventario y los archivos "limpios" usados por los scripts de análisis y por los pipelines de ML.

**Propósito**: mantener una copia reproducible del dataset canónico para la entrega (Módulo 1) y una zona `cleaned/` con versiones procesadas por mes para alimentar los pasos posteriores (melt, features, entrenamientos).

**Ubicaciones clave**:
- `Proyecto/data/medicamentos.csv` — archivo original (enero 2024) ya procesado y versionado.
- `Proyecto/data/cleaned/` — carpeta con archivos limpiados por mes y artefactos de ingestión.
	- `cleaned_febrero_2024.csv` — resultado de limpieza del CSV de febrero.
	- `cleaned_marzo_2024.csv` — resultado de limpieza del CSV de marzo.
	- `cleaned_data-v1.csv` — dataset canónico reproducible (commitado para entrega).

Recomendación de flujo y convenciones
- Los archivos raw mensuales se dejan en `Proyecto/data/` sin commitear cuando provienen de inspección local (evitar subir datos sensibles o pesados). Solo los exportados reproducibles se guardan en `Proyecto/data/cleaned/` y (si es necesario) se commitean según política del proyecto.
- Nombres: `cleaned_<mes>_<año>.csv` (por ejemplo `cleaned_febrero_2024.csv`). Para el dataset consolidado use `cleaned_data-vN.csv` con número de versión.
- Codificación: preferir `latin-1`/`cp1252` para lectura de estos CSVs; normalizar a `utf-8` en los archivos `cleaned/*.csv`.

Schema esperado (archivo cleaned por mes)
- CODIGO: string (código interno)
- DESCRIPCION: string (descripción original)
- PRECIO UNITARIO B/.: float (precio unitario normalizado)
- EXISTENCIA ALMACEN CDPA <MES AÑO>: int/float
- EXISTENCIA ALMACEN CDDI <MES AÑO>: int/float
- EXISTENCIA ALMACEN CDCH <MES AÑO>: int/float
- TOTAL DE EXISTENCIAS DISPONIBLES <MES AÑO>: int/float
- MONTO DE EXISTENCIAS EN B/.: float
- DESCRIPCION_LIMPIA: string (versión normalizada/upper-case para join)

Long-format (recomendado para ML/time-series)
- Archivo: `Proyecto/data/cleaned/long_monthly_YYYY.csv` o `long_monthly_all.csv` para consolidado.
- Columnas sugeridas:
	- `CODIGO` (str)
	- `DESCRIPCION_LIMPIA` (str)
	- `snapshot_date` (date) — usar primer día del mes, p.ej. `2024-02-01`.
	- `precio_unitario` (float)
	- `existencia_cdpa` (int)
	- `existencia_cddi` (int)
	- `existencia_cdch` (int)
	- `total_existencias` (int)
	- `monto_existencias` (float)

Ingestión y limpieza (scripts disponibles)
- `Proyecto/ml/clean_feb.py` — limpia el CSV de febrero y produce `cleaned_febrero_2024.csv` (usa `latin-1` y normaliza numerics).
- `Proyecto/ml/clean_marzo.py` — limpia el CSV de marzo (detecta `sep=';'`) y produce `cleaned_marzo_2024.csv`.
- `Proyecto/ml/export_cleaned.py` — script reproducible usado para generar `cleaned_data-v1.csv` a partir de `medicamentos.csv`.
- `Proyecto/ml/ingest_external.py` — script experimental para detectar múltiples snapshots y concatenarlos en formato long (se mantiene sin commitear por ahora hasta revisión).

Comandos de ejemplo
```pwsh
& "venv/Scripts/Activate.ps1"
python Proyecto/ml/clean_feb.py
python Proyecto/ml/clean_marzo.py
# (cuando esté listo) python Proyecto/ml/ingest_external.py
```

Buenas prácticas y notas de QA
- Hacer backup del raw original antes de ejecutar limpiezas automáticas. Los scripts que modifican raw deben guardar una copia en `data/backups/` si se automatizan.
- Verificar las primeras 20 filas del `cleaned_*.csv` tras ejecutar el script (`head`) y comprobar que `DESCRIPCION_LIMPIA` y `PRECIO UNITARIO B/.` se ven razonables.
- Para columnas numéricas: los scripts eliminan separadores de miles (`,`) y convierten a `float` con `errors='coerce'`. Valores NaN deben revisarse manualmente.

Próximos pasos sugeridos
- Confirmas si la convención de nombres y el esquema long-format te parecen correctos.
- Si OK: puedo consolidar febrero+marzo+enero en `long_monthly_all.csv` y luego proceder con limpieza de abril.

Contacto
- Si hay dudas o prefieres otra estructura, dime y ajusto el documento.
