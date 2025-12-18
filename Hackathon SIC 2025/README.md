
Instalación y ejecución

Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ejecutar tests

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python -m pytest -q
```

Comparación automática de modelos

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
python scripts/compare_models.py
```

Ejecutar la app Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/main_app.py
```

Nota (Windows)

Si `pip install tensorflow` falla por rutas largas, cree el venv en una ruta corta (ej. `C:\proj\optinv_venv`) o use `conda`.

