"""Helper to run ML smoke tests safely.

Steps:
- backup `data/cleaned/cleaned_data-v1.csv`
- write a ML-friendly mapped CSV with columns `nombre_medicamento`, `precio`, `cantidad`, `descuento`
- import and run `ml.clustering_farmacias.main()` and `ml.regresion_precios.main()`
- restore the original CSV

Run from repo root with the project's venv.
"""
from pathlib import Path
import shutil
import sys
import pandas as pd
import traceback

ROOT = Path(__file__).resolve().parent.parent
cleaned = ROOT / 'data' / 'cleaned' / 'cleaned_data-v1.csv'
bak = cleaned.with_name('cleaned_data-v1.bak.csv')

if not cleaned.exists():
    print('cleaned_data-v1.csv not found:', cleaned)
    raise SystemExit(1)

print('Backing up', cleaned, '->', bak)
shutil.copy(cleaned, bak)

try:
    df = pd.read_csv(bak, encoding='utf-8')
    # build mapped df
    def first_of(*cols):
        for c in cols:
            if c in df.columns:
                return df[c]
        return None

    nombre = first_of('DESCRIPCIÓN_LIMPIA', 'DESCRIPCION_LIMPIA', 'DESCRIPCIÓN', 'DESCRIPCION')
    precio = first_of('PRECIO_UNITARIO', 'precio_unitario', 'PRECIO UNITARIO B/.', 'PRECIO')
    total = first_of('total_existencias', 'TOTAL DE EXISTENCIAS  DISPONIBLES ENERO 2024', 'TOTAL DE EXISTENCIAS DISPONIBLES')

    df_ml = pd.DataFrame()
    if nombre is None:
        df_ml['nombre_medicamento'] = df.iloc[:, 0].astype(str)
    else:
        df_ml['nombre_medicamento'] = nombre.astype(str)

    if precio is None:
        df_ml['precio'] = pd.to_numeric(df.iloc[:, 1].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    else:
        df_ml['precio'] = pd.to_numeric(precio.astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    if total is None:
        df_ml['cantidad'] = 0
    else:
        df_ml['cantidad'] = pd.to_numeric(total.astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    df_ml['descuento'] = 0

    # write mapped CSV to the expected path (overwrite)
    print('Writing mapped ML CSV to', cleaned)
    df_ml.to_csv(cleaned, index=False, encoding='utf-8')

    # run clustering and regression
    print('Running clustering_farmacias...')
    sys.path.insert(0, str(ROOT))
    try:
        import ml.clustering_farmacias as cluster_mod
        cluster_mod.main()
    except Exception:
        print('clustering_farmacias failed:')
        traceback.print_exc()

    print('Running regresion_precios...')
    try:
        import ml.regresion_precios as reg_mod
        reg_mod.main()
    except Exception:
        print('regresion_precios failed:')
        traceback.print_exc()

finally:
    # restore backup
    if bak.exists():
        print('Restoring backup', bak, '->', cleaned)
        shutil.move(str(bak), str(cleaned))
    else:
        print('Backup not found; cannot restore')

print('Done')
