"""
Export cleaned dataset using cleaning helpers from Proyecto/funciones.py
Genera: Proyecto/exports/cleaned_data-v1.csv

Ejecución: python Proyecto/ml/export_cleaned.py
"""
from pathlib import Path
import sys
import pandas as pd

# rutas relativas (resuelven desde la localización del script)
THIS_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = THIS_DIR.parent
# asegurarnos de poder importar el módulo `funciones` desde el directorio Proyecto
sys.path.insert(0, str(PROYECTO_DIR))
import funciones
# acceder a los helpers explícitos
_clean_numeric_str = funciones._clean_numeric_str
short_name = funciones.short_name
DATA_DIR = PROYECTO_DIR / "data"
EXPORTS_DIR = PROYECTO_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "medicamentos.csv"
OUTPUT_CSV = EXPORTS_DIR / "cleaned_data-v1.csv"


def find_price_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if "PRECIO" in c.upper()]
    # prefer exact match
    for c in candidates:
        if c.upper().strip() in ("PRECIO UNITARIO B/." , "PRECIO UNITARIO", "PRECIO"):
            return c
    return candidates[0] if candidates else None


def main():
    if not INPUT_CSV.exists():
        print(f"Error: no se encontró {INPUT_CSV}")
        return 2

    # leer CSV con encoding latin1 por si hay caracteres especiales
    df = pd.read_csv(INPUT_CSV, encoding='latin1', low_memory=False)
    original_rows = len(df)

    # eliminar columnas vacías tipo 'Unnamed'
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # detectar columna de precio y limpiarla
    price_col = find_price_column(df)
    if price_col:
        df['PRECIO_UNITARIO'] = _clean_numeric_str(df[price_col])
    else:
        # si no hay columna de precio, crear columna vacía
        df['PRECIO_UNITARIO'] = pd.Series(dtype=float)

    # normalizar columna DESCRIPCIÓN si existe
    if 'DESCRIPCIÓN' in df.columns:
        df['DESCRIPCIÓN_LIMPIA'] = df['DESCRIPCIÓN'].astype(str).apply(short_name)
    elif 'medicamento' in df.columns:
        df['DESCRIPCIÓN_LIMPIA'] = df['medicamento'].astype(str).apply(short_name)
    else:
        df['DESCRIPCIÓN_LIMPIA'] = df.iloc[:,0].astype(str).apply(short_name)

    # quitar filas sin precio
    df_clean = df.copy()
    df_clean = df_clean[df_clean['PRECIO_UNITARIO'].notna()].reset_index(drop=True)

    # guardar CSV limpio en UTF-8
    df_clean.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # imprimir resumen
    print(f"Input: {INPUT_CSV} (rows={original_rows})")
    print(f"Output: {OUTPUT_CSV} (rows={len(df_clean)})")
    print("Primeras 5 filas:")
    print(df_clean.head(5).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
