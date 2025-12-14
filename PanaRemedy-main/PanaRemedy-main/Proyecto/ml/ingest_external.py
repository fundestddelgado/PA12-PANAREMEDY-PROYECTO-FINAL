"""
Ingest CSV files in Proyecto/data/ (excluding cleaned/) and concatenate into a normalized CSV:
Proyecto/data/cleaned/cleaned_data-v2.csv

Behavior:
- For each CSV found in Proyecto/data (non-recursive, excluding Proyecto/data/cleaned),
  - read (latin1 fallback utf-8), detect price and description columns
  - if month-like columns (e.g. 'ENERO 2024') are present, melt them into long form with 'snapshot' date
  - otherwise keep row as-is and attach 'source_file'
- Normalize: create 'DESCRIPCIÓN_LIMPIA' using funciones.short_name and 'PRECIO_UNITARIO' using funciones._clean_numeric_str when possible
- Save result to Proyecto/data/cleaned/cleaned_data-v2.csv
"""
from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np

# ensure Proyecto is importable
THIS_DIR = Path(__file__).resolve().parent
PROY_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROY_DIR))
import funciones

DATA_DIR = PROY_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = CLEANED_DIR / "cleaned_data-v2.csv"

SPANISH_MONTHS = {
    'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,
    'SEPTIEMBRE':9,'SETIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12
}
MONTH_RE = re.compile(r"(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|JULIO|AGOSTO|SEPTIEMBRE|SETIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)\s*\D*?(\d{4})", re.I)


def try_read_csv(path: Path):
    # simple reader kept for compatibility (not used below)
    try:
        return pd.read_csv(path, encoding='latin1', low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding='utf-8', low_memory=False)
        except Exception as e:
            raise


def robust_read(path: Path):
    """Try multiple read_csv options to handle inconsistent CSV formats."""
    attempts = [
        {'encoding': 'latin1', 'engine': 'c', 'sep': ','},
        {'encoding': 'latin1', 'engine': 'python', 'sep': None},
        {'encoding': 'latin1', 'engine': 'python', 'sep': ';'},
        {'encoding': 'utf-8', 'engine': 'python', 'sep': None},
    ]
    last_exc = None
    for opts in attempts:
        try:
            return pd.read_csv(path, low_memory=False, **opts)
        except Exception as e:
            last_exc = e
            # try next
    # Final fallback: try reading as text and attempting to fix unbalanced quotes
    try:
        txt = path.read_text(encoding='latin1')
        # heuristic: replace lone '"' occurrences that break CSV structure
        fixed = txt.replace('\"\n', '"\n')
        from io import StringIO
        return pd.read_csv(StringIO(fixed), low_memory=False, engine='python')
    except Exception:
        raise last_exc


def find_price_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if 'PRECIO' in c.upper()]
    for c in candidates:
        up = c.upper().strip()
        if up in ("PRECIO UNITARIO B/." , "PRECIO UNITARIO", "PRECIO"):
            return c
    return candidates[0] if candidates else None


def find_description_column(df: pd.DataFrame):
    for c in ['DESCRIPCIÓN','DESCRIPCION','medicamento','MEDICAMENTO','descripcion']:
        if c in df.columns:
            return c
    # fallback: first object dtype column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]


def detect_month_cols(df: pd.DataFrame):
    return [c for c in df.columns if MONTH_RE.search(c.upper())]


def parse_month_from_col(colname: str):
    m = MONTH_RE.search(colname.upper())
    if not m:
        return None
    month_name = m.group(1).upper()
    year = int(m.group(2))
    month = SPANISH_MONTHS.get(month_name)
    if not month:
        return None
    return f"{year:04d}-{month:02d}-01"


def process_file(path: Path):
    df = try_read_csv(path)
    original_shape = df.shape
    desc_col = find_description_column(df)
    price_col = find_price_column(df)
    month_cols = detect_month_cols(df)

    records = []

    if month_cols:
        # id_vars: keep basic columns (codigo/descripcion/price if present)
        id_vars = []
        for c in ['CODIGO','codigo']:
            if c in df.columns:
                id_vars.append(c)
                break
        id_vars.append(desc_col)
        if price_col:
            id_vars.append(price_col)
        # remove duplicates
        id_vars = [c for c in id_vars if c in df.columns]
        melted = df.melt(id_vars=id_vars, value_vars=month_cols, var_name='month_col', value_name='month_value')
        # parse month_col to date
        melted['snapshot'] = melted['month_col'].apply(parse_month_from_col)
        # create common columns
        melted['source_file'] = path.name
        # treat month_value as existencias
        melted = melted.rename(columns={'month_value':'existencias'})
        return melted
    else:
        # No month columns: keep rows as-is
        df2 = df.copy()
        df2['source_file'] = path.name
        return df2


def main():
    csv_files = [p for p in DATA_DIR.glob('*.csv') if not p.parts[-2:] == ('data','cleaned') and 'cleaned_data' not in p.name]
    # filter out cleaned folder files
    csv_files = [p for p in csv_files if 'cleaned' not in str(p.parent)]

    if not csv_files:
        print('No CSV files to process in Proyecto/data/')
        return 1

    print('Files to process:')
    for p in csv_files:
        print(' -', p.name)

    processed = []
    for p in csv_files:
        try:
            dfp = process_file(p)
            # normalize description and price where possible
            desc_col = find_description_column(dfp)
            if desc_col in dfp.columns:
                dfp['DESCRIPCIÓN_LIMPIA'] = dfp[desc_col].astype(str).apply(funciones.short_name)
            if 'PRECIO_UNITARIO' not in dfp.columns and find_price_column(dfp):
                pc = find_price_column(dfp)
                dfp['PRECIO_UNITARIO'] = funciones._clean_numeric_str(dfp[pc])
            elif 'PRECIO_UNITARIO' in dfp.columns:
                dfp['PRECIO_UNITARIO'] = funciones._clean_numeric_str(dfp['PRECIO_UNITARIO'])
            # append
            processed.append(dfp)
        except Exception as e:
            print(f'Error procesando {p.name}: {e}')

    if not processed:
        print('No data processed successfully')
        return 2

    all_df = pd.concat(processed, ignore_index=True, sort=False)
    # basic normalization: strip column names
    all_df.columns = [c.strip() for c in all_df.columns]

    # remove rows without description or price entirely NaN (keep existencias-only rows)
    # save
    all_df.to_csv(OUTPUT, index=False, encoding='utf-8')

    print(f'Wrote consolidated file: {OUTPUT} (rows={len(all_df)})')
    # quick summary
    print('Unique presentations:', all_df['DESCRIPCIÓN_LIMPIA'].nunique() if 'DESCRIPCIÓN_LIMPIA' in all_df.columns else 'n/a')
    if 'snapshot' in all_df.columns:
        s = pd.to_datetime(all_df['snapshot'], errors='coerce')
        if s.notna().any():
            print('Snapshot date range:', s.min(), '->', s.max())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
