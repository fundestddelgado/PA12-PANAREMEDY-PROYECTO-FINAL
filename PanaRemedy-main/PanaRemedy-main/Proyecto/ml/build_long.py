from pathlib import Path
import pandas as pd
import re
from datetime import datetime

CLEAN_DIR = Path(__file__).resolve().parent.parent / 'data' / 'cleaned'
OUT = CLEAN_DIR / 'long_monthly_all.csv'

month_map = {
    'ENERO': '2024-01-01',
    'FEBRERO': '2024-02-01',
    'MARZO': '2024-03-01',
    'ABRIL': '2024-04-01',
    'MAYO': '2024-05-01',
    'JUNIO': '2024-06-01',
    'JULIO': '2024-07-01',
    'AGOSTO': '2024-08-01',
    'SEPTIEMBRE': '2024-09-01',
    'OCTUBRE': '2024-10-01',
    'NOVIEMBRE': '2024-11-01',
    'DICIEMBRE': '2024-12-01',
}

files = sorted([p for p in CLEAN_DIR.glob('cleaned_*.csv') if 'long_' not in p.name.lower()])
print('Found cleaned files:', files)

def find_col(cols, keywords):
    cols_up = {c.upper(): c for c in cols}
    for k in keywords:
        for cu, orig in cols_up.items():
            if k in cu:
                return orig
    return None

frames = []
for f in files:
    print('Processing', f.name)
    df = pd.read_csv(f, encoding='utf-8', low_memory=False)
    cols = list(df.columns)
    # detect month from filename first
    month = None
    for m in month_map:
        if m in f.name.upper():
            month = m
            break
    # fallback: detect any month in column names
    if month is None:
        for c in cols:
            cu = c.upper()
            for m in month_map:
                if m in cu:
                    month = m
                    break
            if month:
                break
    snapshot = month_map.get(month, None)
    if snapshot is None:
        # default: use file modified date as snapshot
        snapshot = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-01')
    # find key columns
    codigo_col = find_col(cols, ['CODIGO']) or cols[0]
    desc_col = find_col(cols, ['DESCRIPCION','DESCRIPCIÓN'])
    precio_col = find_col(cols, ['PRECIO UNITARIO','PRECIO_UNITARIO','PRECIO'])
    total_col = find_col(cols, ['TOTAL DE EXISTENCIAS','TOTAL_DE_EXISTENCIAS','TOTAL'])
    monto_col = find_col(cols, ['MONTO DE EXISTENCIAS','MONTO_DE_EXISTENCIAS','MONTO'])
    cdpa_col = find_col(cols, ['CDPA'])
    cddi_col = find_col(cols, ['CDDI'])
    cdch_col = find_col(cols, ['CDCH'])
    # Prepare standardized DataFrame
    out = pd.DataFrame()
    out['CODIGO'] = df[codigo_col].astype(str)
    out['DESCRIPCION_ORIG'] = df[desc_col].astype(str) if desc_col else df[codigo_col].astype(str)
    # precio
    if precio_col and precio_col in df.columns:
        out['precio_unitario'] = pd.to_numeric(df[precio_col].astype(str).str.replace(',','').str.replace('"',''), errors='coerce')
    else:
        out['precio_unitario'] = pd.NA
    # existencias
    def get_num(col):
        if col and col in df.columns:
            return pd.to_numeric(df[col].astype(str).str.replace(',','').str.replace('"',''), errors='coerce')
        else:
            return pd.NA
    out['existencia_cdpa'] = get_num(cdpa_col)
    out['existencia_cddi'] = get_num(cddi_col)
    out['existencia_cdch'] = get_num(cdch_col)
    out['total_existencias'] = get_num(total_col)
    out['monto_existencias'] = get_num(monto_col)
    out['DESCRIPCION_LIMPIA'] = df.get('DESCRIPCION_LIMPIA') if 'DESCRIPCION_LIMPIA' in df.columns else df.get('DESCRIPCIÓN_LIMPIA') if 'DESCRIPCIÓN_LIMPIA' in df.columns else out['DESCRIPCION_ORIG'].str.upper()
    out['snapshot_date'] = snapshot
    frames.append(out)

if not frames:
    print('No frames to concatenate')
    raise SystemExit(1)
all_df = pd.concat(frames, ignore_index=True)
# reorder
all_df = all_df[['CODIGO','DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','snapshot_date','precio_unitario','existencia_cdpa','existencia_cddi','existencia_cdch','total_existencias','monto_existencias']]
OUT.parent.mkdir(parents=True, exist_ok=True)
all_df.to_csv(OUT, index=False, encoding='utf-8')
print('Wrote', OUT, 'rows=', len(all_df))
