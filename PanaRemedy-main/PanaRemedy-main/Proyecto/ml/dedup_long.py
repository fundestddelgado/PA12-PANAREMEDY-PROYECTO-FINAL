from pathlib import Path
import pandas as pd

IN_FILE = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'long_monthly_all.csv'
OUT_FILE = IN_FILE.parent / 'long_monthly_all_dedup.csv'

print('Reading', IN_FILE)
df = pd.read_csv(IN_FILE, encoding='utf-8', low_memory=False)
print('Rows input:', len(df))

# Ensure snapshot_date is parsed consistently
if 'snapshot_date' in df.columns:
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.strftime('%Y-%m-01')
else:
    df['snapshot_date'] = '1970-01-01'

# Columns expected
num_cols = ['precio_unitario','existencia_cdpa','existencia_cddi','existencia_cdch','total_existencias','monto_existencias']
text_cols = ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG']

# Normalize numeric columns: remove commas if they slipped in
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','').str.replace('"',''), errors='coerce')
    else:
        df[c] = pd.NA

# Group by CODIGO + snapshot_date and aggregate
agg_funcs = {
    'DESCRIPCION_LIMPIA': lambda s: s.dropna().astype(str).replace('nan','').replace('', pd.NA).drop_duplicates().iat[0] if len(s.dropna().astype(str).replace('nan','').replace('', pd.NA))>0 else pd.NA,
    'DESCRIPCION_ORIG': lambda s: s.dropna().astype(str).replace('nan','').replace('', pd.NA).drop_duplicates().iat[0] if len(s.dropna().astype(str).replace('nan','').replace('', pd.NA))>0 else pd.NA,
    'precio_unitario': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').median()) if len(s.dropna())>0 else pd.NA,
    'existencia_cdpa': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').sum()) if len(s.dropna())>0 else 0,
    'existencia_cddi': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').sum()) if len(s.dropna())>0 else 0,
    'existencia_cdch': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').sum()) if len(s.dropna())>0 else 0,
    'total_existencias': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').sum()) if len(s.dropna())>0 else 0,
    'monto_existencias': lambda s: float(pd.to_numeric(s.dropna(), errors='coerce').sum()) if len(s.dropna())>0 else 0,
}

group_cols = ['CODIGO','snapshot_date']
# Ensure grouping columns exist
for gc in group_cols:
    if gc not in df.columns:
        raise SystemExit(f'Missing column {gc} in input')

print('Grouping by', group_cols)
# Apply aggregation
agg_df = df.groupby(group_cols).agg(agg_funcs).reset_index()

# Fill remaining NA: precio -> 0 or keep NaN? we'll fill with NA -> 0 for numeric
for c in num_cols:
    if c in agg_df.columns:
        agg_df[c] = agg_df[c].fillna(0)

for c in ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG']:
    if c in agg_df.columns:
        agg_df[c] = agg_df[c].fillna('')

# Reorder columns
cols = ['CODIGO','DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','snapshot_date'] + num_cols
agg_df = agg_df[cols]

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
agg_df.to_csv(OUT_FILE, index=False, encoding='utf-8')
print('Wrote', OUT_FILE, 'rows=', len(agg_df))

# Print basic summary
print('\nSummary rows per snapshot:')
print(agg_df.groupby('snapshot_date').size().to_string())
