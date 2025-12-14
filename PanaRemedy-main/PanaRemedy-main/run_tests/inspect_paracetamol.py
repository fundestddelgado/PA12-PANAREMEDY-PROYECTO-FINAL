from pathlib import Path
import pandas as pd
import re

csv = Path('Proyecto/data/cleaned/cleaned_data-v1.csv')
print('CSV exists:', csv.exists())
df = pd.read_csv(csv, encoding='utf-8')
print('Columns:', df.columns.tolist())

# detect description column
desc_col = None
for c in ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','DESCRIPCIÓN','DESCRIPCION','CODIGO']:
    if c in df.columns:
        desc_col = c
        break
print('desc_col:', desc_col)

# candidate price columns
price_candidates = [c for c in ['precio_unitario','PRECIO_UNITARIO','PRECIO','PRECIO UNITARIO B/.','PRECIO UNITARIO B/.'] if c in df.columns]
# also include any column containing PRECIO
for c in df.columns:
    if 'PRECIO' in str(c).upper() and c not in price_candidates:
        price_candidates.append(c)
print('price candidates found:', price_candidates)

# monto/total candidates
monto_candidates = [c for c in df.columns if 'MONTO' in str(c).upper() or 'VALOR' in str(c).upper()]
total_candidates = [c for c in df.columns if 'EXIST' in str(c).upper() or 'TOTAL' in str(c).upper()]
print('monto candidates:', monto_candidates)
print('total candidates (existencias):', total_candidates)

# helper clean
import numpy as np
import pandas as pd

def clean_nums(series):
    s = series.astype(str).fillna('')
    s = s.str.replace(',', '', regex=False)
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

# show stats for each candidate
for col in price_candidates:
    nums = clean_nums(df[col])
    print('\nColumn:', col)
    print('  non-null count:', int(nums.notna().sum()))
    print('  sample values:', nums.dropna().head(10).tolist())
    print('  describe:\n', nums.describe())

# show for monto and total
for col in monto_candidates:
    nums = clean_nums(df[col])
    print('\nMonto column:', col)
    print('  sample values:', nums.dropna().head(5).tolist())
    print('  describe:\n', nums.describe())

for col in total_candidates[:6]:
    nums = clean_nums(df[col])
    print('\nTotal/Exist column:', col)
    print('  sample values:', nums.dropna().head(5).tolist())
    print('  describe:\n', nums.describe())

# filter paracetamol rows
if desc_col:
    mask = df[desc_col].astype(str).str.contains('PARACETAMOL', case=False, na=False)
    print('\nParacetamol rows:', int(mask.sum()))
    if mask.sum() > 0:
        sample = df.loc[mask].head(20)
        print(sample.to_string(index=False))
else:
    print('No description column to filter by')
