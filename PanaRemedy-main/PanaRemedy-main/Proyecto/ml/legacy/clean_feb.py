from pathlib import Path
import pandas as pd

src = Path(__file__).resolve().parent.parent / 'data' / 'cuadro-de-inventario-de-medicamentos-feb.-2024 (1).csv'
out = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'cleaned_febrero_2024.csv'

cols = ['CODIGO','DESCRIPCION','PRECIO UNITARIO B/.',
        'EXISTENCIA ALMACEN CDPA FEBRERO 2024',
        'EXISTENCIA ALMACEN CDDI FEBRERO 2024',
        'EXISTENCIA ALMACEN CDCH FEBRERO 2024',
        'TOTAL DE EXISTENCIAS DISPONIBLES FEBRERO 2024',
        'MONTO DE EXISTENCIAS EN B/.']

print('Reading', src)
# try latin-1 (we detected it earlier)
df = pd.read_csv(src, header=None, encoding='latin-1', quotechar='"', engine='python')
if df.shape[1] != len(cols):
    print('Warning: unexpected column count', df.shape[1])

# assign columns (if more/less, truncate/pad)
if df.shape[1] >= len(cols):
    df = df.iloc[:, :len(cols)]
else:
    # pad with empty columns
    for i in range(len(cols) - df.shape[1]):
        df[i+df.shape[1]] = ''

df.columns = cols

# Clean numeric columns: remove thousands separators and convert
num_cols = [c for c in cols if 'EXISTENCIA' in c or c=='MONTO DE EXISTENCIAS EN B/.' or 'PRECIO' in c]
for c in num_cols:
    df[c] = df[c].astype(str).str.replace('"','')
    df[c] = df[c].str.replace(',','')
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Clean description
df['DESCRIPCION_LIMPIA'] = df['DESCRIPCION'].astype(str).str.strip().str.upper()

out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8')
print('Wrote', out, 'rows=', len(df))
