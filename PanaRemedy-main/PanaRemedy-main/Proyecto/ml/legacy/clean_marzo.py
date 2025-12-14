from pathlib import Path
import pandas as pd

src = Path(__file__).resolve().parent.parent / 'data' / 'inventario-de-medicamentos-marzo-2024.csv'
out = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'cleaned_marzo_2024.csv'

cols = ['CODIGO','DESCRIPCION','PRECIO UNITARIO B/.',
        'EXISTENCIA ALMACEN CDPA MARZO 2024',
        'EXISTENCIA ALMACEN CDDI MARZO 2024',
        'EXISTENCIA ALMACEN CDCH MARZO 2024',
        'TOTAL DE EXISTENCIAS DISPONIBLES MARZO 2024',
        'MONTO DE EXISTENCIAS EN B/.']

print('Reading', src)
# try several encodings
encodings=['utf-8','latin-1','cp1252']
df=None
for enc in encodings:
    try:
        # try semicolon first (many files use ; as separator)
        df=pd.read_csv(src, header=None, encoding=enc, sep=';', quotechar='"', engine='python')
        print('OK encoding',enc)
        break
    except Exception as e:
        print('Failed',enc, e)
if df is None:
    raise SystemExit('All encodings failed')

if df.shape[1] != len(cols):
    print('Warning: unexpected column count', df.shape[1])

# assign/truncate/pad columns
if df.shape[1] >= len(cols):
    df = df.iloc[:, :len(cols)]
else:
    for i in range(len(cols) - df.shape[1]):
        df[i+df.shape[1]] = ''

df.columns = cols

# Clean numeric columns
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
