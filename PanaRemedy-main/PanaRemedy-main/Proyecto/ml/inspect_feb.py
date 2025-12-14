import pandas as pd
p='Proyecto/data/cuadro-de-inventario-de-medicamentos-feb.-2024 (1).csv'
encodings=['utf-8','latin-1','cp1252']
for enc in encodings:
    try:
        df=pd.read_csv(p, header=None, encoding=enc, quotechar='"', engine='python')
        print('OK encoding',enc)
        break
    except Exception as e:
        print('Failed',enc, e)
else:
    raise SystemExit('All encodings failed')
print('shape', df.shape)
print('\nFirst 5 rows:')
print(df.head(5).to_string(index=False))
print('\nSample row types and reprs:')
row=df.iloc[0].tolist()
for i,val in enumerate(row):
    print(i, type(val), repr(str(val))[:120])
print('\nColumn count per row (first 20 rows):')
for i in range(min(20,len(df))):
    print(i, len(df.iloc[i].tolist()))
