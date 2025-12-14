import pandas as pd
p='Proyecto/data/inventario-de-medicamentos-marzo-2024.csv'
for sep in [',',';','\t','|']:
    try:
        df = pd.read_csv(p, sep=sep, encoding='latin-1', engine='python', header=None)
        print('sep',repr(sep),'shape',df.shape)
        nonnull = df.notnull().all(axis=1).sum()
        print(' rows with all non-null fields:', nonnull)
    except Exception as e:
        print('sep',repr(sep),'failed',e)
