import sys
import os
import traceback
import pandas as pd
# Ensure repo root is on sys.path so imports from Proyecto work when running script directly
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    import Proyecto.funciones as funcs
except Exception:
    import funciones as funcs

path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned', 'long_monthly_all_dedup.csv'))
print('Loading:', path)
try:
    df = pd.read_csv(path, encoding='utf-8')
    print('Loaded rows:', len(df))
    print('Columns:', list(df.columns))
except Exception as e:
    print('Error loading CSV:', e)
    raise

med = 'paracetamol'
print('Calling compare_farmacias with', med)
try:
    res = funcs.compare_farmacias(df, med)
    print('Result type:', type(res))
    print(res.head(30))
except Exception as e:
    print('Exception occurred:')
    traceback.print_exc()

print('\nCalling compare_medicamentos (no med filter)')
try:
    res2 = funcs.compare_medicamentos(df)
    print('Result type:', type(res2))
    print(res2.head(10))
except Exception as e:
    print('Exception occurred in compare_medicamentos:')
    traceback.print_exc()

print('\nCalling price_trend for', med)
try:
    trend = funcs.price_trend(df, med)
    print('Trend type:', type(trend))
    print('Trend values (head):', trend.head(10).tolist() if hasattr(trend, 'head') else list(trend)[:10])
except Exception as e:
    print('Exception occurred in price_trend:')
    traceback.print_exc()
