import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
try:
    from funciones import cargar_csv
except Exception:
    try:
        from Proyecto.funciones import cargar_csv
    except Exception:
        cargar_csv = None

if cargar_csv is None:
    # fallback simple loader if funciones.cargar_csv isn't importable
    def cargar_csv(ruta):
        import pandas as _pd
        try:
            return _pd.read_csv(ruta, encoding='utf-8', low_memory=False)
        except Exception:
            return _pd.read_csv(ruta, encoding='latin1', low_memory=False)

def _select_price_column(df):
    for c in [
        'precio_unitario', 'PRECIO_UNITARIO', 'PRECIO UNITARIO B/.',
        'precio', 'PRECIO'
    ]:
        if c in df.columns:
            return c
    return None


def _select_quantity_column(df):
    for c in [
        'total_existencias',
        'TOTAL DE EXISTENCIAS  DISPONIBLES ENERO 2024',
        'TOTAL DE EXISTENCIAS DISPONIBLES',
        'cantidad'
    ]:
        if c in df.columns:
            return c
    return None


def _select_value_column(df):
    for c in [
        'monto_existencias',
        'MONTO DE EXISTENCIAS EN B/.',
        'descuento'
    ]:
        if c in df.columns:
            return c
    return None


def _clean_numeric(series):
    s = series.astype(str).str.replace(',', '')
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)


def main():
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'cleaned_data-v1.csv'
    df = cargar_csv(str(ruta))

    if df is None or df.empty:
        print("No se pudo cargar el dataset.")
        return

    price_col = _select_price_column(df)
    qty_col = _select_quantity_column(df)
    val_col = _select_value_column(df)

    if price_col is None:
        print(" No se encontró una columna de precio.")
        return

    if qty_col is None:
        df['__cantidad'] = 0
        qty_col = '__cantidad'

    if val_col is None:
        df['__valor'] = 0
        val_col = '__valor'

    df[qty_col] = _clean_numeric(df[qty_col])
    df[val_col] = _clean_numeric(df[val_col])
    y = _clean_numeric(df[price_col])

    X = df[[qty_col, val_col]].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n REGRESIÓN LINEAL - PRECIO")
    print("R²:", round(r2_score(y_test, y_pred), 4))
    print("\nCoeficientes:")
    for col, coef in zip(X.columns, modelo.coef_):
        print(f" - {col}: {coef:.4f}")


if __name__ == "__main__":
    main()

