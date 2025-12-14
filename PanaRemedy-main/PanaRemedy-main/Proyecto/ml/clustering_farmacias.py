import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from funciones import cargar_csv
except Exception:
    try:
        from Proyecto.funciones import cargar_csv
    except Exception:
            cargar_csv = None

if cargar_csv is None:
    def cargar_csv(ruta):
        import pandas as _pd
        try:
            return _pd.read_csv(ruta, encoding='utf-8', low_memory=False)
        except Exception:
            return _pd.read_csv(ruta, encoding='latin1', low_memory=False)


def _select_columns(df):
    columnas = []

    for c in [
        'precio_unitario', 'PRECIO_UNITARIO', 'PRECIO UNITARIO B/.',
        'precio', 'PRECIO'
    ]:
        if c in df.columns:
            columnas.append(c)
            break

    for c in [
        'total_existencias',
        'TOTAL DE EXISTENCIAS DISPONIBLES',
        'cantidad'
    ]:
        if c in df.columns:
            columnas.append(c)
            break

    # fallback: try to find any column that looks like an existence/quantity column
    if len(columnas) < 2:
        for c in df.columns:
            cu = str(c).upper()
            if 'EXIST' in cu or 'TOTAL' in cu or 'CANT' in cu:
                if c not in columnas:
                    columnas.append(c)
                    break

    return columnas


def _clean_numeric(series):
    s = series.astype(str).str.replace(',', '')
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)


def main():
    from pathlib import Path
    ruta = Path(__file__).resolve().parent.parent / 'data' / 'cleaned' / 'cleaned_data-v1.csv'
    df = cargar_csv(str(ruta)) if cargar_csv is not None else None

    if df is None or df.empty:
        print("No se pudo cargar el dataset.")
        return

    columnas = _select_columns(df)

    if len(columnas) < 2:
        print(" No se encontraron columnas suficientes para clustering.")
        return

    X = df[columnas].copy()

    for col in columnas:
        X[col] = _clean_numeric(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    print("\n CLUSTERING DE PRODUCTOS")
    print("Columnas usadas:", columnas)
    print("\nCentroides (escalados):")
    print(kmeans.cluster_centers_)

    print("\nConteo por cluster:")
    print(df['cluster'].value_counts())


if __name__ == "__main__":
    main()

