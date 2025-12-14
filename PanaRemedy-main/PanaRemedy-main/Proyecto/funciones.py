import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def short_name(desc: Optional[str]) -> str:
    """Return a short label for a medication description.

    Drops parenthetical content and keeps text before the first comma.
    """
    if desc is None:
        return ""
    s = str(desc).strip()
    if not s:
        return s
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    if "," in s:
        s = s.split(",")[0]
    return s.strip()


# ---------- Internal helpers ----------
def _clean_numeric_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).copy()
    s = s.str.replace('"', '', regex=False).str.replace("'", '', regex=False)
    s = s.str.replace('\u00A0', '', regex=False).str.strip()
    s = s.str.replace(',', '', regex=False)
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')


def _ensure_col(df: pd.DataFrame, col: str):
    return col in df.columns

# ---------- Tendencias ----------
def price_trend(df: pd.DataFrame, medicamento: str) -> pd.Series:
    """Devuelve la serie de precio unitario para el medicamento.

    Soporta dos formatos de `df`:
    - Formato wide (original): busca en la columna `DESCRIPCIÓN` y en `PRECIO UNITARIO B/.`.
    - Formato long (recomendado): si `snapshot_date` o `precio_unitario` están presentes
      agrupa por fecha y devuelve la serie ordenada por fecha (valores numéricos).

    Devuelve una `pd.Series` con los precios ordenados cronológicamente (si hay fechas),
    o bien una serie de precios en el orden original cuando no hay fechas.
    """
    # Detect long-format
    if 'snapshot_date' in df.columns or 'precio_unitario' in df.columns:
        df2 = df.copy()
        # normalize date
        if 'snapshot_date' in df2.columns:
            df2['__snap'] = pd.to_datetime(df2['snapshot_date'], errors='coerce')
        else:
            df2['__snap'] = pd.NaT

        # build mask across several possible description columns
        mask = pd.Series(False, index=df2.index)
        for c in ['DESCRIPCION_LIMPIA', 'DESCRIPCION_ORIG', 'DESCRIPCIÓN', 'DESCRIPCION', 'CODIGO']:
            if c in df2.columns:
                mask = mask | df2[c].astype(str).str.contains(medicamento, case=False, na=False)

        sel = df2[mask]
        if sel.empty:
            return pd.Series(dtype=float)

        # take median price per snapshot date (handles multiple rows per date)
        if 'precio_unitario' in sel.columns:
            series = sel.groupby('__snap')['precio_unitario'].median().sort_index()
        else:
            # fallback: try common price columns
            price_col = None
            for pc in ['PRECIO UNITARIO B/.', 'PRECIO_UNITARIO', 'PRECIO']:
                if pc in sel.columns:
                    price_col = pc
                    break
            if price_col is None:
                return pd.Series(dtype=float)
            series = sel.groupby('__snap')[price_col].apply(lambda s: _clean_numeric_str(s).median()).sort_index()

        # return values as a plain series (index dropped) to keep compatibility with plotting
        return series.dropna().reset_index(drop=True).squeeze()

    # fallback: original wide-format behavior
    subset = df[df.get("DESCRIPCIÓN", df.columns[0]).astype(str).str.contains(medicamento, case=False, na=False)]
    col = "PRECIO UNITARIO B/."
    if not _ensure_col(subset, col):
        return pd.Series(dtype=float)
    s = _clean_numeric_str(subset[col]).dropna().reset_index(drop=True)
    return s


# ---------- Comparaciones ----------
def compare_medicamentos(df: pd.DataFrame) -> pd.Series:
    # If long-format, compute mean (or median) precio per presentation
    if 'precio_unitario' in df.columns or 'snapshot_date' in df.columns:
        # prefer cleaned description
        key = None
        for k in ['DESCRIPCION_LIMPIA', 'DESCRIPCION_ORIG', 'DESCRIPCIÓN', 'DESCRIPCION']:
            if k in df.columns:
                key = k
                break
        if key is None:
            return pd.Series(dtype=float)
        tmp = df.copy()
        tmp['precio_unitario'] = pd.to_numeric(tmp.get('precio_unitario') if 'precio_unitario' in tmp.columns else tmp.get('PRECIO UNITARIO B/.'), errors='coerce')
        res = tmp.groupby(key)['precio_unitario'].median().dropna()
        return res

    # fallback wide
    col = "PRECIO UNITARIO B/."
    if not _ensure_col(df, col):
        return pd.Series(dtype=float)
    tmp = df.copy()
    tmp[col] = _clean_numeric_str(tmp[col])
    return tmp.groupby("DESCRIPCIÓN")[col].mean()


def compare_farmacias(df: pd.DataFrame, medicamento: str) -> pd.Series:
    # If long-format: return mean price per snapshot date for the medicamento
    if 'precio_unitario' in df.columns or 'snapshot_date' in df.columns:
        tmp = df.copy()
        # find description column
        desc_cols = [c for c in ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','DESCRIPCIÓN','DESCRIPCION','CODIGO'] if c in tmp.columns]
        if not desc_cols:
            return pd.Series(dtype=float)
        mask = pd.Series(False, index=tmp.index)
        for c in desc_cols:
            mask = mask | tmp[c].astype(str).str.contains(medicamento, case=False, na=False)
        sel = tmp[mask].copy()
        if sel.empty:
            return pd.Series(dtype=float)
        # group by snapshot_date
        if 'snapshot_date' in sel.columns:
            sel['__snap'] = pd.to_datetime(sel['snapshot_date'], errors='coerce')
        else:
            sel['__snap'] = pd.NaT
        sel['precio_unitario'] = pd.to_numeric(sel.get('precio_unitario') if 'precio_unitario' in sel.columns else sel.get('PRECIO UNITARIO B/.'), errors='coerce')
        series = sel.groupby('__snap')['precio_unitario'].median().sort_index()
        # convert index to string labels for compatibility with plotting
        series.index = series.index.strftime('%Y-%m-%d')
        return series.dropna()

    # fallback: original behavior using existencia columns
    columns = [col for col in df.columns if ("ALMACÉN" in col.upper()) or ("EXISTENCIA" in col.upper())]
    subset = df[df.get('DESCRIPCIÓN', df.columns[0]).astype(str).str.contains(medicamento, case=False, na=False)]
    if subset.empty:
        return pd.Series(dtype=float)

    def _mean_for(col):
        vals = _clean_numeric_str(subset[col]).dropna()
        return float(vals.mean()) if vals.size > 0 else float('nan')

    result = {col: _mean_for(col) for col in columns}
    return pd.Series(result)


# ---------- Sumatorios / Resumen ----------
# summary helpers removed (not used by the GUI)


# ---------- Visualización ----------
def plot_trend(series: pd.Series, medicamento: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    # Si la serie tiene más de un valor, plot lineal; si solo uno, dibujar barra
    if len(series) > 1:
        # asegurar valores numéricos y eliminar entradas no válidas
        y = _clean_numeric_str(series).dropna().values
        ax.plot(y, marker="o")
        ax.set_xticks(list(range(len(y))))
        # no mostrar valores numéricos extra en etiquetas de eje x
        ax.set_xticklabels([str(i) for i in range(len(y))])
    else:
        # mostrar sólo nombre corto en la etiqueta (sin anotar el precio)
        try:
            val = float(pd.to_numeric(series.iloc[0].astype(str).replace(',', ''), errors='coerce'))
        except Exception:
            val = None
        ax.bar([short_name(medicamento)], [val if val is not None else 0])
    ax.set_title(f"Tendencia / Precio: {short_name(medicamento)}")
    ax.set_xlabel("Índice")
    ax.set_ylabel("Precio (B/.)")
    ax.grid(True)
    # Ajuste manual de espacio para evitar que etiquetas queden cortadas
    if len(series) > 20:
        fig.subplots_adjust(bottom=0.30)
    # aplicar tight layout para mejorar ajuste dentro del canvas embebido
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    return plt


def plot_comparison(series: pd.Series, title="Comparación de precios"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels_raw = list(series.index.astype(str))
    values_raw = pd.to_numeric(series.values, errors='coerce')
    labels_clean, values_clean = [], []
    for lab, val in zip(labels_raw, values_raw):
        lab_short = short_name(lab)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúñü]", lab_short):
            labels_clean.append(lab_short)
            values_clean.append(float(val))
    if not labels_clean:
        labels_clean = [short_name(l) for l in labels_raw]
        values_clean = [float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0 for v in values_raw]
    x = list(range(len(labels_clean)))
    ax.bar(x, values_clean, color="skyblue")
    ax.set_title(title)
    # Rotar etiquetas con tamaño reducido para muchas barras
    rotation = 45 if len(labels_clean) <= 20 else 30
    fontsize = 10 if len(labels_clean) <= 20 else 8
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, rotation=rotation, ha='right', fontsize=fontsize)
    ax.set_ylabel("Precio (B/.)")
    ax.grid(axis="y")
    # Si hay muchas etiquetas, empujar el eje x para evitar que queden apretadas
    if len(labels_clean) > 20:
        fig.subplots_adjust(bottom=0.35)
    # aplicar tight layout para mejorar ajuste dentro del canvas embebido
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    return plt


def save_chart(plt_obj, filename="grafico.png"):
    # kept for backwards compatibility: save current figure
    try:
        fig = plt.gcf()
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)
        print(f"Gráfico guardado como {filename}")
    except Exception:
        pass

__all__ = [
    'price_trend', 'compare_medicamentos', 'compare_farmacias',
    'resumen_general', 'medicamento_mas_caro', 'medicamento_mas_barato',
    'plot_trend', 'plot_comparison', 'save_chart', 'cargar_csv'
]

# ---------- Funciones adicionales (desde carpeta 'prueba') ----------
def ranking_medicamentos(ruta_csv, top_n=10):
    try:
        df = pd.read_csv(ruta_csv, encoding='latin1')
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {ruta_csv}")

    # Si es un CSV de compras con columna 'cantidad'
    if 'cantidad' in df.columns:
        ranking = (
            df.groupby('medicamento')['cantidad']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        ranking['posición'] = range(1, len(ranking) + 1)
        return ranking.head(top_n)

    # Si es un CSV de medicamentos, sumar las columnas de existencia
    # Buscamos columnas que contengan 'EXISTENCIA' o 'TOTAL' (caso-insens.)
    existencia_cols = [c for c in df.columns if ('EXISTENCIA' in c.upper()) or ('TOTAL' in c.upper())]
    if not existencia_cols:
        raise ValueError("El CSV no contiene columna 'cantidad' ni columnas de existencias reconocibles.")

    # Normalizar y convertir a numérico (quitar comas, tratar NaN)
    for col in existencia_cols:
        df[col] = _clean_numeric_str(df[col]).fillna(0)

    # Determinar llave de agrupación: usar 'DESCRIPCIÓN' si existe, sino 'medicamento' o la primera columna de texto
    if 'DESCRIPCIÓN' in df.columns:
        key = 'DESCRIPCIÓN'
    elif 'medicamento' in df.columns:
        key = 'medicamento'
    else:
        # buscar primera columna no numérica como nombre
        non_numeric = [c for c in df.columns if df[c].dtype == object]
        key = non_numeric[0] if non_numeric else df.columns[0]

    df['total_existencias'] = df[existencia_cols].sum(axis=1)
    ranking = (
        df.groupby(key)['total_existencias']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking['posición'] = range(1, len(ranking) + 1)
    ranking = ranking.rename(columns={'total_existencias': 'existencias'})
    return ranking.head(top_n)


def medicamentos_similares(ruta_csv, nombre_medicamento):
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {ruta_csv}")

    columnas_requeridas = {"medicamento", "sintomas"}
    if not columnas_requeridas.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {columnas_requeridas}")

    df["sintomas"] = df["sintomas"].fillna("").apply(lambda x: [s.strip().lower() for s in str(x).split(";") if s.strip()])

    base = df[df["medicamento"].str.lower() == nombre_medicamento.lower()]
    if base.empty:
        return pd.DataFrame({"mensaje": [f"Medicamento '{nombre_medicamento}' no encontrado."]})

    sintomas_base = set(base.iloc[0]["sintomas"])
    df["similitud"] = df["sintomas"].apply(lambda s: len(sintomas_base.intersection(s)))

    similares = df[df["medicamento"].str.lower() != nombre_medicamento.lower()]
    similares = similares.sort_values(by="similitud", ascending=False)

    return similares[["medicamento", "sintomas", "similitud"]].head(5)


def graficar_ranking(df):
    fig, ax = plt.subplots(figsize=(8,5))

    # Determinar columnas posibles
    if "medicamento" in df.columns and "cantidad" in df.columns:
        labels = [short_name(x) for x in df["medicamento"].astype(str).tolist()]
        values = df["cantidad"].tolist()
        title = "Top Medicamentos Más Comprados"
        ylabel = "Cantidad Vendida"
    elif "DESCRIPCIÓN" in df.columns and "existencias" in df.columns:
        labels = [short_name(x) for x in df["DESCRIPCIÓN"].astype(str).tolist()]
        values = df["existencias"].tolist()
        title = "Top Medicamentos por Existencias"
        ylabel = "Existencias"
    else:
        # Intentar tomar la primera columna de texto como etiqueta y la segunda como valor
        labels = df.iloc[:,0].astype(str).tolist()
        values = df.iloc[:,1].tolist()
        title = "Top Medicamentos"
        ylabel = df.columns[1] if len(df.columns) > 1 else "Valor"

    x = list(range(len(labels)))
    ax.bar(x, values, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Medicamento")
    ax.set_ylabel(ylabel)
    rotation = 45 if len(labels) <= 20 else 30
    fontsize = 10 if len(labels) <= 20 else 8
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha='right', fontsize=fontsize)
    if len(labels) > 20:
        fig.subplots_adjust(bottom=0.35)

    # Guardar en la carpeta static dentro de la carpeta Proyecto
    carpeta_static = os.path.join(os.path.dirname(__file__), "static")
    carpeta_static = os.path.normpath(carpeta_static)
    os.makedirs(carpeta_static, exist_ok=True)

    ruta_guardado = os.path.join(carpeta_static, "ranking_medicamentos.png")
    fig.savefig(ruta_guardado, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {ruta_guardado}")


# ---------------- Data helpers: list/load/concat CSVs ----------------
# (Working with a single default dataset now; list/load/concat helpers removed.)

from pathlib import Path
import csv

def cargar_csv(ruta, encoding: str = None, sep: str = None, required_cols: list = None, verbose: bool = False):
    """Cargar un CSV con heurísticas de encoding y separador.

    Args:
        ruta: ruta al archivo (str or Path).
        encoding: opcional, forzar encoding (p. ej. 'latin1').
        sep: opcional, forzar separador (p. ej. ',',';').
        required_cols: lista opcional de columnas que deben existir; si no,
            la función devolverá None.
        verbose: si True imprime información diagnóstica.

    Returns:
        pd.DataFrame o None si falla la lectura o faltan columnas requeridas.
    """
    p = Path(ruta)
    if not p.exists():
        if verbose:
            print(f"cargar_csv: archivo no encontrado: {p}")
        return None

    encodings_to_try = [encoding] if encoding else ['utf-8', 'latin1', 'cp1252']
    delims = [sep] if sep else [',', ';', '\t', '|']

    df = None
    for enc in encodings_to_try:
        for d in delims:
            try:
                df = pd.read_csv(p, encoding=enc, sep=d, low_memory=False)
                if verbose:
                    print(f"cargar_csv: leído con encoding={enc} sep={repr(d)} rows={len(df)}")
                break
            except Exception:
                df = None
        if df is not None:
            break

    if df is None:
        # último intento: dejar que pandas detecte con sniff
        try:
            sample = p.read_text(encoding='latin1', errors='replace')
            sn = csv.Sniffer()
            try:
                guessed = sn.sniff(sample[:8192])
                delim = guessed.delimiter
            except Exception:
                delim = ','
            df = pd.read_csv(p, encoding='latin1', sep=delim, low_memory=False)
            if verbose:
                print(f"cargar_csv: fallback leído con latin1 sep={repr(delim)} rows={len(df)}")
        except Exception as e:
            if verbose:
                print(f"cargar_csv: no se pudo leer {p}: {e}")
            return None

    # limpiar columnas unnamed añadidas por pandas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # normalizar nombres: strip
    df.columns = [str(c).strip() for c in df.columns]

    if required_cols:
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            if verbose:
                print(f"cargar_csv: faltan columnas requeridas: {miss}")
            return None

    return df

__all__ = [
    'short_name', 'price_trend', 'compare_medicamentos', 'compare_farmacias',
    'plot_trend', 'plot_comparison', 'save_chart', 'ranking_medicamentos',
    'forecast_linear', 'forecast_ma', 'cargar_csv'
]


# ---------------- Forecast implementations ----------------
def forecast_linear(df: pd.DataFrame, medicamento: str, horizon: int = 6, test_size: float = 0.2):
    series = price_trend(df, medicamento)
    if series is None or series.empty:
        raise ValueError('No hay datos para el medicamento especificado')

    s = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna().reset_index(drop=True)
    n = len(s)
    if n < 3:
        raise ValueError('Se requieren al menos 3 observaciones para entrenar el modelo')

    X = np.arange(n).reshape(-1, 1)
    y = s.values

    if test_size and 0 < test_size < 1:
        split = max(1, int(n * (1 - test_size)))
    else:
        split = n

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_fore = np.arange(n, n + horizon).reshape(-1, 1)
    y_fore = model.predict(X_fore)
    y_pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    if len(y_pred_test) > 0:
        mae = float(mean_absolute_error(y_test, y_pred_test))
        try:
            rmse = float(mean_squared_error(y_test, y_pred_test, squared=False))
        except TypeError:
            mse = float(mean_squared_error(y_test, y_pred_test))
            rmse = float(np.sqrt(mse))
    else:
        mae = None
        rmse = None

    metrics = {'mae': mae, 'rmse': rmse}
    idx = list(range(n, n + horizon))
    forecast_df = pd.DataFrame({'step': idx, 'y_pred': y_fore})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n), y, label='observed')
    if len(X_test) > 0:
        ax.plot(np.arange(split, n), y_pred_test, label='pred_test', linestyle='--')
    ax.plot(idx, y_fore, label='forecast', marker='o')
    ax.set_title(f'Forecast (Linear) - {medicamento}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Precio (B/.)')
    ax.legend()
    ax.grid(True)

    return forecast_df, metrics, fig, (y_test, y_pred_test)


def forecast_ma(df: pd.DataFrame, medicamento: str, horizon: int = 6, window: int = 3, test_size: float = 0.2):
    series = price_trend(df, medicamento)
    if series is None or series.empty:
        raise ValueError('No hay datos para el medicamento especificado')

    s = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna().reset_index(drop=True)
    n = len(s)
    if n < 3:
        raise ValueError('Se requieren al menos 3 observaciones para entrenar el modelo')

    if test_size and 0 < test_size < 1:
        split = max(1, int(n * (1 - test_size)))
    else:
        split = n

    y = s.values
    y_train = y[:split]
    y_test = y[split:]

    window = int(max(1, min(window, n)))

    y_pred_test = []
    history = list(y_train)
    for i in range(len(y_test)):
        if len(history) >= window:
            pred = float(np.mean(history[-window:]))
        else:
            pred = float(np.mean(history))
        y_pred_test.append(pred)
        history.append(y_test[i])

    if len(y) >= window:
        last_window_mean = float(np.mean(y[-window:]))
    else:
        last_window_mean = float(np.mean(y))

    y_fore = np.array([last_window_mean] * horizon)

    if len(y_pred_test) > 0:
        y_pred_test_arr = np.array(y_pred_test)
        mae = float(np.mean(np.abs(y_test - y_pred_test_arr)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred_test_arr) ** 2)))
    else:
        mae = None
        rmse = None

    metrics = {'mae': mae, 'rmse': rmse}
    idx = list(range(n, n + horizon))
    forecast_df = pd.DataFrame({'step': idx, 'y_pred': y_fore})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(n), y, label='observed')
    if len(y_pred_test) > 0:
        ax.plot(np.arange(split, n), y_pred_test, label='pred_test_MA', linestyle='--')
    ax.plot(idx, y_fore, label='forecast_MA', marker='o')
    ax.set_title(f'Forecast (MA window={window}) - {medicamento}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Precio (B/.)')
    ax.legend()
    ax.grid(True)

    return forecast_df, metrics, fig, (y_test, np.array(y_pred_test))
