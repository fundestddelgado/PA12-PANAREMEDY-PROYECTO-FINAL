import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# se cargar el dataset limpio
def cargar_dataset(ruta="cleaned_data-v1.csv"):
    # intenta primero con utf-8 y si falla cae a latin1
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except Exception:
        df = pd.read_csv(ruta, encoding="latin1")
    return df
    

# Mapa de calor #1: precio vs existencia

def heatmap_precio_existencia(df):
    # buscar columnas posibles
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    precio_col = _find_col(df, ["PRECIO_UNITARIO", "precio_unitario", "PRECIO UNITARIO B/.", "PRECIO", "PRECIO_UNITARIO"])
    exist_col = _find_col(df, ["TOTAL DE EXISTENCIAS  DISPONIBLES ENERO 2024", "total_existencias", "TOTAL DE EXISTENCIAS DISPONIBLES", "TOTAL_DE_EXISTENCIAS"])

    if precio_col is None or exist_col is None:
        raise KeyError("Columnas de precio o existencias no encontradas en el DataFrame.")

    # limpiar numeros: quitar comas y caracteres no numéricos
    def _clean_num(series):
        s = series.astype(str).fillna("")
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'[^0-9.\-]', '', regex=True)
        return pd.to_numeric(s, errors='coerce')

    precios = _clean_num(df[precio_col])
    existencias = _clean_num(df[exist_col]).fillna(0)

    # Definir rango de precios y existencia (bins)
    bins_precios = np.linspace(precios.min(), precios.max(), 12) 
    bins_exist = np.linspace(existencias.min(), existencias.max(), 12)

    # Construir matriz 2D de frecuencias
    heatmap, xedges, yedges = np.histogram2d(precios, existencias, bins=[bins_precios, bins_exist])

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, origin='lower', aspect='auto')
    plt.colorbar(label="Cantidad de productos")
    plt.xlabel("Rangos de Precio")
    plt.ylabel("Rangos de Existencias")
    plt.title("Heatmap Precio vs Existencias")

    return fig

#Mapa de calor #2: Rangos de costos

def clasificar_precio(valor):
    if valor < 5:
        return "bajo"
    elif valor < 15:
        return "medio"
    elif valor < 50:
        return "alto"
    else:
        return "muy alto"
    
def clasificar_existencias(valor):
    if valor < 20:
        return "muy bajas"
    elif valor < 100:
        return "baja"
    elif valor < 300:
        return "altas"
    else:
        return "muy altas"
    
def heatmap_rangos(df):
    # work on a copy to avoid SettingWithCopyWarning when caller passes a slice
    df = df.copy()

    # buscar columnas posibles y limpiar
    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    precio_col = _find_col(df, ["PRECIO_UNITARIO", "precio_unitario", "PRECIO UNITARIO B/.", "PRECIO", "PRECIO_UNITARIO"])
    exist_col = _find_col(df, ["TOTAL DE EXISTENCIAS  DISPONIBLES ENERO 2024", "total_existencias", "TOTAL DE EXISTENCIAS DISPONIBLES", "TOTAL_DE_EXISTENCIAS"])

    if precio_col is None or exist_col is None:
        raise KeyError("Columnas de precio o existencias no encontradas en el DataFrame.")

    def _clean_num(series):
        s = series.astype(str).fillna("")
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'[^0-9.\-]', '', regex=True)
        return pd.to_numeric(s, errors='coerce').fillna(0)

    # assign into explicit columns using .loc to avoid chained-assignment issues
    df.loc[:, "RANGO_PRECIO"] = _clean_num(df[precio_col]).apply(clasificar_precio)
    df.loc[:, "RANGO_EXIST"] = _clean_num(df[exist_col]).apply(clasificar_existencias)

    # Matriz cruzada
    matriz = pd.crosstab(df["RANGO_PRECIO"], df["RANGO_EXIST"])

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(matriz, cmap="viridis")
    plt.colorbar(label="Cantidad de productos")

    plt.xticks(range(len(matriz.columns)), matriz.columns)
    plt.yticks(range(len(matriz.index)), matriz.index)

    plt.xlabel("Existencias")
    plt.ylabel("Precio")
    plt.title("Heatmap por Rangos de Precio y Existencias")

    return fig
