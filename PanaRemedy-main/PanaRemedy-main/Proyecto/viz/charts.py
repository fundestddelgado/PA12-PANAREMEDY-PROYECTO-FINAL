import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_violin_by_presentation(df: pd.DataFrame, med: str, desc_col: str, price_col: str, max_boxes: int = 15):
    """Return a matplotlib Figure with violin plots of `price_col` grouped by `desc_col` for entries matching `med`."""
    if desc_col not in df.columns:
        raise KeyError(f"Descripción column '{desc_col}' not in dataframe")
    if price_col not in df.columns:
        raise KeyError(f"Precio column '{price_col}' not in dataframe")

    subset = df[df[desc_col].astype(str).str.contains(med, case=False, na=False)]
    if subset.empty:
        raise ValueError("No hay registros para el medicamento solicitado")

    groups = []
    labels = []
    for desc, grp in subset.groupby(desc_col):
        s = grp[price_col].astype(str).str.replace(',', '', regex=False).str.replace(r'[^0-9.\-]', '', regex=True)
        vals = pd.to_numeric(s, errors='coerce').dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(desc))

    if not groups:
        raise ValueError("No hay valores numéricos de precio para graficar")

    # limit number of boxes
    if len(groups) > max_boxes:
        groups = groups[:max_boxes]
        labels = labels[:max_boxes]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
    parts = ax.violinplot(groups, showmeans=False, showmedians=True, vert=False)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels([lbl if len(lbl) < 40 else lbl[:37] + '...' for lbl in labels])
    ax.set_xlabel('Precio (B/.)')
    ax.set_title(f'Distribución de precios por presentación - {med}')
    ax.grid(axis='x')
    return fig


def plot_price_histogram(df: pd.DataFrame, price_col: str, bins: int = 40):
    """Return a matplotlib Figure with a histogram of `price_col`."""
    if price_col not in df.columns:
        raise KeyError(f"Precio column '{price_col}' not in dataframe")

    s = df[price_col].astype(str).str.replace(',', '', regex=False).str.replace(r'[^0-9.\-]', '', regex=True)
    vals = pd.to_numeric(s, errors='coerce').dropna().values
    if vals.size == 0:
        raise ValueError('No hay valores numéricos para el histograma')

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(vals, bins=bins, color='skyblue', edgecolor='black')
    ax.set_title('Histograma de precios')
    ax.set_xlabel('Precio (B/.)')
    ax.set_ylabel('Frecuencia')
    ax.grid(axis='y')
    return fig
