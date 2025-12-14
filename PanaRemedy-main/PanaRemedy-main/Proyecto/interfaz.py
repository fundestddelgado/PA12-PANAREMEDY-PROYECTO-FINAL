# librerias
import os 
import sys
import subprocess
import threading
import shutil
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # preferred when running as package
    import Proyecto.funciones as funcs
except ModuleNotFoundError:
    # fallback when running the script directly: import local module
    import funciones as funcs
try:
    import Proyecto.viz.charts as viz_charts
except Exception:
    try:
        import viz.charts as viz_charts
    except Exception:
        # try load from file path
        try:
            import importlib.util, sys, pathlib
            p = pathlib.Path(__file__).resolve().parent / 'viz' / 'charts.py'
            if p.exists():
                spec = importlib.util.spec_from_file_location('viz_charts_local', str(p))
                viz_charts = importlib.util.module_from_spec(spec)
                sys.modules['viz_charts_local'] = viz_charts
                spec.loader.exec_module(viz_charts)
            else:
                viz_charts = None
        except Exception:
            viz_charts = None
try:
    import Proyecto.ml.heatmap as ml_heatmap
except Exception:
    try:
        import ml.heatmap as ml_heatmap
    except Exception:
        # try load from file path
        try:
            import importlib.util, sys, pathlib
            p2 = pathlib.Path(__file__).resolve().parent / 'ml' / 'heatmap.py'
            if p2.exists():
                spec2 = importlib.util.spec_from_file_location('ml_heatmap_local', str(p2))
                ml_heatmap = importlib.util.module_from_spec(spec2)
                sys.modules['ml_heatmap_local'] = ml_heatmap
                spec2.loader.exec_module(ml_heatmap)
            else:
                ml_heatmap = None
        except Exception:
            ml_heatmap = None

def find_column_by_pattern(df: pd.DataFrame, patterns):
    
    cols = list(df.columns)
    up = [c.upper().replace(" ", "") for c in cols]
    for p in patterns:
        p_up = p.upper().replace(" ", "")
        for orig, u in zip(cols, up):
            if p_up in u:
                return orig
    return None

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Quita comas, espacios y devuelve numeric Series (coerce en errores)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("\u00A0", "", regex=False).str.strip(), errors="coerce")

def limpiar_frame(frame):
    
    for widget in frame.winfo_children():
        # Si es un canvas de Matplotlib
        if hasattr(widget, "get_tk_widget"):
            try:
                widget.get_tk_widget().destroy()
            except:
                pass
            widget.destroy()
            continue

        # Si es Treeview
        if isinstance(widget, ttk.Treeview):
            widget.destroy()
            continue

        # Otros widgets
        try:
            widget.destroy()
        except:
            pass
        
# Pagina Principal
root = tk.Tk()
root.title("PanaRemedy")
root.geometry("1000x600")
root.resizable(False, False)

# Barra superior
header_frame = tk.Frame(root, bg="#0abfbc", height=60)
header_frame.pack(fill=tk.X)

# Logo
logo_frame = tk.Frame(header_frame, bg="#0abfbc")
logo_frame.pack(side=tk.LEFT, padx=2)

ruta_logo = os.path.join(os.path.dirname(__file__),".", "assets", "logo.png")
img = Image.open(ruta_logo)
img = img.resize((40, 40))
logo_image = ImageTk.PhotoImage(img)

logo_label = tk.Label(logo_frame, image=logo_image, bg="#0abfbc")
logo_label.pack(side=tk.LEFT)

# Título
logo_text = tk.Label(
    logo_frame, 
    text="PanaRemedy", 
    font=("Arial", 18, "bold"), 
    fg="white", 
    bg="#0abfbc")
logo_text.pack(side=tk.LEFT, padx=5)

# Buscador
# Default dataset loading: prefer `long_monthly_all_dedup.csv` and expose `df` in memory
BASE_DIR = os.path.dirname(__file__)
DF_LONG_PATH = os.path.join(BASE_DIR, "data", "cleaned", "long_monthly_all_dedup.csv")
DF_WIDE_PATH = os.path.join(BASE_DIR, "data", "medicamentos.csv")

# Load long-format if available, else fall back to wide-format
def _load_default_dataset():
    global df, dataset_path
    try:
        if os.path.exists(DF_LONG_PATH):
            # load long format and keep as-is (functions now support long-format)
            df_local = pd.read_csv(DF_LONG_PATH, encoding='utf-8', low_memory=False)
            dataset_path = DF_LONG_PATH
            df = df_local
        else:
            df_local = pd.read_csv(DF_WIDE_PATH, encoding='latin1', low_memory=False)
            dataset_path = DF_WIDE_PATH
            df = df_local
    except Exception as e:
        messagebox.showerror('Error carga dataset', f'No se pudo cargar dataset por defecto: {e}')
        df = pd.DataFrame()


# run default load at startup
_load_default_dataset()


def _select_description_col(df_local: pd.DataFrame):
    """Return the best column name to use as description for this dataframe."""
    for c in ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','DESCRIPCIÓN','DESCRIPCION','CODIGO']:
        if c in df_local.columns:
            return c
    return None


def _select_price_col(df_local: pd.DataFrame):
    for c in ['precio_unitario', 'PRECIO UNITARIO B/.', 'PRECIO_UNITARIO', 'PRECIO']:
        if c in df_local.columns:
            return c
    # fallback: try any column that contains the word 'PRECIO' (case-insensitive)
    for c in df_local.columns:
        if 'PRECIO' in str(c).upper():
            return c
    return None

# Dataset selection removed: app uses the default `medicamentos.csv` loaded at startup

def buscar_medicamento():
    busqueda = entrada_busqueda.get().strip()
    if not busqueda:
        return
    # pick description column compatible with long/wide formats
    desc_col = _select_description_col(df)
    if desc_col is None:
        messagebox.showerror('Error', 'Dataset no contiene columna de descripción reconocible')
        return

    med_result = df[df[desc_col].astype(str).str.lower().str.contains(busqueda.lower(), na=False)]
    
    if not med_result.empty:
        info_med = tk.Toplevel(root)
        info_med.title(f"Información de {busqueda}")

        texto = tk.Text(info_med, height=20, width=85)
        texto.pack(padx=10, pady=10)
        
        #la informacion a mostrarse será el nombre del medicamento y su precio
        price_col = _select_price_col(df)
        for _, row in med_result.iterrows():
            texto.insert(tk.END, f"DESCRIPCIÓN: {row.get(desc_col, '')}\n")
            texto.insert(tk.END, f"PRECIO: {row.get(price_col, '')}\n\n")
        texto.config(state='disabled')
    else:
        ventana_error = tk.Toplevel(root)
        ventana_error.title("No encontrado")
        tk.Label(ventana_error, text="no se encontró el medicamento").pack(padx=20, pady=20) 

#entrada de la información
frame_busqueda = tk.Frame(header_frame, bg="#0abfbc")
frame_busqueda.pack(side=tk.RIGHT, padx=10)

entrada_busqueda = tk.Entry(frame_busqueda, width=30)
entrada_busqueda.pack(side=tk.LEFT, padx=5, pady=10)
#boton de busqueda 
btn_busqueda = tk.Button(
    frame_busqueda, 
    text="Buscar", 
    bg="white", 
    fg="#0abfbc",
    font=("Arial", 10, "bold"),
    command=buscar_medicamento
    )
btn_busqueda.pack(side=tk.LEFT, padx=5, pady=10)

# ---------- Cleaner controls ----------
# Option to purge reports after running cleaner
purge_reports_var = tk.BooleanVar(value=False)
chk_purge = tk.Checkbutton(frame_busqueda, text='Purgar reports después', variable=purge_reports_var, bg='#0abfbc')
chk_purge.pack(side=tk.LEFT, padx=5)

def _run_cleaner_worker(purge: bool):
    """Run cleaner.py in a background thread, then optionally purge reports and reload dataset."""
    try:
        cleaner_path = os.path.normpath(os.path.join(BASE_DIR, 'ml', 'cleaner.py'))
        cmd = [sys.executable, cleaner_path, '--run-all', '--input-dir', os.path.join(BASE_DIR, 'data')]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout or ''
        err = proc.stderr or ''
        # Optionally purge reports
        reports_dir = os.path.join(BASE_DIR, 'ml', 'reports')
        if purge and os.path.exists(reports_dir):
            try:
                for fname in os.listdir(reports_dir):
                    fpath = os.path.join(reports_dir, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                # small pause to ensure file system settled
                time.sleep(0.2)
            except Exception:
                pass
        # reload dataset on main thread
        try:
            root.after(100, _load_default_dataset)
        except Exception:
            pass
        # show results in a messagebox on main thread
        def _show():
            msg = 'Cleaner terminado.'
            if out:
                msg += '\n\nSalida:\n' + out[:2000]
            if err:
                msg += '\n\nErrores:\n' + err[:2000]
            messagebox.showinfo('Cleaner', msg)
        root.after(200, _show)
    except Exception as e:
        root.after(100, lambda: messagebox.showerror('Cleaner', f'Error al ejecutar cleaner: {e}'))

def run_cleaner_async():
    purge = bool(purge_reports_var.get())
    t = threading.Thread(target=_run_cleaner_worker, args=(purge,), daemon=True)
    t.start()

# Add button to trigger cleaner
btn_cleaner = tk.Button(frame_busqueda, text='Actualizar datasets', bg='white', fg='#0abfbc', command=run_cleaner_async)
btn_cleaner.pack(side=tk.LEFT, padx=5)

# (No selector) The app loads the long-format dedup dataset by default if available.

# Apartados
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

tab_inicio = ttk.Frame(notebook)
tab_reportes = ttk.Frame(notebook)
tab_graficos = ttk.Frame(notebook)
tab_pronosticos = ttk.Frame(notebook)

notebook.add(tab_inicio, text="Inicio")
notebook.add(tab_reportes, text="Análisis")
notebook.add(tab_graficos, text="Gráficas")
notebook.add(tab_pronosticos, text="Predicciones")

# Pestaña de inicio
lbl_inicio = tk.Label(tab_inicio, text="Plataforma nacional de analisis de medicamento",
                    font=("Arial", 16, "bold"))
lbl_inicio.pack(pady=20)

desc_inicio = tk.Label(
    tab_inicio,
    text="PanaRemedy te permite consultar la disponibilidad y precio de tus medicamentos en las farmacias más cercanas.",
    font=("Arial", 12),
    wraplength=500,        
    justify="center",      
    anchor="center"        
)
desc_inicio.pack(pady=20, padx=10) 

# muestra de medicamentos
def mostrar_medicamentos():
    # show currently selected dataset
    global dataset_path
    ruta_csv_local = dataset_path
    try:
        if os.path.basename(ruta_csv_local).startswith('long_monthly'):
            # show the long dedup file
            df = pd.read_csv(ruta_csv_local, encoding='utf-8', low_memory=False)
        else:
            df = pd.read_csv(ruta_csv_local, encoding='latin1', low_memory=False)
    except Exception as e:
        messagebox.showerror('Error', f'No se pudo leer el dataset: {e}')
        return
    
    # ventana con la base de los medicamentos    
    ventana_tabla = tk.Toplevel(root)
    ventana_tabla.title("Base de Datos de Medicamentos")
    ventana_tabla.geometry("800x500")
    ventana_tabla.config(bg="#ffffff")
        
    columnas = list(df.columns)
    tabla = ttk.Treeview(ventana_tabla, columns=columnas, show="headings")
    tabla.pack(fill="both", expand=True)
    
    for col in columnas:
        tabla.heading(col, text=col)
        tabla.column(col, width=120, anchor="center")
    
    for _, fila in df.iterrows():
        tabla.insert("","end", values=list(fila))
        
# boton para ver los medicamentos
btn_medicamentos = tk.Button(
    tab_inicio,
    text="Medicamentos",
    bg="#0abfbc",
    fg="white", 
    font=("Arial", 12),
    command=mostrar_medicamentos
    )
btn_medicamentos.pack(pady=20)

# ----------------- Canvases y controles embebidos -----------------
canvases = {}

# --- Inicio: ranking ---
frame_inicio_left = tk.Frame(tab_inicio, width=280)
frame_inicio_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
frame_inicio_right = tk.Frame(tab_inicio, bg='white')
frame_inicio_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(frame_inicio_left, text='Ranking Top N:', font=(None, 10)).pack(anchor='w')
spin_rank = tk.Spinbox(frame_inicio_left, from_=1, to=50, width=5)
spin_rank.pack(pady=6)

def generar_ranking_tab():
    try:
        top_n = int(spin_rank.get() or 10)
    except Exception:
        top_n = 10
    try:
        ranking = funcs.ranking_medicamentos(dataset_path, top_n=top_n)
    except Exception as e:
        messagebox.showerror('Error', f'No se pudo obtener ranking: {e}')
        return
    # crear figura local para mostrar
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(7,4))
    # mostrar sólo nombre corto (antes de la coma) para las etiquetas
    labels_full = ranking.iloc[:,0].astype(str).tolist()
    try:
        labels = [funcs.short_name(l) for l in labels_full]
    except Exception:
        labels = labels_full
    values = ranking.iloc[:,1].tolist()
    y = list(range(len(labels)))
    ax.barh(y, values, color='skyblue')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f'Top {top_n} por existencias')
    ax.set_xlabel('Existencias')

    # embed
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # remove previous
    prev = canvases.get("inicio")
    if prev:
        try:
            prev.get_tk_widget().destroy()
        except:
            try:
                prev.destroy()
            except:
                pass

    canvas = FigureCanvasTkAgg(fig, master=frame_inicio_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['inicio'] = canvas
    try:
        _plt.close(fig)
    except Exception:
        pass

btn_rank_tab = tk.Button(frame_inicio_left, text='Generar Ranking', command=generar_ranking_tab)
btn_rank_tab.pack(fill='x', pady=6)

# --- RECOMENDACIÓN (TF-IDF + cosine) ---
tk.Label(frame_inicio_left, text="Recomendación (nombre limpio):").pack(anchor="w", pady=(12,0))
entrada_reco = tk.Entry(frame_inicio_left, width=28)
entrada_reco.pack(pady=6)

def recomendar_similares_ui(descripcion: str, top_k:int=5):
    """
    Recomienda top_k medicamentos similares usando TF-IDF sobre DESCRIPCIÓN_LIMPIA o DESCRIPCIÓN.
    Retorna DataFrame con resultados.
    """
    if df is None or df.empty:
        raise ValueError("Dataset no cargado.")
    texto_col = None
    for c in ["DESCRIPCIÓN_LIMPIA", "DESCRIPCION_LIMPIA", "DESCRIPCIÓN", "DESCRIPCION"]:
        if c in df.columns:
            texto_col = c
            break
    if texto_col is None:
        raise KeyError("No se encontró columna de texto para recomendaciones.")

    corpus = df[texto_col].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    matriz = vectorizer.fit_transform(corpus)
    consulta = vectorizer.transform([descripcion])
    sims = cosine_similarity(consulta, matriz).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    return df.iloc[idxs].copy()

def ejecutar_recomendacion_ui():
    q = entrada_reco.get().strip()
    if not q:
        messagebox.showinfo("Recomendación", "Escribe el nombre limpio (DESCRIPCIÓN_LIMPIA) o parte del nombre.")
        return
    try:
        resultados = recomendar_similares_ui(q, top_k=6)
    except Exception as e:
        messagebox.showerror("Recomendación", str(e))
        return

    # Mostrar resultados como tabla embebida en frame_inicio_right (reemplaza ranking si existe)
    prev = canvases.get("inicio")
    if prev:
        prev.get_tk_widget().destroy()

    # Construir Treeview en frame_inicio_right
    cols = list(resultados.columns)
    tabla_win = ttk.Treeview(frame_inicio_right, columns=cols, show="headings")
    for col in cols:
        tabla_win.heading(col, text=col)
        tabla_win.column(col, width=120, anchor="center")
    for _, row in resultados.iterrows():
        tabla_win.insert("", "end", values=list(row))
    tabla_win.pack(fill=tk.BOTH, expand=True)
    canvases["inicio"] = tabla_win  # guardamos widget en vez de FigureCanvas

btn_reco = tk.Button(frame_inicio_left, text="Recomendar Similares", command=ejecutar_recomendacion_ui, bg="#0abfbc", fg="white")
btn_reco.pack(fill="x", pady=6)

btn_limpiar_inicio = tk.Button(
    frame_inicio_left,
    text="Limpiar Resultados",
    command=lambda: limpiar_frame(frame_inicio_right),
    bg="#d9534f",
    fg="white"
)
btn_limpiar_inicio.pack(fill="x", pady=6)

# --- Análisis: comparación ---
frame_analisis_left = tk.Frame(tab_reportes, width=280)
frame_analisis_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
frame_analisis_right = tk.Frame(tab_reportes, bg='white')
frame_analisis_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(frame_analisis_left, text='Medicamento (comparación):').pack(anchor='w')
entrada_med_comp = tk.Entry(frame_analisis_left, width=30)
entrada_med_comp.pack(pady=6)

def generar_comparacion_tab():
    med = entrada_med_comp.get().strip()
    try:
        if med:
            series = funcs.compare_farmacias(df, med)
            title = f'Comparación por almacenes - {med}'
        else:
            series = funcs.compare_medicamentos(df)
            title = 'Comparación de medicamentos (precio medio)'
    except Exception as e:
        messagebox.showerror('Error', f'Error al calcular comparación: {e}')
        return
    if series is None or series.empty:
        messagebox.showinfo('Sin datos', 'No hay datos para la comparación')
        return
    # usar plot_comparison para dibujar
    try:
        _plt = funcs.plot_comparison(series, title=title)
        fig = _plt.gcf()
    except Exception as e:
        messagebox.showerror('Error dibujo', str(e))
        return
    prev = canvases.get('analisis')
    if prev:
        prev.get_tk_widget().destroy()
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    canvas = FigureCanvasTkAgg(fig, master=frame_analisis_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['analisis'] = canvas
    try:
        _plt.close(fig)
    except Exception:
        pass

btn_comp_tab = tk.Button(frame_analisis_left, text='Generar Comparación', command=generar_comparacion_tab)
btn_comp_tab.pack(fill='x', pady=6)

# --- Detección de anomalías ---
# Helpers para detección (autónomos y robustos)
def entrenar_modelo_regresion(df_local: pd.DataFrame):

    # detectar columnas candidatas
    col_exist = find_column_by_pattern(df_local, ["TOTALDEEXISTENCIAS", "TOTAL DE EXISTENCIAS", "EXISTENCIA", "EXISTENCIAS"])
    col_prec = find_column_by_pattern(df_local, ["PRECIO_UNITARIO", "PRECIOUNITARIO", "PRECIO UNITARIO", "PRECIO"])
    if col_exist is None or col_prec is None:
        raise KeyError(f"No se encontraron columnas de existencias ({col_exist}) o precio ({col_prec}).")
    Xs = clean_numeric_series(df_local[col_exist]).fillna(0.0).values.reshape(-1, 1)
    ys = clean_numeric_series(df_local[col_prec]).fillna(0.0).values
    model = LinearRegression()
    # proteger contra datasets vacíos
    if Xs.size == 0 or ys.size == 0 or len(Xs) != len(ys):
        raise ValueError("Datos insuficientes o desalineados para entrenar el modelo.")
    model.fit(Xs, ys)
    return model, col_exist, col_prec

def detectar_anomalias_df(df_local: pd.DataFrame, modelo: LinearRegression, col_exist: str, col_prec: str, umbral: float=0.30):
    """Detecta anomalías comparando precio real vs predicho (umbral relativo).
    Retorna (anomalos_df, figura_matplotlib)
    """
    dfc = df_local.copy()
    dfc["_exist_num"] = clean_numeric_series(dfc[col_exist]).fillna(0.0)
    dfc["_precio_num"] = clean_numeric_series(dfc[col_prec]).fillna(0.0)
    # predicción (manejo de ceros/NaN)
    preds = modelo.predict(dfc[["_exist_num"]].values.reshape(-1, 1))
    dfc["_pred"] = preds
    # evitar división por cero al calcular desviación relativa: si pred == 0 -> usar denom = 1e-9
    denom = np.where(np.abs(dfc["_pred"].values) < 1e-9, 1e-9, np.abs(dfc["_pred"].values))
    dfc["_rel_dev"] = np.abs(dfc["_precio_num"].values - dfc["_pred"].values) / denom
    dfc["_anomaly"] = dfc["_rel_dev"] > umbral
    anomalos = dfc[dfc["_anomaly"]].copy()

    # preparar figura: scatter con linea de regresión
    fig = Figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.scatter(dfc["_exist_num"], dfc["_precio_num"], label="Datos reales", alpha=0.6)
    # ordenar por existencias para trazar línea
    order = np.argsort(dfc["_exist_num"].values)
    ax.plot(dfc["_exist_num"].values[order], dfc["_pred"].values[order], color="red", linewidth=2, label="Predicción (regresión)")
    # marcar anomalías
    if not anomalos.empty:
        ax.scatter(anomalos["_exist_num"], anomalos["_precio_num"], facecolors='none', edgecolors='k', marker='o', s=80, label="Anomalías")
    ax.set_xlabel("Existencias")
    ax.set_ylabel("Precio")
    ax.set_title("Precio real vs predicho (anomalías marcadas)")
    ax.legend()
    fig.tight_layout()
    return anomalos, fig

def ejecutar_anomalias_ui():
    """Función ligada al botón: entrena modelo si es necesario, detecta anomalías y muestra tabla + gráfico en frame_analisis_right."""
    # entrenar modelo (usar dataset completo)
    try:
        modelo_local, col_exist, col_prec = entrenar_modelo_regresion(df)
    except Exception as e:
        messagebox.showerror("Anomalías", f"No se pudo entrenar el modelo: {e}")
        return
    # obtener umbral desde UI (si deseas permitir ajuste, reemplazar con Scale)
    umbral = 0.30  # por defecto 30%
    try:
        anomalos_df, fig = detectar_anomalias_df(df, modelo_local, col_exist, col_prec, umbral=umbral)
    except Exception as e:
        messagebox.showerror("Anomalías", f"Error al detectar anomalías: {e}")
        return
    # Mostrar tabla en el mismo recuadro derecho (frame_analisis_right)
    for w in frame_analisis_right.winfo_children():
        w.destroy()  # limpiar contenido previo en el recuadro
    cols_show = [c for c in anomalos_df.columns if not c.startswith("_")] + ["_rel_dev"]
    # si no hay anomalías, informar
    if anomalos_df.empty:
        tk.Label(frame_analisis_right, text="No se detectaron anomalías con el umbral actual.", font=("Arial", 12)).pack(padx=10, pady=10)
    else:
        # Treeview para anomalías
        tv = ttk.Treeview(frame_analisis_right, columns=cols_show, show="headings")
        tv.pack(fill=tk.BOTH, expand=True)
        for c in cols_show:
            tv.heading(c, text=c)
            tv.column(c, width=120, anchor="center")
        for _, row in anomalos_df.iterrows():
            vals = [row.get(c, "") for c in cols_show]
            tv.insert("", "end", values=vals)
    # Agregar gráfico debajo (o en una nueva ventana)
    canvas = FigureCanvasTkAgg(fig, master=frame_analisis_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['analisis'] = canvas
    plt.close(fig)

# Button for anomaly detection under comparison controls
btn_anom = tk.Button(frame_analisis_left, text="Detectar Anomalías (ML)", command=ejecutar_anomalias_ui, bg="#0abfbc", fg="white")
btn_anom.pack(fill="x", pady=(6,12))

btn_limpiar_analisis = tk.Button(
    frame_analisis_left,
    text="Limpiar Resultados",
    command=lambda: limpiar_frame(frame_analisis_right),
    bg="#d9534f",
    fg="white"
)
btn_limpiar_analisis.pack(fill="x", pady=6)

# --- Gráficas: tendencia ---
frame_graf_left = tk.Frame(tab_graficos, width=280)
frame_graf_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
frame_graf_right = tk.Frame(tab_graficos, bg='white')
frame_graf_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(frame_graf_left, text='Medicamento (tendencia):').pack(anchor='w')
entrada_med_tend = tk.Entry(frame_graf_left, width=30)
entrada_med_tend.pack(pady=6)

def generar_tendencia_tab():
    med = entrada_med_tend.get().strip()
    if not med:
        messagebox.showinfo('Entrada requerida', 'Escribe el nombre (o parte) del medicamento para la tendencia')
        return
    try:
        series = funcs.price_trend(df, med)
    except Exception as e:
        messagebox.showerror('Error', f'Error al obtener tendencia: {e}')
        return
    if series is None or series.empty:
        messagebox.showinfo('Sin datos', f'No se encontraron datos para: {med}')
        return
    try:
        _plt = funcs.plot_trend(series, med)
        fig = _plt.gcf()
    except Exception as e:
        messagebox.showerror('Error dibujo', str(e))
        return
    prev = canvases.get('graf')
    if prev:
        prev.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['graf'] = canvas
    try:
        _plt.close(fig)
    except Exception:
        pass

btn_tend_tab = tk.Button(frame_graf_left, text='Generar Tendencia', command=generar_tendencia_tab)
btn_tend_tab.pack(fill='x', pady=6)

# --- Gráficas: comparación de precios por presentación (boxplot) ---
tk.Label(frame_graf_left, text='Medicamento (comparar precios por presentación):').pack(anchor='w')
entrada_med_pricecomp = tk.Entry(frame_graf_left, width=30)
entrada_med_pricecomp.pack(pady=6)

def generar_price_comparison_tab():
    med = entrada_med_pricecomp.get().strip()
    if not med:
        messagebox.showinfo('Entrada requerida', 'Escribe el nombre (o parte) del medicamento para comparar precios')
        return
    try:
        desc_col = _select_description_col(df)
        if desc_col is None:
            messagebox.showerror('Error', 'Dataset no contiene columna de descripción reconocible')
            return
        subset = df[df[desc_col].astype(str).str.contains(med, case=False, na=False)]
    except Exception as e:
        messagebox.showerror('Error', f'Error al filtrar dataset: {e}')
        return
    if subset.empty:
        messagebox.showinfo('Sin datos', f'No se encontraron registros para: {med}')
        return

    # Agrupar por presentación y reunir los precios por grupo
    price_col = _select_price_col(df)
    if price_col is None:
        messagebox.showerror('Error', 'No se encontró columna de precio en el dataset')
        return
    groups = []
    labels = []
    for desc, grp in subset.groupby(desc_col):
        # limpiar valores numéricos similares a funciones._clean_numeric_str
        s = grp[price_col].astype(str).str.replace('"', '', regex=False).str.replace("'", '', regex=False)
        s = s.str.replace('\u00A0', '', regex=False).str.strip()
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'[^0-9.\-]', '', regex=True)
        vals = pd.to_numeric(s, errors='coerce').dropna().values
        if len(vals) > 1:
            groups.append(vals.tolist())
            try:
                labels.append(funcs.short_name(desc))
            except Exception:
                labels.append(str(desc))

    if not groups:
        messagebox.showinfo('Sin muestras múltiples', 'No se encontraron presentaciones con más de 1 precio; muchos medicamentos tienen solo 1 registro de precio.')
        return

    # limitar número de cajas para mantener legibilidad
    max_boxes = 15
    if len(groups) > max_boxes:
        groups = groups[:max_boxes]
        labels = labels[:max_boxes]

    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
    # boxplot horizontal para etiquetas en eje y
    ax.boxplot(groups, vert=False, labels=labels, patch_artist=True)
    ax.set_title(f'Comparación de precios por presentación - {med}')
    ax.set_xlabel('Precio (B/.)')
    ax.grid(axis='x')

    prev = canvases.get('pricecomp')
    if prev:
        prev.get_tk_widget().destroy()
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['pricecomp'] = canvas
    try:
        _plt.close(fig)
    except Exception:
        pass

btn_pricecomp = tk.Button(frame_graf_left, text='Generar Comparación de Precios', command=generar_price_comparison_tab)
btn_pricecomp.pack(fill='x', pady=6)

# --- Gráficas adicionales: Violin y Histograma ---
tk.Label(frame_graf_left, text='Violin: distribución por presentación (medicamento):').pack(anchor='w')
entrada_med_violin = tk.Entry(frame_graf_left, width=30)
entrada_med_violin.pack(pady=6)

def generar_violin_tab():
    if viz_charts is None:
        messagebox.showerror('Error', 'Módulo de visualización no disponible')
        return
    med = entrada_med_violin.get().strip()
    if not med:
        messagebox.showinfo('Entrada requerida', 'Escribe el nombre (o parte) del medicamento para violin')
        return
    try:
        desc_col = _select_description_col(df)
        price_col = _select_price_col(df)
        if desc_col is None or price_col is None:
            messagebox.showerror('Error', 'No se encontraron columnas necesarias en el dataset')
            return
        fig = viz_charts.plot_violin_by_presentation(df, med, desc_col, price_col)
    except Exception as e:
        messagebox.showerror('Error', f'No se pudo generar violin: {e}')
        return
    prev = canvases.get('violin')
    if prev:
        prev.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['violin'] = canvas
    try:
        plt.close(fig)
    except Exception:
        pass

btn_violin = tk.Button(frame_graf_left, text='Generar Violin', command=generar_violin_tab)
btn_violin.pack(fill='x', pady=6)

tk.Label(frame_graf_left, text='Histograma: precios (global)').pack(anchor='w')
def generar_histogram_tab():
    if viz_charts is None:
        messagebox.showerror('Error', 'Módulo de visualización no disponible')
        return
    try:
        price_col = _select_price_col(df)
        if price_col is None:
            messagebox.showerror('Error', 'No se encontró columna de precio')
            return
        fig = viz_charts.plot_price_histogram(df, price_col)
    except Exception as e:
        messagebox.showerror('Error', f'No se pudo generar histograma: {e}')
        return
    prev = canvases.get('hist')
    if prev:
        prev.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['hist'] = canvas
    try:
        plt.close(fig)
    except Exception:
        pass

btn_hist = tk.Button(frame_graf_left, text='Generar Histograma', command=generar_histogram_tab)
btn_hist.pack(fill='x', pady=6)

# --- Heatmap controls ---
tk.Label(frame_graf_left, text='Heatmap: Precio vs Existencias (global o por medicamento)').pack(anchor='w')
entrada_med_heat = tk.Entry(frame_graf_left, width=30)
entrada_med_heat.pack(pady=6)

def generar_heatmap_precio_existencia_tab():
    if ml_heatmap is None:
        messagebox.showerror('Error', 'Módulo heatmap no disponible')
        return
    med = entrada_med_heat.get().strip()
    try:
        # choose df subset if med provided
        if med:
            desc_col = _select_description_col(df)
            subset = df[df[desc_col].astype(str).str.contains(med, case=False, na=False)] if desc_col else df
            target = subset if not subset.empty else df
        else:
            target = df
        fig = ml_heatmap.heatmap_precio_existencia(target)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_path = pathlib.Path('run_tests') / 'heatmap_error.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as fh:
            fh.write(tb)
        messagebox.showerror('Error heatmap', f'Error al generar heatmap. Traceguard saved to {log_path}')
        return
    prev = canvases.get('heatmap1')
    if prev:
        prev.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['heatmap1'] = canvas
    try:
        plt.close(fig)
    except Exception:
        pass

btn_heat1 = tk.Button(frame_graf_left, text='Heatmap Precio-Existencia', command=generar_heatmap_precio_existencia_tab)
btn_heat1.pack(fill='x', pady=6)

def generar_heatmap_rangos_tab():
    if ml_heatmap is None:
        messagebox.showerror('Error', 'Módulo heatmap no disponible')
        return
    med = entrada_med_heat.get().strip()
    try:
        if med:
            desc_col = _select_description_col(df)
            subset = df[df[desc_col].astype(str).str.contains(med, case=False, na=False)] if desc_col else df
            target = subset if not subset.empty else df
        else:
            target = df
        fig = ml_heatmap.heatmap_rangos(target)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_path = pathlib.Path('run_tests') / 'heatmap_error.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as fh:
            fh.write(tb)
        messagebox.showerror('Error heatmap', f'Error al generar heatmap. Traceguard saved to {log_path}')
        return
    prev = canvases.get('heatmap2')
    if prev:
        prev.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_graf_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['heatmap2'] = canvas
    try:
        plt.close(fig)
    except Exception:
        pass

btn_heat2 = tk.Button(frame_graf_left, text='Heatmap Rangos Precio-Exist', command=generar_heatmap_rangos_tab)
btn_heat2.pack(fill='x', pady=6)

btn_limpiar_graficas = tk.Button(
    frame_graf_left,
    text="Limpiar Resultados",
    command=lambda: limpiar_frame(frame_graf_right),
    bg="#d9534f",
    fg="white"
)
btn_limpiar_graficas.pack(fill="x", pady=6)

# ------------------------------------------------------------------

# ---------------- Predicciones / Pronósticos ----------------
frame_pred_left = tk.Frame(tab_pronosticos, width=280)
frame_pred_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
frame_pred_right = tk.Frame(tab_pronosticos, bg='white')
frame_pred_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(frame_pred_left, text='Medicamento (pronóstico):').pack(anchor='w')
entrada_med_pred = tk.Entry(frame_pred_left, width=30)
entrada_med_pred.pack(pady=6)

tk.Label(frame_pred_left, text='Horizonte (pasos):').pack(anchor='w')
spin_horizon = tk.Spinbox(frame_pred_left, from_=1, to=52, width=6)
spin_horizon.pack(pady=6)

tk.Label(frame_pred_left, text='MA ventana (window):').pack(anchor='w')
spin_window = tk.Spinbox(frame_pred_left, from_=1, to=12, width=6)
spin_window.pack(pady=6)

tk.Label(frame_pred_left, text='Método:').pack(anchor='w')
metodo_var = tk.StringVar(value='both')
tk.Radiobutton(frame_pred_left, text='Lineal', variable=metodo_var, value='linear').pack(anchor='w')
tk.Radiobutton(frame_pred_left, text='MA (media móvil)', variable=metodo_var, value='ma').pack(anchor='w')
tk.Radiobutton(frame_pred_left, text='Ambos', variable=metodo_var, value='both').pack(anchor='w')

metrics_text = tk.Text(frame_pred_left, height=8, width=32)
metrics_text.pack(pady=6)

def generar_pronostico_tab():
    med = entrada_med_pred.get().strip()
    if not med:
        messagebox.showinfo('Entrada requerida', 'Escribe el nombre (o parte) del medicamento para pronosticar')
        return
    try:
        horizon = int(spin_horizon.get() or 6)
    except Exception:
        horizon = 6
    try:
        window = int(spin_window.get() or 3)
    except Exception:
        window = 3

    metodo = metodo_var.get()
    metrics_text.delete('1.0', tk.END)
    # compute observed series once and validate
    try:
        series = funcs.price_trend(df, med)
    except Exception as e:
        messagebox.showerror('Error', f'Error al obtener datos: {e}')
        return

    if series is None or series.empty:
        messagebox.showinfo('Sin datos', f'No se encontraron datos para: {med}')
        return

    # prepare recent observed values for display
    recent_vals = []
    last_price = None
    try:
        recent = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna().reset_index(drop=True)
        recent_vals = recent.tail(3).tolist()
        last_price = recent_vals[-1] if len(recent_vals) > 0 else None
    except Exception:
        recent_vals = []
        last_price = None

    # prepare numeric observed series and validate parameters
    try:
        y = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna().reset_index(drop=True).values
        n = len(y)
    except Exception:
        y = None
        n = 0

    # ensure window is a sensible integer relative to series length
    try:
        window = int(window)
    except Exception:
        window = 3
    if n > 0:
        window = max(1, min(window, n))

    # call forecasting functions separately and capture errors per method
    figs = []
    legends = []
    forecast_errors = []

    if metodo in ('linear', 'both'):
        try:
            fore_lin, metrics_lin, fig_lin, test_lin = funcs.forecast_linear(df, med, horizon=horizon, test_size=0.2)
            figs.append((fig_lin, 'Linear'))
            legends.append(('Linear', metrics_lin))
        except Exception as e:
            forecast_errors.append(f"Linear: {e}")

    if metodo in ('ma', 'both'):
        try:
            fore_ma, metrics_ma, fig_ma, test_ma = funcs.forecast_ma(df, med, horizon=horizon, window=window, test_size=0.2)
            figs.append((fig_ma, 'MA'))
            legends.append(('MA', metrics_ma))
        except Exception as e:
            forecast_errors.append(f"MA: {e}")

    if forecast_errors:
        messagebox.showwarning('Aviso pronóstico', 'Algunos métodos fallaron:\n' + '\n'.join(forecast_errors))

    # build combined plot on the right canvas using the observed series
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(7,4))

    try:
        y = pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').dropna().reset_index(drop=True).values
        n = len(y)
    except Exception:
        y = None
        n = 0

    if y is not None and n > 0:
        ax.plot(np.arange(n), y, label='observed', color='black')

    colors = {'Linear': 'tab:blue', 'MA': 'tab:orange'}
    for fig_obj, tag in figs:
        try:
            for line in fig_obj.axes[0].get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                lbl = line.get_label()
                if lbl == 'observed':
                    continue
                ax.plot(xdata, ydata, label=f"{tag}: {lbl}", color=colors.get(tag, None))
        except Exception:
            continue

    ax.set_title(f'Pronóstico - {med}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Precio (B/.)')
    ax.legend()
    ax.grid(True)

    prev = canvases.get('pred')
    if prev:
        prev.get_tk_widget().destroy()
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    canvas = FigureCanvasTkAgg(fig, master=frame_pred_right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvases['pred'] = canvas

    if recent_vals:
        metrics_text.insert(tk.END, f"Últimos precios (últimos {len(recent_vals)}): {', '.join([str(round(v,2)) for v in recent_vals])}\n")
        metrics_text.insert(tk.END, f"Último precio conocido: {round(last_price,2) if last_price is not None else 'N/A'}\n\n")

    # recolección de pronósticos
    forecasts = {}
    if 'fore_lin' in locals():
        try:
            forecasts['Linear'] = fore_lin['y_pred'].tolist()
        except Exception:
            forecasts['Linear'] = []
    if 'fore_ma' in locals():
        try:
            forecasts['MA'] = fore_ma['y_pred'].tolist()
        except Exception:
            forecasts['MA'] = []

    for name, vals in forecasts.items():
        if vals:
            vals_rounded = [round(float(x),2) for x in vals]
            metrics_text.insert(tk.END, f"{name} - Pronóstico próximos {len(vals_rounded)}: {vals_rounded}\n")
            metrics_text.insert(tk.END, f"{name} - Precio siguiente: {vals_rounded[0]}\n\n")
        else:
            metrics_text.insert(tk.END, f"{name}: sin pronóstico disponible\n\n")

    # MAE, RMSE
    for name, met in legends:
        metrics_text.insert(tk.END, f"{name}:\n")
        metrics_text.insert(tk.END, f"  MAE: {met.get('mae')}\n")
        metrics_text.insert(tk.END, f"  RMSE: {met.get('rmse')}\n\n")

    try:
        _plt.close(fig)
        for fobj, _ in figs:
            try:
                _plt.close(fobj)
            except Exception:
                pass
    except Exception:
        pass

btn_pred_tab = tk.Button(frame_pred_left, text='Generar Pronóstico', command=generar_pronostico_tab)
btn_pred_tab.pack(fill='x', pady=6)

btn_limpiar_pred = tk.Button(
    frame_pred_left,
    text="Limpiar Resultados",
    command=lambda: limpiar_frame(frame_pred_right),
    bg="#d9534f",
    fg="white"
)
btn_limpiar_pred.pack(fill="x", pady=6)

# pie de pagina
footer = tk.Label(root, text="2025 PanaRemedy",font=("Arial", 10), fg="#666")
footer.pack(side=tk.BOTTOM, pady=10)

# ejecutar la aplicación
root.mainloop()