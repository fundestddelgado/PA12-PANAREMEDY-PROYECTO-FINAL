from pathlib import Path
import sys
import argparse
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

try:
    from Proyecto.viz import charts
    import Proyecto.funciones as funcs
except Exception:
    # try local imports fallback
    try:
        import viz.charts as charts
        import funciones as funcs
    except Exception:
        charts = None
        funcs = None

OUT = Path('run_tests') / 'output'
OUT.mkdir(parents=True, exist_ok=True)
CSV = Path('Proyecto/data/cleaned/cleaned_data-v1.csv')


def detect_price_col(df):
    for c in ['precio_unitario','PRECIO_UNITARIO','PRECIO','PRECIO UNITARIO B/.']:
        if c in df.columns:
            return c
    for c in df.columns:
        if 'PRECIO' in str(c).upper():
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--med', help="Nombre (o parte) del medicamento. Usa 'top' para el primer ítem del ranking.")
    args = p.parse_args()

    if charts is None:
        print('No se encontró el módulo Proyecto.viz.charts; asegúrate de que esté presente.')
        raise SystemExit(1)

    if not CSV.exists():
        print('CSV limpio no encontrado en', CSV)
        raise SystemExit(1)

    df = pd.read_csv(CSV, encoding='utf-8')

    med = args.med
    # if no med provided, ask interactively
    if not med:
        med = input(
            "Escribe el nombre (o parte) del medicamento, o 'top' para usar el top del ranking: "
        ).strip()

    if med.lower() == 'top':
        if funcs is None:
            print("No disponible funciones de ranking; pasa --med con un nombre.")
            raise SystemExit(1)
        try:
            rank = funcs.ranking_medicamentos(str(CSV), top_n=1)
            med = rank.iloc[0,0]
            print('Usando top:', med)
        except Exception as e:
            print('No se pudo obtener ranking:', e)
            raise SystemExit(1)

    price_col = detect_price_col(df)
    desc_col = None
    for c in ['DESCRIPCION_LIMPIA','DESCRIPCION_ORIG','DESCRIPCIÓN','DESCRIPCION','CODIGO']:
        if c in df.columns:
            desc_col = c
            break

    if price_col is None or desc_col is None:
        print('Columnas necesarias no encontradas en el CSV (descripcion/precio).')
        raise SystemExit(1)

    safe_med = ''.join(ch for ch in med if ch.isalnum() or ch in (' ', '_', '-')).strip().replace(' ', '_')[:50]
    try:
        print('Generando histograma para', med)
        subset = df[df[desc_col].astype(str).str.contains(med, case=False, na=False)]
        if subset.empty:
            print('No hay registros para', med, '; generando histograma global en su lugar')
            target_df = df
        else:
            target_df = subset
        fig1 = charts.plot_price_histogram(target_df, price_col)
        out1 = OUT / f'hist_{safe_med}.png'
        fig1.savefig(out1)
        print('Guardado:', out1)
    except Exception:
        traceback.print_exc()
        print('Fallo al generar histograma')

    try:
        print('Generando violin para', med)
        fig2 = charts.plot_violin_by_presentation(df, med, desc_col, price_col)
        out2 = OUT / f'violin_{safe_med}.png'
        fig2.savefig(out2)
        print('Guardado:', out2)
    except Exception:
        traceback.print_exc()
        print('Fallo al generar violin')


if __name__ == '__main__':
    main()
