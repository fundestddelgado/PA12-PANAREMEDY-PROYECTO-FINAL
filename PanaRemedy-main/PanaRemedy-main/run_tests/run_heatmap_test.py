from pathlib import Path
import sys
import traceback

# ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from Proyecto.ml import heatmap
    import pandas as pd
except Exception as e:
    print('Failed to import heatmap module:', e)
    raise

out_dir = Path('run_tests') / 'output'
out_dir.mkdir(parents=True, exist_ok=True)
csv = Path('Proyecto/data/cleaned/cleaned_data-v1.csv')
print('Using CSV:', csv, 'exists=', csv.exists())

try:
    df = heatmap.cargar_dataset(str(csv))
    fig1 = heatmap.heatmap_precio_existencia(df)
    fig1.savefig(out_dir / 'heatmap_precio_existencia.png')
    print('Saved heatmap_precio_existencia.png')
    fig2 = heatmap.heatmap_rangos(df)
    fig2.savefig(out_dir / 'heatmap_rangos.png')
    print('Saved heatmap_rangos.png')
    print('Heatmap tests completed successfully.')
except Exception:
    traceback.print_exc()
    print('Heatmap tests failed. See traceback above.')
