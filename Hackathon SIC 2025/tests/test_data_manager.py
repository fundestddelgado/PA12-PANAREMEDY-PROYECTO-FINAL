import pandas as pd
from src.data_manager import generar_dataset_simulado, cargar_y_limpiar_datos, resample_datos, get_estadisticas_basicas


def test_generar_y_cargar(tmp_path):
    out = tmp_path / "sales_sample.csv"
    # generar dataset simulado
    path = generar_dataset_simulado(str(out), start='2023-01-01', days=60, sku='SKU_TEST')
    assert out.exists()

    # cargar y limpiar
    df = cargar_y_limpiar_datos(str(out))
    assert 'date' in df.columns or 'fecha' in df.columns
    assert 'sales' in df.columns
    assert (df['sales'] >= 0).all()


def test_resample_y_estadisticas(tmp_path):
    # construir pequeño dataframe diario
    dates = pd.date_range(start='2023-01-01', periods=14, freq='D')
    rows = [{'date': d, 'sku': 'SKU_X', 'sales': i % 5 + 1} for i, d in enumerate(dates)]
    df = pd.DataFrame(rows)

    # resample semanal
    res = resample_datos(df, frecuencia='W')
    assert 'sales' in res.columns
    # should have less or equal rows than original
    assert len(res) <= len(df)

    stats = get_estadisticas_basicas(df)
    assert set(stats.keys()) >= {'mean', 'std', 'min', 'max'}
    assert isinstance(stats['mean'], float)
