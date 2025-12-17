"""Lógica de negocio para inventario: punto de reorden y cantidad a pedir."""
import numpy as np


def calcular_punto_reorden(prediccion_demanda: float, lead_time: int, stock_seguridad: float) -> float:
    return float(prediccion_demanda * lead_time + stock_seguridad)


def calcular_cantidad_a_pedir(stock_actual: int, punto_reorden: float, objetivo_maximo: int) -> int:
    if stock_actual < punto_reorden:
        falta = max(0, int(np.round(objetivo_maximo - stock_actual)))
        return falta
    return 0
