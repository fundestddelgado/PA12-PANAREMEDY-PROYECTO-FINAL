"""Cálculo de punto de reorden y stock de seguridad."""
import numpy as np
import pandas as pd


def safety_stock(std_dev_demand, lead_time_weeks=2, z_score=1.65):
    # std_dev_demand: std dev of demand per period (weekly)
    return z_score * std_dev_demand * np.sqrt(lead_time_weeks)


def reorder_point(forecast_qty, std_dev_demand, lead_time_weeks=2, z_score=1.65):
    ss = safety_stock(std_dev_demand, lead_time_weeks, z_score)
    return np.maximum(0, forecast_qty * lead_time_weeks + ss)


def suggest_order(current_stock, reorder_point_qty, predicted_consumption_lead_time):
    # suggested order quantity to reach reorder point + predicted consumption for the lead time
    needed = reorder_point_qty - current_stock
    return int(max(0, np.round(needed)))
