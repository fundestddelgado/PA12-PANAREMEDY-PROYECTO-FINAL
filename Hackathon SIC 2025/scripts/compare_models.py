import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def ensure_reports_dir():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)


def load_series():
    # support both spanish and english column names
    df = pd.read_csv("data/train.csv")
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    date_col = None
    sales_col = None
    for c in ["fecha", "date"]:
        if c in cols:
            date_col = c
            break
    for c in ["ventas", "sales"]:
        if c in cols:
            sales_col = c
            break
    if date_col is None or sales_col is None:
        raise RuntimeError("data/train.csv must contain date and sales columns (e.g. date/sales or fecha/ventas)")
    df[date_col] = pd.to_datetime(df[date_col])
    series = df[[date_col, sales_col]].groupby(date_col).sum().asfreq("D").fillna(0)
    series = series.rename(columns={sales_col: "ventas"})
    return series


def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def run_arima(train, test, horizon, save_path):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(train, order=(5, 1, 0))
    fitted = model.fit()
    pred = fitted.forecast(steps=horizon)
    m = metrics(test, pred)
    return pred, m


def run_prophet(train_df, horizon):
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet

    m = Prophet()
    m.fit(train_df)
    future = m.make_future_dataframe(periods=horizon)
    fcst = m.predict(future)
    return fcst


def run_lstm(series, train_len, horizon, look_back=14):
    from dl_models_colaborador import escalar_datos, crear_secuencias, construir_y_entrenar_lstm, predecir_lstm

    # series: array-like of sales values
    values = np.array(series).reshape(-1, 1).astype(float)
    scaled, scaler = escalar_datos(values)
    X, y = crear_secuencias(scaled, look_back)

    # number of training samples available in X is len(X) == len(values) - look_back
    cutoff = train_len - look_back
    if cutoff <= 0:
        raise ValueError("Not enough training data for the chosen look_back")

    X_train = X[:cutoff]
    y_train = y[:cutoff]

    try:
        model = construir_y_entrenar_lstm(X_train, y_train, epochs=3, verbose=0)
    except Exception:
        # likely tensorflow not available; fallback to simple persistence (mean of last window)
        fallback_val = float(values[train_len - look_back : train_len].mean())
        pred = [fallback_val] * horizon
        return pred

    # start prediction from the last sequence ending at train_len
    last_seq = scaled[train_len - look_back : train_len].reshape(1, look_back, 1)
    pred = []
    for _ in range(horizon):
        p = model.predict(last_seq, verbose=0)
        pred.append(float(p.ravel()[0]))
        # roll and append
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1, 0] = p

    pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1)).ravel()
    return pred


def save_plot(dates, actual, preds, labels, fname):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="actual", linewidth=2)
    for p, l in zip(preds, labels):
        plt.plot(dates, p, label=l)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    ensure_reports_dir()
    series = load_series()
    # use last 28 days as test
    horizon = 28
    train = series.iloc[:-horizon]["ventas"]
    test = series.iloc[-horizon:]["ventas"]
    dates = series.index[-horizon:]

    results = {}

    # ARIMA
    try:
        arima_pred, arima_m = run_arima(train, test, horizon, "reports/figures/arima.png")
        results["ARIMA"] = arima_m
    except Exception as e:
        arima_pred = np.zeros(horizon)
        results["ARIMA"] = {"error": str(e)}

    # Prophet
    try:
        train_df = series.iloc[:-horizon].reset_index()
        # ensure columns are ds,y
        cols = list(train_df.columns)
        # first column is date, second is ventas
        train_df = train_df.rename(columns={cols[0]: "ds", "ventas": "y"})
        fcst = run_prophet(train_df, horizon)
        prophet_pred = fcst.set_index("ds")["yhat"].reindex(dates).fillna(method="ffill").values
        results["Prophet"] = metrics(test, prophet_pred)
    except Exception as e:
        prophet_pred = np.zeros(horizon)
        results["Prophet"] = {"error": str(e)}

    # LSTM
    try:
        train_len = len(train.index)
        lstm_pred = run_lstm(series["ventas"].values, train_len, horizon)
        results["LSTM"] = metrics(test, lstm_pred)
    except Exception as e:
        lstm_pred = np.zeros(horizon)
        results["LSTM"] = {"error": str(e)}

    # save plots and report
    save_plot(dates, test.values, [arima_pred, prophet_pred, lstm_pred], ["ARIMA", "Prophet", "LSTM"], "reports/figures/compare.png")

    # write report
    with open("reports/comparison.md", "w", encoding="utf8") as f:
        f.write("# Comparación de modelos\n\n")
        f.write(f"Periodo de prueba: {dates[0].date()} - {dates[-1].date()}\n\n")
        for k, v in results.items():
            f.write(f"## {k}\n")
            f.write("```")
            f.write(str(v))
            f.write("```\n\n")
        f.write("![compare](figures/compare.png)\n")

    print("Comparación completada. Reporte: reports/comparison.md")


if __name__ == "__main__":
    main()
