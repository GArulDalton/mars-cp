"""Build (y, yhat) streams for real datasets using a sliding-refit base forecaster.
Base model: HistGradientBoostingRegressor refit every R steps on a trailing window."""
import numpy as np, pandas as pd, os, pickle
from sklearn.ensemble import HistGradientBoostingRegressor

from paths import DATA
D = str(DATA)

def lagframe(y, lags, extra=None):
    df = pd.DataFrame({"y": y})
    for l in lags:
        df[f"lag{l}"] = df["y"].shift(l)
    if extra is not None:
        for k, v in extra.items():
            df[k] = v
    df = df.dropna().reset_index(drop=True)
    return df["y"].values, df.drop(columns="y").values

def online_forecast(y, X, train=2000, refit=250, window=3000, seed=0):
    yhat = np.full(len(y), np.nan)
    model = None
    for t in range(train, len(y)):
        if model is None or (t - train) % refit == 0:
            lo = max(0, t - window)
            model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=seed)
            model.fit(X[lo:t], y[lo:t])
            # batch-predict the next `refit` points (model fixed in between)
            hi = min(len(y), t + refit)
            yhat[t:hi] = model.predict(X[t:hi])
    return y[train:], yhat[train:]

def elec():
    df = pd.read_csv(f"{D}/elec.csv")
    y = df["nswdemand"].values
    per = df["period"].values
    extra = {"sinp": np.sin(2*np.pi*per), "cosp": np.cos(2*np.pi*per),
             "price1": pd.Series(df["nswprice"].values).shift(1).values,
             "vic1": pd.Series(df["vicdemand"].values).shift(1).values,
             "tr1": pd.Series(df["transfer"].values).shift(1).values}
    return lagframe(y, [1, 2, 3, 48, 336], extra)

def power():
    df = pd.read_csv(f"{D}/household_power_consumption.txt", sep=";", na_values="?",
                     low_memory=False, usecols=["Date", "Time", "Global_active_power"])
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    s = df.set_index("dt")["Global_active_power"].resample("h").mean().interpolate()
    h = s.index.hour.values; dw = s.index.dayofweek.values
    extra = {"sinh": np.sin(2*np.pi*h/24), "cosh": np.cos(2*np.pi*h/24), "dow": dw}
    return lagframe(s.values, [1, 2, 3, 24, 168], extra)

def temp():
    df = pd.read_csv(f"{D}/daily-min-temperatures.csv")
    d = pd.to_datetime(df["Date"]).dt.dayofyear.values
    extra = {"sind": np.sin(2*np.pi*d/365.25), "cosd": np.cos(2*np.pi*d/365.25)}
    return lagframe(df["Temp"].values.astype(float), [1, 2, 3, 7], extra)

def machine():
    df = pd.read_csv(f"{D}/machine_temperature_system_failure.csv")
    return lagframe(df["value"].values, [1, 2, 3, 6, 12, 288])

if __name__ == "__main__":
    out = {}
    cfg = {"ELEC2": (elec, 2000, 250), "Power": (power, 2000, 250),
           "MelbTemp": (temp, 730, 90), "MachineTemp": (machine, 2000, 250)}
    for name, (fn, tr, rf) in cfg.items():
        y, X = fn()
        yy, yh = online_forecast(y, X, train=tr, refit=rf)
        out[name] = (yy, yh)
        print(name, len(y), len(yy), "MAE", np.mean(np.abs(yy - yh)))
    pickle.dump(out, open(f"{D}/streams.pkl", "wb"))
