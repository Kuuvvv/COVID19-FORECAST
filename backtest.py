"""
Backtest: hold out the last 7 days of each region, fit Prophet on the rest
(using the SAME pipeline as forecast.py), forecast the held-out week, and
compare against what actually happened.

Usage (Command Prompt):
    python backtest.py

Outputs (in C:\\CovidProject):
    backtest.csv          region, date, actual, actual_raw, predicted, lower_bound, upper_bound
    backtest_metrics.csv  region, mae, wmape, wmape_raw, mape, coverage_90

'actual' is the 7-day smoothed series, which is what the model is trained to
predict (and what the dashboard plots). 'actual_raw' is the unsmoothed daily
count, kept so you can report error against raw numbers too.
"""
import logging
import os

import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

BASE = r"C:\CovidProject"
HOLDOUT = 7  # days held out, same as forecast horizon

actuals = pd.read_csv(os.path.join(BASE, "actuals.csv"), parse_dates=["date"])
regions = sorted(pd.read_csv(os.path.join(BASE, "forecasts.csv"))["region"].unique())

rows, metrics = [], []

for i, region in enumerate(regions, 1):
    d = (
        actuals[actuals["country"] == region]
        .sort_values("date")[["date", "new_cases"]]
        .rename(columns={"date": "ds", "new_cases": "y_raw"})
        .reset_index(drop=True)
    )
    # Same smoothing as forecast.py (trailing window, so no look-ahead leakage)
    d["y"] = d["y_raw"].rolling(7, min_periods=1).mean()

    train = d.iloc[:-HOLDOUT][["ds", "y"]]
    test = d.iloc[-HOLDOUT:]

    # Same settings as forecast.py
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                daily_seasonality=False, interval_width=0.9)
    m.fit(train)

    fc = m.predict(test[["ds"]])
    pred = fc["yhat"].clip(lower=0).round(0).to_numpy()
    lo = fc["yhat_lower"].clip(lower=0).round(0).to_numpy()
    hi = fc["yhat_upper"].clip(lower=0).round(0).to_numpy()
    act = test["y"].to_numpy()
    act_raw = test["y_raw"].to_numpy()

    err = np.abs(pred - act)
    err_raw = np.abs(pred - act_raw)
    nonzero = act > 0
    metrics.append({
        "region": region,
        "mae": err.mean(),
        # WMAPE: total absolute error / total actual cases (stable near zero)
        "wmape": err.sum() / act.sum() * 100 if act.sum() > 0 else np.nan,
        "wmape_raw": err_raw.sum() / act_raw.sum() * 100 if act_raw.sum() > 0 else np.nan,
        "mape": (err[nonzero] / act[nonzero]).mean() * 100 if nonzero.any() else np.nan,
        "coverage_90": ((act >= lo) & (act <= hi)).mean() * 100,
    })
    rows.append(pd.DataFrame({
        "region": region,
        "date": test["ds"].to_numpy(),
        "actual": act,
        "actual_raw": act_raw,
        "predicted": pred,
        "lower_bound": lo,
        "upper_bound": hi,
    }))
    print(f"[{i}/{len(regions)}] {region}: WMAPE {metrics[-1]['wmape']:.1f}%")

bt = pd.concat(rows, ignore_index=True)
mt = pd.DataFrame(metrics).round(2)
bt.to_csv(os.path.join(BASE, "backtest.csv"), index=False)
mt.to_csv(os.path.join(BASE, "backtest_metrics.csv"), index=False)

print("\n=== Backtest summary (last 7 known days held out) ===")
print(mt.sort_values("wmape").to_string(index=False))
print(f"\nMedian WMAPE: {mt['wmape'].median():.1f}%   Mean WMAPE: {mt['wmape'].mean():.1f}%")
print(f"Avg 90% interval coverage: {mt['coverage_90'].mean():.1f}%")
