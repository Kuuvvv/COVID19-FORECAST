import glob
import logging
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

DATA_GLOB = r"C:\CovidProject\datalake\raw_data\**\*.parquet"
TOP_N = 10          # number of regions to forecast
HORIZON = 7         # days ahead
OUT_FILE = r"C:\CovidProject\forecasts.csv"
ACTUALS_FILE = r"C:\CovidProject\actuals.csv"

# 1. Load and de-duplicate
files = glob.glob(DATA_GLOB, recursive=True)
print(f"Loading {len(files)} parquet files...")
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.drop_duplicates(subset=['country', 'state', 'date'])

# 2. Aggregate provinces to country level (cumulative totals add up)
daily = df.groupby(['country', 'date'], as_index=False)['confirmed_cases'].sum()
daily = daily.sort_values(['country', 'date'])

# 3. Cumulative -> daily new cases
daily['new_cases'] = daily.groupby('country')['confirmed_cases'].diff().fillna(0)
daily['new_cases'] = daily['new_cases'].clip(lower=0)   # fix data corrections that go negative

# 4. Pick top regions by total cases
top = (daily.groupby('country')['confirmed_cases'].max()
       .nlargest(TOP_N).index.tolist())
print("Forecasting:", top)

# 5. Fit Prophet per region and forecast 7 days ahead
results = []
for region in top:
    d = daily[daily['country'] == region][['date', 'new_cases']]
    d = d.rename(columns={'date': 'ds', 'new_cases': 'y'})
    # smooth weekly reporting noise a little
    d['y'] = d['y'].rolling(7, min_periods=1).mean()

    m = Prophet(weekly_seasonality=True, yearly_seasonality=False,
                daily_seasonality=False, interval_width=0.9)
    m.fit(d)
    future = m.make_future_dataframe(periods=HORIZON)
    fc = m.predict(future).tail(HORIZON)

    out = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    out.columns = ['forecast_date', 'predicted_cases', 'lower_bound', 'upper_bound']
    out[['predicted_cases', 'lower_bound', 'upper_bound']] = \
        out[['predicted_cases', 'lower_bound', 'upper_bound']].clip(lower=0).round(0)
    out.insert(0, 'region', region)
    results.append(out)
    print(f"  done: {region}")

pd.concat(results).to_csv(OUT_FILE, index=False)
daily[daily['country'].isin(top)][['country', 'date', 'confirmed_cases', 'new_cases']] \
    .to_csv(ACTUALS_FILE, index=False)
print(f"Saved forecasts to {OUT_FILE} and actuals to {ACTUALS_FILE}")