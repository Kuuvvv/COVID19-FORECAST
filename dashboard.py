import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="COVID-19 Forecast", layout="wide")
st.title("COVID-19 Case Trend Forecasting")

@st.cache_data
def load():
    fc = pd.read_csv(r"C:\CovidProject\forecasts.csv", parse_dates=["forecast_date"])
    ac = pd.read_csv(r"C:\CovidProject\actuals.csv", parse_dates=["date"])
    return fc, ac

fc, ac = load()

# Sidebar controls
region = st.sidebar.selectbox("Region", sorted(fc["region"].unique()))
days = st.sidebar.slider("Days of history to show", 30, 365, 90)

a = ac[ac["country"] == region].sort_values("date").copy()
a["avg7"] = a["new_cases"].rolling(7, min_periods=1).mean()
f = fc[fc["region"] == region].sort_values("forecast_date")

last_week = a["new_cases"].tail(7).sum()
proj_week = f["predicted_cases"].sum()
delta = proj_week - last_week
pct = (delta / last_week * 100) if last_week else 0

capacity = st.sidebar.number_input(
    "Baseline weekly case capacity",
    min_value=0, value=int(last_week * 1.2), step=1000,
    key=f"cap_{region}",
)

# Chart: actuals vs forecast with confidence band
hist = a.tail(days)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["avg7"], name="Actual (7-day avg)",
                         line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(
    x=list(f["forecast_date"]) + list(f["forecast_date"][::-1]),
    y=list(f["upper_bound"]) + list(f["lower_bound"][::-1]),
    fill="toself", fillcolor="rgba(255,127,14,0.2)",
    line=dict(color="rgba(0,0,0,0)"), name="90% interval"))
fig.add_trace(go.Scatter(x=f["forecast_date"], y=f["predicted_cases"], name="Forecast",
                         line=dict(color="#ff7f0e", dash="dash")))
fig.update_layout(title=f"{region}: daily new cases, actual vs 7-day forecast",
                  xaxis_title="Date", yaxis_title="New cases per day",
                  hovermode="x unified", height=450)
st.plotly_chart(fig, use_container_width=True)

# Resource allocation panel
st.subheader("Resource allocation outlook")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last 7 days (actual)", f"{last_week:,.0f}")
c2.metric("Next 7 days (forecast)", f"{proj_week:,.0f}", f"{pct:+.1f}%")
c3.metric("Baseline capacity", f"{capacity:,.0f}")
gap = proj_week - capacity
c4.metric("Surge vs capacity", f"{gap:,.0f}", "OVER capacity" if gap > 0 else "within capacity",
          delta_color="inverse" if gap > 0 else "normal")

if gap > 0:
    st.warning(f"Projected cases exceed baseline capacity by {gap:,.0f} next week. "
               "Consider pre-positioning beds, PPE and staffing.")
else:
    st.success("Projected cases are within baseline capacity.")

st.subheader("Forecast table")
st.dataframe(f.assign(forecast_date=f["forecast_date"].dt.date), use_container_width=True)
st.caption(f"Data source: Johns Hopkins CSSE, streamed via Kafka. Latest data point: {a['date'].max().date()}.")