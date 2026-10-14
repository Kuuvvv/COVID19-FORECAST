import pandas as pd
import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC_NAME = 'covid-case-updates'

print("Loading dataset...")
df = pd.read_csv('covid_data.csv')
date_columns = df.columns[4:]
df_melted = df.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
    value_vars=date_columns,
    var_name='Date',
    value_name='ConfirmedCases'
)

df_melted['Date'] = pd.to_datetime(df_melted['Date'])
df_melted = df_melted.sort_values(by='Date')  # remove .head(1000) for the full run

print("Streaming to Kafka...")
for index, row in df_melted.iterrows():
    payload = {
        "country": row['Country/Region'],
        "state": row['Province/State'] if pd.notna(row['Province/State']) else "N/A",
        "date": row['Date'].strftime('%Y-%m-%d'),
        "confirmed_cases": int(row['ConfirmedCases'])
    }
    producer.send(TOPIC_NAME, value=payload)
    print(f"Sent: {payload['country']} ({payload['date']}) - Cases: {payload['confirmed_cases']}")
    time.sleep(0.001)

producer.flush()
print("Finished streaming.")