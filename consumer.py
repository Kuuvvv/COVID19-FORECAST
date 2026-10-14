import json
import os
import pandas as pd
from kafka import KafkaConsumer

BASE_DIR = r"C:\CovidProject\datalake\raw_data"
BATCH_SIZE = 500

consumer = KafkaConsumer(
    'covid-case-updates',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    group_id='hdfs-writer-full',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    consumer_timeout_ms=10000   # stop after 10s with no new messages
)

def flush(batch, part):
    df = pd.DataFrame(batch)
    df['date'] = pd.to_datetime(df['date'])
    for date, group in df.groupby('date'):
        folder = os.path.join(
            BASE_DIR,
            f"year={date.year}",
            f"month={date.month:02d}",
            f"day={date.day:02d}"
        )
        os.makedirs(folder, exist_ok=True)
        group.to_parquet(os.path.join(folder, f"part-{part}.parquet"), index=False)

print("Listening for messages...")
batch, part, total = [], 0, 0
for msg in consumer:
    batch.append(msg.value)
    total += 1
    if len(batch) >= BATCH_SIZE:
        flush(batch, part)
        print(f"Wrote batch {part} ({total} messages so far)")
        batch, part = [], part + 1

if batch:
    flush(batch, part)
print(f"Done. Total messages stored: {total}")