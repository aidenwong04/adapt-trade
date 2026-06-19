import pandas as pd 
from binance import Client
import os

from pathlib import Path

client = Client("","")

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "2 years ago UTC")

df = pd.DataFrame(klines)

df.columns = ["date", "open", "high", "low", "close", "volume",
              "close_time", "quote_vol", "trades", "taker_buy_base",
              "taker_buy_quote", "ignore"]

df = df[["date", "open", "high", "low", "close", "volume"]]

df["date"] = pd.to_datetime(df["date"], unit="ms")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col]).astype(float)


DATA_PATH = Path(__file__).parent.parent.parent / "data"
df.to_csv(f'{DATA_PATH}/btcusdt_1h.csv', index=False)
print(f"Downloaded {len(df)} candles from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
