import pandas as pd 
from binance import Client
import os

client = Client("","")

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "6 months ago UTC")

df = pd.DataFrame(klines)

df.columns = ["date", "open", "high", "low", "close", "volume",
              "close_time", "quote_vol", "trades", "taker_buy_base",
              "taker_buy_quote", "ignore"]

df = df[["date", "open", "high", "low", "close", "volume"]]

df["date"] = pd.to_datetime(df["date"], unit="ms")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col]).astype(float)

os.makedirs('data', exist_ok=True)
df.to_csv('data/btcusdt_15m.csv', index=False)
print(f"Downloaded {len(df)} candles from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
