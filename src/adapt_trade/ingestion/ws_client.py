import asyncio
import pandas as pd
from binance import AsyncClient, BinanceSocketManager

async def stream_ohlcv(symbols, q: asyncio.Queue):
    streams = [s.lower() + '@kline_15m' for s in symbols]

    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    multi_socket = bm.multiplex_socket(streams)

    while True:
        try:
            async with multi_socket as stream:
                print("Connected to streams. Listening for data...")
                while True:
                    res = await stream.recv()
                    if res and 'data' in res:
                        k = res['data']['k']
                        if not k['x']:
                            continue
                        candle = {
                            'symbol': k['s'],
                            'ts':     pd.to_datetime(k['T'], unit="ms"),
                            'open':   float(k['o']),
                            'high':   float(k['h']),
                            'low':    float(k['l']),
                            'close':  float(k['c']),
                            'volume': float(k['v']),
                        }
                        await q.put(candle)  # ← push to queue instead of callback

        except Exception as e:
            print(f"Connection lost: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

        finally:
            await client.close_connection()
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            multi_socket = bm.multiplex_socket(streams)