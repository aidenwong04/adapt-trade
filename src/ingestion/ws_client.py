import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

import queue

async def stream_ohlcv(symbols, callback):
    '''
    '''
    streams = [s.lower() + '@kline_15m' for s in symbols]
    
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    
    # multiplex_socket takes a list of stream names and builds the URL automatically
    multi_socket = bm.multiplex_socket(streams)
    while True:
        try: 
            async with multi_socket as stream:
                print("Connected to streams. Listening for data...")
                while True:
                    res = await stream.recv()
                    # Binance combined streams wrap data in an outer dict: {'stream': '...', 'data': {...}}
                    if res and 'data' in res:
                        k = res["data"]["k"]
                        if not k['x']: # candle still open
                            continue
                        # candle closed. create the ohlcv data
                        candle = {
                            'symbol': k['s'],
                            'ts': k['T'],   # close timestamp in milliseconds
                            'o': float(k['o']),
                            'h': float(k['h']),
                            'l': float(k['l']),
                            'c': float(k['c']),
                            'v': float(k['v']),
                        }

                        callback(candle) # pass the result to the callback function (to update the model)
        except Exception as e:
            print(f"Connection lost due to error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

        finally:
            # Clean up the previous client connection state before restarting
            await client.close_connection()
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            multi_socket = bm.multiplex_socket(streams)

def run_stream(symbols, queue):
    callback = queue.put
    asyncio.run(stream_ohlcv(symbols, callback))
