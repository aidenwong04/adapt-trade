import queue, threading
from adapt_trade.ingestion.ws_client import run_stream

q = queue.Queue()
t = threading.Thread(target=run_stream, args=[['BTCUSDT'], q], daemon=True)
t.start()
print("Waiting for a closed candle... (up to 15 min)")
print(q.get())
