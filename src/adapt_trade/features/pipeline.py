from river import stats
from river.utils import Rolling

class FeaturePipeline:
    def __init__(self):
        self.mean_ret = Rolling(stats.Mean(), window_size=10)
        self.var_ret = Rolling(stats.Var(), window_size=10)
        self.mean_vol = Rolling(stats.Mean(), window_size=20)
        #self.ewm_ret = stats.EWMean(fading_factor=0.6)
        self.ewm = stats.EWMean(fading_factor=0.6)
        self._prev_close = None
        self._n_bars = 0

        # RSI Trackers
        self.ewm_gain = stats.EWMean(fading_factor=0.4)
        self.ewm_loss = stats.EWMean(fading_factor=0.4)

        # MACD trackers
        self.ewm_fast = stats.EWMean(fading_factor=0.154)
        self.ewm_slow = stats.EWMean(fading_factor=0.074)

    def update(self, candle):
        close = candle['close']
        volume = candle['volume']
        opn = candle['open']

        if self._prev_close is None:
            ret = 0
        else:
            ret = (close / self._prev_close) - 1
        
        gain = max(ret,0)
        loss = max(-ret,0)
        
        self.mean_ret.update(ret)
        self.var_ret.update(ret)
        # self.ewm_ret.update(ret)
        self.mean_vol.update(volume)
        self.ewm.update(close)
        
        #RSI
        self.ewm_gain.update(gain)
        self.ewm_loss.update(loss)

        #MACD
        self.ewm_fast.update(close)
        self.ewm_slow.update(close)
        macd = self.ewm_fast.get() - self.ewm_slow.get()

        self._prev_close = close
        self._n_bars += 1

        if self._n_bars < 10:
            return None
        else:
            avg_gain = self.ewm_gain.get()
            avg_loss = self.ewm_loss.get()

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            features = {
                'ret_1': ret,
                'cum_ret_10': self.mean_ret.get(),
                'volatility_10': self.var_ret.get() ** 0.5,
                'bar_range': (candle['high'] - candle['low']) / candle['open'],
                'vol_ratio': volume / self.mean_vol.get(),
                'close_vs_ewm': close / self.ewm.get(),
                'bar_direction': 1 if close >= opn else 0,
                'rsi':rsi,
                'macd':macd
            }
            return features
    
    
