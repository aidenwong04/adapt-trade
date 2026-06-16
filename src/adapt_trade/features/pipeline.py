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

    def update(self, candle):
        close = candle['close']
        volume = candle['volume']
        opn = candle['open']

        if self._prev_close is None:
            ret = 0
        else:
            ret = (close / self._prev_close) - 1
        
        self.mean_ret.update(ret)
        self.var_ret.update(ret)
        self.ewm_ret.update(ret)
        self.mean_vol.update(volume)
        self.ewm.update(close)

        self._prev_close = close
        self._n_bars += 1

        if self._n_bars < 10:
            return None
        else:
            features = {
                'ret_1': ret,
                'cum_ret_10': self.mean_ret.get(),
                'volatility_10': self.var_ret.get() ** 0.5,
                'bar_range': (candle['high'] - candle['low']) / candle['open'],
                'vol_ratio': volume / self.mean_vol.get(),
                'close_vs_ewm': close / self.ewm.get(),
                'bar_direction': 1 if close >= opn else 0
            }
            return features
    
    
