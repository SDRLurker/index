# -*- coding: utf-8 -*-
import numpy as np
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc

def make_tick_input(symbol, start, end, tick_unit = 0.01):
    hist = web.DataReader(symbol,'yahoo',start,end)
    ticks = 1 / tick_unit
    s_open = np.round(hist.ix[0]['Open'] * ticks) 
    hist['open_tick'] = np.round(hist['Open'] * ticks) - s_open
    hist['body'] = np.round((hist['Close']-hist['Open']) * ticks)
    hist['upper'] = np.round((hist['High']-np.maximum(hist['Open'], hist['Close'])) * ticks)
    hist['lower'] = np.round((np.minimum(hist['Open'], hist['Close'])-hist['Low']) * ticks)
    return hist

start = datetime.datetime(2016,9,1)
end = datetime.datetime(2016,9,30)
hist = make_tick_input('^DJI', start, end)
print(hist)

fid, ax = plt.subplots(figsize=(16,6))
ax.set_xticks(range(0, len(hist)))
ax.set_xticklabels(hist.index.strftime('%Y-%m-%d'), rotation=70)
candlestick2_ohlc(ax, hist['Open'], hist['High'], hist['Low'], hist['Close'],
	width=1, colorup='r', colordown='b')
plt.show()

