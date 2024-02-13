import yfinance as yf
import talib
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import japanize_matplotlib

# データの範囲
start = dt.date(2020, 1, 1)
end = dt.date(2023, 1, 1)

# データ取得
ticker = 'NVDA'
data = yf.download(ticker, start=start, end=end)

# カラムの名前を変更
data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

data['EMA'] = talib.EMA(data['Close'], timeperiod=75)
macd, signal, _ = talib.MACD(
    data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close', linewidth=0.7, color='blue')
plt.plot(data.index, data['EMA'], label='EMA', linewidth=0.7, color='red')


plt.xlabel('日付')
plt.ylabel('価格 / 指標値')
plt.grid()
plt.legend()

plt.show()
