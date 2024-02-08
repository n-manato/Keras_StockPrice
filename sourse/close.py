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

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close', linewidth=0.7, color='red')

plt.title('終値とテクニカル指標の比較', fontsize=16)
plt.xlabel('日付')
plt.ylabel('価格 / 指標値')
plt.grid()
plt.legend()

plt.show()