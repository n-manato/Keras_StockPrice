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
ticker = 'GS'
data = yf.download(ticker, start=start, end=end)

# カラムの名前を変更
data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# テクニカル指標の計算
data['SMA'] = talib.SMA(data['Close'], timeperiod=14)
data['EMA'] = talib.EMA(data['Close'], timeperiod=14)
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['MOM'] = talib.MOM(data['Close'], timeperiod=14)
macd, signal, _ = talib.MACD(
    data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = macd
data['ATR'] = talib.ATR(data['High'], data['Low'],
                        data['Close'], timeperiod=14)
dx = talib.DX(data['High'], data['Low'], data['Close'])
data['DX'] = dx
data.dropna(inplace=True)
print(data)

# グラフ表示
fig, axs = plt.subplots(4, 2, figsize=(12, 12))
fig.suptitle('テクニカル指標のグラフ', fontsize=16)

axs[0, 0].plot(data.index, data['Close'], label='Close', linewidth=0.7)
axs[0, 0].set_title('Close')
axs[0, 0].set_xlabel('日付')
axs[0, 0].set_ylabel('価格')
axs[0, 0].grid()

axs[0, 1].plot(data.index, data['SMA'], label='SMA',
               linewidth=0.7, color='orange')
axs[0, 1].set_title('SMA')
axs[0, 1].set_xlabel('日付')
axs[0, 1].set_ylabel('価格')
axs[0, 1].grid()

axs[1, 0].plot(data.index, data['EMA'], label='EMA',
               linewidth=0.7, color='green')
axs[1, 0].set_title('EMA')
axs[1, 0].set_xlabel('日付')
axs[1, 0].set_ylabel('価格')
axs[1, 0].grid()

axs[1, 1].plot(data.index, data['RSI'], label='RSI',
               linewidth=0.7, color='red')
axs[1, 1].set_title('RSI')
axs[1, 1].set_xlabel('日付')
axs[1, 1].set_ylabel('RSI')
axs[1, 1].grid()

axs[2, 0].plot(data.index, data['MOM'], label='MOM',
               linewidth=0.7, color='purple')
axs[2, 0].set_title('MOM')
axs[2, 0].set_xlabel('日付')
axs[2, 0].set_ylabel('MOM')
axs[2, 0].grid()

axs[2, 1].plot(data.index, data['MACD'], label='MACD',
               linewidth=0.7, color='brown')
axs[2, 1].set_title('MACD')
axs[2, 1].set_xlabel('日付')
axs[2, 1].set_ylabel('MACD')
axs[2, 1].grid()

axs[3, 0].plot(data.index, data['ATR'], label='ATR',
               linewidth=0.7, color='pink')
axs[3, 0].set_title('ATR')
axs[3, 0].set_xlabel('日付')
axs[3, 0].set_ylabel('ATR')
axs[3, 0].grid()

axs[3, 1].plot(data.index, data['DX'], label='DX', linewidth=0.7, color='gray')
axs[3, 1].set_title('DX')
axs[3, 1].set_xlabel('日付')
axs[3, 1].set_ylabel('DX')
axs[3, 1].grid()

plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close', linewidth=0.7, color='blue')
plt.plot(data.index, data['SMA'], label='SMA', linewidth=0.7, color='orange')
plt.plot(data.index, data['EMA'], label='EMA', linewidth=0.7, color='green')
plt.plot(data.index, data['RSI'], label='RSI', linewidth=0.7, color='red')
plt.plot(data.index, data['MOM'], label='MOM', linewidth=0.7, color='purple')
plt.plot(data.index, data['MACD'], label='MACD', linewidth=0.7, color='brown')
plt.plot(data.index, data['ATR'], label='ATR', linewidth=0.7, color='pink')
plt.plot(data.index, data['DX'], label='DX', linewidth=0.7, color='gray')

plt.title('終値とテクニカル指標の比較', fontsize=16)
plt.xlabel('日付')
plt.ylabel('価格 / 指標値')
plt.grid()
plt.legend()

plt.show()
