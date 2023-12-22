# 必要なライブラリをインポート
import japanize_matplotlib
import datetime as dt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential, Model
import pandas as pd
import numpy as np
import pandas_datareader as web
import math
import warnings
import yfinance as yf
import talib
import csv
import os

warnings.filterwarnings('ignore')

# データの範囲
start = dt.date(2021, 1, 1)
end = dt.date(2022, 1, 1)

brand_list = ['NVDA', 'PFE', 'XOM', 'GS',
              'GM', 'NOC', 'CAT', 'WMT', 'MMM', 'BA']


for brand in brand_list:
    # フォルダのパス
    folder_path = "change_term_short"
    csv_file = os.path.join(folder_path, f'{brand}.csv')
    # ヘッダーを書き込む
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['brand', 'window_size', 'n_hidden',
                        'units1', 'term', 'r2_score', 'rmse'])
    for num in range(10, 130, 5):
        print(brand)
        # データ取得
        ticker = brand
        data = yf.download(ticker, start=start, end=end)

        # カラムの名前を変更
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # テクニカル指標の計算
        term = num
        data['SMA'] = talib.SMA(data['Close'], timeperiod=term)
        data['EMA'] = talib.EMA(data['Close'], timeperiod=term)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=term)
        data['MOM'] = talib.MOM(data['Close'], timeperiod=term)
        macd, signal, _ = talib.MACD(
            data['Close'], fastperiod=12, slowperiod=12+term, signalperiod=9)
        data['MACD'] = macd
        data['ATR'] = talib.ATR(data['High'], data['Low'],
                                data['Close'], timeperiod=term)
        dx = talib.DX(data['High'], data['Low'], data['Close'])
        data['DX'] = dx
        data.dropna(inplace=True)

        # 終値データとMAデータを結合
        combined_data = data[['Close', 'SMA', 'EMA',
                              'RSI', 'MOM', 'MACD', 'ATR', 'DX']]
        # print(combined_data)
        # 結合後のデータを正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(combined_data)
        # print(scaler)
        # print(scaled_data.shape)

        # 訓練データと正解データの作成
        training_data_len = math.ceil(len(data) * .7)
        window_size = 35

        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []

        for i in range(window_size, len(train_data)):
            # ウィンドウサイズ分のデータを含める
            x_train.append(train_data[i - window_size:i, :])
            y_train.append(train_data[i, 0])  # 正解データは終値のみ使用

        x_train, y_train = np.array(x_train), np.array(y_train)
        # print(x_train.shape[0], x_train.shape[1], x_train.shape[2])

        # 3次元に変換
        x_train_3D = np.reshape(
            x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        # LSTMモデルを定義
        n_hidden = 200  # 大きくすればするほど複雑なパターンを習得できる。重くなる。
        units1 = 150
        units2 = 1

        input_layer = Input(shape=(x_train_3D.shape[1], x_train_3D.shape[2]))
        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        lstm2 = LSTM(n_hidden, return_sequences=False)(lstm1)
        dense1 = Dense(units1)(lstm2)
        output_layer = Dense(units2)(dense1)

        model = Model(inputs=input_layer, outputs=output_layer)

        # モデルをコンパイル
        model.compile(optimizer='adam', loss='mean_squared_error')

        # モデルの学習
        batch_size = 1
        epochs = 50
        model.fit(x_train_3D, y_train, batch_size=batch_size, epochs=epochs)

        # 検証データの準備
        test_data = scaled_data[training_data_len - window_size:, :]
        x_test = []
        y_test = scaled_data[training_data_len:, :]

        for i in range(window_size, len(test_data)):
            x_test.append(test_data[i - window_size:i, :])

        x_test = np.array(x_test)
        x_test = np.reshape(
            x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # モデルに検証データを代入して予測
        predictions = model.predict(x_test)
        # print(predictions)
        predictions = predictions[:, 0]
        # print(predictions)
        # print(y_test)

        # モデルの精度を評価
        y_test = y_test[:, 0]  # 正しい形状に変更
        r2_score_value = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f'r2_score: {r2_score_value:.4f}')
        print(f'rmse: {rmse:.4f}')

        # グラフ表示
        train = data[:training_data_len]
        valid = data[training_data_len:]

        # 正規化された値を元のスケールに戻す（逆正規化）
        predictions = predictions.reshape(-1, 1)
        # print('predictons_last = ', predictions.shape)
        y_test = y_test.reshape(-1, 1)
        # print('ytest=', y_test)
        # 予測値の逆正規化
        # print(scaler.scale_[3])
        # print(scaler.min_[3])
        predicted_prices = (
            predictions * (scaler.data_max_[0] - scaler.data_min_[0])) + scaler.data_min_[0]
        # print('predicted_prices', predicted_prices)
        # 正解値の逆正規化
        actual_prices = data.iloc[training_data_len:,
                                  data.columns.get_loc('Close')].values
        # valid DataFrameに予測結果を追加
        valid['Predictions'] = predicted_prices[:, 0]

        # print(predicted_prices.shape)
        # print(actual_prices.shape)

        # グラフを表示する領域をfigとする
        fig = plt.figure(figsize=(12, 6))

        # グラフ間の余白を設定する
        fig.subplots_adjust(wspace=0.6, hspace=0.2)

        # GridSpecでfigを縦10、横15に分割する
        gs = gridspec.GridSpec(9, 14)

        # 分割した領域のどこを使用するかを設定する
        # gs[a1:a2, b1:b2]は、縦の開始位置(a1)から終了位置(a2)、横の開始位置(b1)から終了位置(b2)
        ax1 = plt.subplot(gs[0:8, 0:8])
        ax2 = plt.subplot(gs[0:5, 9:14])

        # 1番目のグラフを設定する
        ax1.set_title('9カラムの履歴と予測結果', fontsize=16)
        ax1.set_xlabel('日付', fontsize=12)
        ax1.set_ylabel('終値 ＄', fontsize=12)
        ax1.plot(data['Close'], label='実際の価格')
        ax1.plot(valid.index, valid['Predictions'], label='予測の価格')
        ax1.legend(loc='lower right')
        ax1.grid()

        # 2番目のグラフを設定する
        ax2.set_title('予測の価格と実際の価格の散布図表示', fontsize=16)
        ax2.set_xlabel('予測の価格', fontsize=12)
        ax2.set_ylabel('実際の価格', fontsize=12)
        ax2.scatter(actual_prices, predicted_prices,
                    label=f'r2_score: {r2_score_value:.4f} \n rmse: {rmse:.4f}')
        ax2.plot(actual_prices, actual_prices, 'k-')
        ax2.legend()
        ax2.grid()

        # plt.show()  # グラフを表示
        # フォルダが存在しない場合は作成する
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存するファイル名
        save_file_path = os.path.join(
            folder_path, f"{brand}_{window_size}_{n_hidden}_{units1}_{term}.png")
        plt.savefig(save_file_path)  # グラフを保存
        # plt.show()  # グラフを表示
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([brand, window_size, n_hidden, units1, term,
                            f'{r2_score_value:.4f}', f'{rmse:.4f}'])
