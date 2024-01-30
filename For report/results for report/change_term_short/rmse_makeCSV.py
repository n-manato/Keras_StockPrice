import pandas as pd
import os

def process_csv(folder_path):
    rmses = {}

    # フォルダ内のすべてのファイルを処理
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 拡張子を取り除く
            file_name_without_extension = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            # CSVファイルを読み込む
            df = pd.read_csv(file_path)

            # 'rmse'の列が存在するか確認
            if 'rmse' in df.columns:
                # 'rmse'列の値を取得
                r2_values = df['rmse'].tolist()
                # キーをヘッダーにして辞書に追加
                rmses[file_name_without_extension] = r2_values

    return rmses

def save_to_csv(rmses, output_file):
    # DataFrameに変換してCSVファイルに書き込み
    df = pd.DataFrame(rmses)
    df.to_csv(output_file, index=False)

folder_path = "./"
output_file = "rmses.csv"

# フォルダ内のCSVファイルからrmseを取得
rmses = process_csv(folder_path)

# rmsesをCSVファイルに保存
save_to_csv(rmses, output_file)
