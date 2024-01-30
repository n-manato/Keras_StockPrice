import pandas as pd
import os

def process_csv(folder_path):
    r2_scores = {}

    # フォルダ内のすべてのファイルを処理
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 拡張子を取り除く
            file_name_without_extension = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            # CSVファイルを読み込む
            df = pd.read_csv(file_path)

            # 'r2_score'の列が存在するか確認
            if 'r2_score' in df.columns:
                # 'r2_score'列の値を取得
                r2_values = df['r2_score'].tolist()
                # キーをヘッダーにして辞書に追加
                r2_scores[file_name_without_extension] = r2_values

    return r2_scores

def save_to_csv(r2_scores, output_file):
    # DataFrameに変換してCSVファイルに書き込み
    df = pd.DataFrame(r2_scores)
    df.to_csv(output_file, index=False)

folder_path = "./"
output_file = "r2_scores.csv"

# フォルダ内のCSVファイルからr2_scoreを取得
r2_scores = process_csv(folder_path)

# r2_scoresをCSVファイルに保存
save_to_csv(r2_scores, output_file)
