import os

# フォルダのパス
folder_path = "./"

# フォルダ内のファイル名を取得
file_names = os.listdir(folder_path)

# ファイル名を出力
for file_name in file_names:
    if file_name != "name.py":
        print("figure_for_report/Best_of_short_windowsize/{}".format(file_name))
