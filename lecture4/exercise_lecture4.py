import os
import glob
import natsort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import basic_functions as b_func
import copy

def get_min_sleep_hr(sleep_time, df_hr, df_analysis_data):

    # 取得されたデータの睡眠時心拍の抽出
    tmp_df_sleep_hr = df_hr[df_hr['sleep_flg']==1]# df_hrを用いて睡眠中の心拍数を抽出
    tmp_df_sleep_hr.drop(index=tmp_df_sleep_hr[tmp_df_sleep_hr['hr_value']==0].index, inplace=True)

    # 1日ごとの睡眠時心拍の抽出し、1日ごｔの睡眠時心拍の最小値を導出
    tmp_dict = dict()
    for date in sleep_time:
        start = sleep_time[date]['start']
        end = sleep_time[date]['end']
        sleep_hr = tmp_df_sleep_hr.loc[start:end, 'hr_value'].values
        min_sleep_hr = np.min(sleep_hr)
        tmp_dict[date] = min_sleep_hr

    df_analysis_data['min_sleep_hr'] = [0] * len(df_analysis_data.index.tolist())
    for date in tmp_dict:
        df_analysis_data.loc[date, 'min_sleep_hr'] = tmp_dict[date]

    return df_analysis_data

def make_scatter_plot(df_analysis_data, save_path):

    #=======演習1=======
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    # ここで作図を実装
    ax.set_xlabel('Minimum Value of Sleep HR')
    ax.set_ylabel('RHR')
    plt.tight_layout()
    plt.savefig(save_path)
    #===ここまでが演習1===

def linear_reg(df_analysis_data, save_path):

    #=======演習2=======
    # 学習用データ、テストデータに分割
    X, y = df_analysis_data[['min_sleep_hr']].values, df_analysis_data[['RHR']].values
    X_train, X_test, y_train, y_test = 1# 8:2にデータを分割

    # 回帰直線の導出

    #===ここまでが演習2===

    #=======演習3=======
    # 回帰直線を可視化
    # ここで擬似的な入力による回帰直線を導出
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    # ここに作図の実装
    ax.set_xlabel('Minimum Value of Sleep HR')
    ax.set_ylabel('RHR')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ex_linear_reg_minSleepHR_RHR.png'))
    #===ここまでが演習3===

    #=======演習4=======
    # 推定値と実際の値の比較
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    # ここに実装
    ax.set_xlabel('True RHR', fontsize=20)
    ax.set_ylabel('Predicted RHR', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ex_compare_true_pred_RHR.png'))
    #===ここまでが演習4===

    #=======演習5=======
    threshold = np.median(df_analysis_data['RHR'])
    true_condition = copy.deepcopy(y_test)
    # ここで正解の体調決定

    pred_condition = copy.deepcopy(pred_RHR)
    # ここで推定の体調決定

    acc = accuracy_score(y_true=true_condition, y_pred=pred_condition)
    print('Accuracy: {}'.format(acc))
    #===ここまでが演習5===

def main():

    #==============================
    # データセット構築・解析のための準備
    #==============================
    # 使用するデータのパスなどの設定
    cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    file_path = os.path.join(cwd, 'analysis_data')
    fitbit_path = glob.glob(os.path.join(file_path, 'activities', '*'))
    hr_path = os.path.join(fitbit_path[0], 'hr')
    sleep_path = os.path.join(fitbit_path[0], 'sleep')

    # データファイルのパスを取得
    hr_data_paths = natsort.natsorted(glob.glob(os.path.join(hr_path, '*.txt')))
    sleep_data_paths = natsort.natsorted(glob.glob(os.path.join(sleep_path, '*.txt')))

    # 生データの読み込み
    df_analysis_data = b_func.get_RHR(hr_data_paths=hr_data_paths)
    df_hr, sleep_time = b_func.get_sleep_hr_data(hr_data_paths, sleep_data_paths)
    df_hr['time'] = pd.to_datetime(df_hr['time'])
    df_hr = df_hr.reset_index(drop=True)
    df_hr = df_hr.set_index('time')
    df_analysis_data = df_analysis_data.set_index('Date')

    #==============================
    # データセット構築
    #==============================
    df_analysis_data = get_min_sleep_hr(sleep_time, df_hr, df_analysis_data)

    #==============================
    # 以降、データ解析
    #==============================
    # 演習1は、以下のmake_scatter_plot関数を改変する
    save_path = os.path.join(cwd, 'analysis_results', 'lecture4')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    make_scatter_plot(df_analysis_data, os.path.join(save_path, 'min_sleepHR_RHR.png'))

    # 演習2-5は、以下のlinear_reg関数を改変する
    linear_reg(df_analysis_data, save_path)

if __name__ == '__main__':
    main()