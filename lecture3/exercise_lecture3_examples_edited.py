import sys
print(sys.path)
import os
import glob
import pandas as pd
import numpy as np
import natsort
import datetime
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
import load_data
from collections import Counter

def Dimension_Reduction_daily_data(daily_data, save_path):

    df_average = {
                'date':[], 'whole average':[], 'wake min':[], 'wake max':[], 'sleep min':[], 'sleep max':[],
                'sleep temp min': [], 'sleep temp var': []
                }
    df = pd.DataFrame()
    for tmp in daily_data:
        df[tmp] = daily_data[tmp]
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    time = daily_data['time']
    datetime_start = datetime.datetime.strptime(time[0], '%Y-%m-%d %H:%M:%S')
    datetime_end = datetime.datetime.strptime(time[-1], '%Y-%m-%d %H:%M:%S')
    f_date = datetime_start
    for _ in range((datetime_end - datetime_start).days):

        date = datetime.datetime(f_date.year, f_date.month, f_date.day)
        df_average['date'].append(date)

        tmp_df = df[f_date:f_date + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)]
        tmp_df = tmp_df[tmp_df['value']!=0]

        #=======演習1=======
        average_whole_hr = np.average(tmp_df['value'].values)
        df_average['whole average'].append(average_whole_hr)

        min_wake_hr = np.min(tmp_df['value'][tmp_df['sleep']==0].values)
        df_average['wake min'].append(min_wake_hr)

        max_wake_hr = np.max(tmp_df['value'][tmp_df['sleep']==0].values)
        df_average['wake max'].append(max_wake_hr)

        min_sleep_hr = np.min(tmp_df['value'][tmp_df['sleep']==1].values)
        df_average['sleep min'].append(min_sleep_hr)

        max_sleep_hr = np.max(tmp_df['value'][tmp_df['sleep']==1].values)
        df_average['sleep max'].append(max_sleep_hr)

        min_sleep_temp = np.min(tmp_df['temperature'][tmp_df['sleep']==1].values)
        df_average['sleep temp min'].append(min_sleep_temp)

        var_sleep_temp = np.var(tmp_df['temperature'][tmp_df['sleep']==1].values)
        df_average['sleep temp var'].append(var_sleep_temp)
        #===ここまでが演習1===

        f_date += datetime.timedelta(days=1)
    
    new_df = pd.DataFrame()
    for tmp in df_average:
        if tmp != 'date':
            new_df[tmp] = df_average[tmp]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    data_matrix = new_df.values

    #=======演習2=======
    X_decomp = umap.UMAP(n_components=2).fit_transform(data_matrix)
    np.save(os.path.join(save_path, 'fitbitX'), X_decomp[:,0])
    np.save(os.path.join(save_path, 'fitbitY'), X_decomp[:,1])

    for ii, tmp_date in enumerate(df_average['date']):
        ax.scatter(X_decomp[ii, 0], X_decomp[ii, 1], s=20, label=tmp_date)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dimension_reduction_fig.png'))
    #===ここまでが演習2===

    return df_average

def kmeans_daily_data(df_average, save_path):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    tmp_X1 = np.load(os.path.join(save_path, 'fitbitX.npy'))
    tmp_X2 = np.load(os.path.join(save_path, 'fitbitY.npy'))

    X_decomp = np.zeros((tmp_X1.shape[0], 2))
    X_decomp[:, 0] = tmp_X1
    X_decomp[:, 1] = tmp_X2

    #=======演習3=======
    # 次元削減さらえたデータX_decompを用いて、クラスタリングせよ。
    # なお、クラスタリングにはscikit-learnのK-meansを用いること。
    # また、辞書型の変数label_values_indを以下のように定義せよ。
    # label_values_ind = {0:ラベル0のX_decompの値, 1:ラベル1のX_decompの値}
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit_predict(X = X_decomp)
    label_values_ind = {0: [], 1: []}
    for i in range(X_decomp.shape[0]):
        if kmeans.labels_[i] == 0:
            label_values_ind[0].append(X_decomp[i, :])
        else:
            label_values_ind[1].append(X_decomp[i, :])
    #===ここまでが演習3===

    #=======演習4=======
    # ここで、matplotlibで散布図を作成し、クラスタごとに点の色を変更せよ。
    gr0 = np.array(label_values_ind[0])
    gr1 = np.array(label_values_ind[1])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(gr0[:, 0], gr0[:, 1], color = 'green')
    ax.scatter(gr1[:, 0], gr1[:, 1], color = 'blue')

    # クラスタの重心（クラスタ内の座標値の平均）を計算し、クラスタの重心を散布図上に示せ。
    # また、クラスタ内のデータの平均を求め、図内で可視化せよ。
    for i in label_values_ind:
        x_ind = kmeans.cluster_centers_[i][0]# x座標の平均
        y_ind = kmeans.cluster_centers_[i][1]# y座標の平均
        ax.scatter(x_ind, y_ind, s=50, label='Label-{}'.format(i), color='black')
        avg_list = [] # このリストにデータの平均値を格納する。
        for tmp in df_average:
            a = 1
            # このfor文内でデータの平均値を導出
        ax.text(x_ind, y_ind, avg_list, size=15,color='#555555') # クラスタの重心の点の近くにテキストを表示
    #===ここまでが演習4===

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'kmeans_fig.png'))

    df_average['labels'] = kmeans.labels_.tolist()

    return df_average

def main():
    cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    file_path = os.path.join(cwd, 'analysis_data')
    fitbit_path = glob.glob(os.path.join(file_path, 'activities', '*'))
    analysis_res = os.path.join(cwd, 'analysis_results')
    if not os.path.exists(analysis_res):
        os.mkdir(analysis_res)
    fig_res = os.path.join(analysis_res, 'figures')
    if not os.path.exists(fig_res):
        os.mkdir(fig_res)
    hr_path = os.path.join(fitbit_path[0], 'hr')
    sleep_path = os.path.join(fitbit_path[0], 'sleep')

    json_path = os.path.join(file_path, 'data_collection_hr_sleep.json')
    if not os.path.exists(json_path):
        hr_data_paths = natsort.natsorted(glob.glob(os.path.join(hr_path, '*.txt')))
        sleep_data_paths = natsort.natsorted(glob.glob(os.path.join(sleep_path, '*.txt')))

        data_collection = {}
        for hr_data_path in hr_data_paths:
            date = os.path.basename(hr_data_path).split('.txt')[0]
            data_collection = load_data.get_Data_activity(hr_data_path, date, data_collection)

        value_list = [0] * data_collection['value'].shape[0]
        if 'sleep' not in data_collection:
            data_collection['sleep'] = np.array([])
        data_collection['sleep'] = np.append(data_collection['sleep'], np.array(value_list))

        for sleep_data_path in sleep_data_paths:
            data_collection = load_data.get_Data_sleep(sleep_data_path, data_collection)
        
        tmp_df = pd.DataFrame()
        tmp_df['time'] = data_collection['time'].tolist()
        tmp_df['time'] = tmp_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        data_collection['value'] = data_collection['value'].tolist()
        data_collection['time'] = tmp_df['time'].tolist()
        data_collection['sleep'] = data_collection['sleep'].tolist()

        with open(json_path, 'w') as f:
            json.dump(data_collection, f, indent=4)
    else:
        json_file = os.path.join(json_path)
        with open(json_file) as f:
            s = f.read()
        data_collection = json.loads(s)
    
    halshare = pd.read_csv(os.path.join(file_path, '20220617_lvnsu003.csv'))
    tmp_datetime = [tmp.split('+09')[0] for tmp in halshare['date_measured'].tolist()]
    halshare['date_measured'] = tmp_datetime
    halshare['date_measured'] = pd.to_datetime(halshare['date_measured'])
    halshare = halshare.sort_values(by=['date_measured']) # ハルシェのデータは、時間でソートされていないのでソートを実行する
    halshare_list = [0] * len(data_collection['time'])
    for i, tmp_datetime in enumerate(halshare['date_measured'].tolist()):
        str_datetime = datetime.datetime.strftime(tmp_datetime, '%Y-%m-%d %H:%M:%S')[:-2] + '00'
        if str_datetime in data_collection['time']:
            ind = data_collection['time'].index(str_datetime)
            halshare_list[ind] = halshare['temperature'][i]
    data_collection['temperature'] = halshare_list

    df_average = Dimension_Reduction_daily_data(data_collection, fig_res)
    df_average = kmeans_daily_data(df_average, fig_res)

    #=======演習5=======
    # ここで、df_averageの結果をcsvファイルに格納
    pd.DataFrame(df_average).to_csv(os.path.join(analysis_res, 'df_average.csv'))
    #===ここまでが演習5===

if __name__ == '__main__':
    main()
