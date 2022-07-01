import os
import fitbit
from ast import literal_eval
import json
import time
from datetime import datetime, timedelta

TOKEN = {
    'USER_ID': 'B2CPC9',
    'CLIENT_ID': '238KZ4',
    'CLIENT_SECRET': '48b4986bc23652a13dcf14662e06226e177eed14'
    }

cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
TOKEN_FILE = os.path.join(cwd, 'analysis_data', TOKEN['USER_ID'] + '.txt')

def read_token(USER_ID, fpath):
    print('Read Current Token for {}'.format(USER_ID))

    file_name = os.path.join(fpath, '{}.txt'.format(USER_ID))
    with open(file_name) as f:
        tokens = f.read()

    token_dict = literal_eval(tokens)
    access_token = token_dict['access_token']
    refresh_token = token_dict['refresh_token']
    return access_token, refresh_token

def update_token(token):
    print('Token Updating '),
    f = open(TOKEN_FILE, 'w')
    f.write(str(token))
    f.close()
    return

def main():
    USER_ID = TOKEN['USER_ID']
    CLIENT_ID = TOKEN['CLIENT_ID']
    CLIENT_SECRET = TOKEN['CLIENT_SECRET']

    cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    analysis_path = os.path.join(cwd, 'analysis_data')

    start_date = datetime(2022, 6, 3) # データを取得開始したい日の日付を入力
    today = datetime.now()
    tmp = today - start_date

    for i in range(tmp.days + 1):

        ACCESS_TOKEN, REFRESH_TOKEN = read_token(USER_ID, analysis_path)

        # ID等の設定
        print('Start connection to Fitbit')
        authd_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET
                                ,access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN
                                ,refresh_cb=update_token)
        DATE = start_date + timedelta(days=i)
        DATE = DATE.date()
        print(DATE, USER_ID)

        # 1時間あたり150回程度しか、APIにアクセス出来ないので、ダウンロードするデータの種類は絞る
        for kind_of_data in ['hr', 'steps']: #, 'calories', 'minutesFairlyActive', 'minutesLightlyActive', 'minutesSedentary', 'minutesVeryActive']:

            if kind_of_data == 'hr':
                tmp_name = 'heart'
            else:
                tmp_name = kind_of_data
            fitbit_data = authd_client.intraday_time_series('activities/{}'.format(tmp_name), DATE, detail_level='1min')
            if len(str(fitbit_data)) > 0:
                save_path = os.path.join(analysis_path, 'activities', USER_ID, kind_of_data)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = os.path.join(save_path, '{}.txt'.format(DATE))
                with open(file_name, 'w') as f:
                    json.dump(fitbit_data, f, indent=4)

        data_sleep = authd_client.sleep(date=DATE)
        if len(str(data_sleep)) > 0:
            save_path = os.path.join(analysis_path, 'activities', USER_ID, 'sleep')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = os.path.join(save_path, '{}.txt'.format(DATE))
            with open(file_name, 'w') as f:
                json.dump(data_sleep, f, indent=4)
        time.sleep(15)

if __name__ in '__main__':
    main()
