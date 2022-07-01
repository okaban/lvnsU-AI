import numpy as np
import datetime
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"

#=========================
# Get HR data
#=========================
def get_Data_activity(dataPath, date, data_collection):

    tmp_datetime = datetime.datetime.strptime(date, '%Y-%m-%d')

    date_list = [tmp_datetime + datetime.timedelta(minutes=tmp) for tmp in range(1440)]
    value_list = [0] * 1440
    if 'time' not in data_collection:
        data_collection['time'] = np.array([])
        data_collection['value'] = np.array([])
    data_collection['time'] = np.append(data_collection['time'], np.array(date_list))
    data_collection['value'] = np.append(data_collection['value'], np.array(value_list))

    with open(dataPath) as f:
        s = f.read()

    if s != '':
        tmp_s = s.replace("'", '"')
        tmpData = json.loads(tmp_s)

        kindOfData = 'activities-heart-intraday'  
        if kindOfData in tmpData:
            for oneMinute_data in tmpData[kindOfData]['dataset']:
                tmp = datetime.datetime.strptime(date + ' ' + oneMinute_data['time'], '%Y-%m-%d %H:%M:%S')
                data_collection['value'][data_collection['time'] == tmp] = oneMinute_data['value']

    return data_collection

#=========================
# Get Sleep data
#=========================
def get_Data_sleep(dataPath, data_collection):

    with open(dataPath) as f:
        s = f.read()

    tmp_s = s.replace("'", '"')
    tmp_s = tmp_s.replace('True', '"True"')
    tmp_s = tmp_s.replace('False', '"False"')
    tmpData = json.loads(tmp_s)

    if tmpData['sleep'] != []:
        for ii in range(len(tmpData['sleep'])):
            for sleep_data in tmpData['sleep'][ii]['levels']['data']:
                duration_minutes = sleep_data['seconds'] // 60
                start_datetime = datetime.datetime.strptime(sleep_data['dateTime'].split('.000')[0], '%Y-%m-%dT%H:%M:%S')

                for minute in range(duration_minutes):
                    tmp_datetime = start_datetime + datetime.timedelta(minutes=minute)
                    hour_min = datetime.datetime(tmp_datetime.year, tmp_datetime.month, tmp_datetime.day, tmp_datetime.hour, tmp_datetime.minute)

                    data_collection['sleep'][data_collection['time'] == hour_min] = int(1)

    return data_collection