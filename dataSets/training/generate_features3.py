#coding=utf-8
'''
构造纯时间特征的训练集合预测集

在generate_feature2.py的基础上构造区分端口和方向的纯时间特征训练集和预测集

2015-05-08：时间特征的构造还要仔细斟酌一下，不需要再按20分钟为单位构造2个小时的特征了，就用当前时间特征预测流量就好

注意！！！！！！！
构造训练集的代码和预测的代码不能同时跑，因为会出现训练集覆盖现象，而预测代码不是一次性读取所有文件，所以有两种解决方法：
1. 用两套文件
2. 按顺序运行

'''

import pandas as pd
import numpy as np
from pandas.tseries.offsets import *

def preprocessing():
    '''
        预处理训练集
        '''
    volume_df = pd.read_csv("volume(table 6)_training.csv")

    # 替换所有有标签含义的数字
    volume_df['tollgate_id'] = volume_df['tollgate_id'].replace({1: "1S", 2: "2S", 3: "3S"})
    volume_df['direction'] = volume_df['direction'].replace({0: "entry", 1: "exit"})
    volume_df['has_etc'] = volume_df['has_etc'].replace({0: "No", 1: "Yes"})
    volume_df['vehicle_type'] = volume_df['vehicle_type'].replace({0: "passenger", 1: "cargo"})
    volume_df['time'] = volume_df['time'].apply(lambda x: pd.Timestamp(x))

    # 剔除10月1日至10月6日数据（每个收费站在该日期附近都有异常）
    volume_df = volume_df[(volume_df["time"] < pd.Timestamp("2016-10-01 00:00:00")) |
                          (volume_df["time"] > pd.Timestamp("2016-10-07 00:00:00"))]

    '''
    处理预测集
    '''
    volume_test = pd.read_csv("../testing_phase1/volume(table 6)_test1.csv")
    # 替换所有有标签含义的数字
    volume_test['tollgate_id'] = volume_test['tollgate_id'].replace({1: "1S", 2: "2S", 3: "3S"})
    volume_test['direction'] = volume_test['direction'].replace({0: "entry", 1: "exit"})
    volume_test['has_etc'] = volume_test['has_etc'].replace({0: "No", 1: "Yes"})
    volume_test['vehicle_type'] = volume_test['vehicle_type'].replace({0: "passenger", 1: "cargo"})
    volume_test['time'] = volume_test['time'].apply(lambda x: pd.Timestamp(x))

    return volume_df, volume_test


# 创建之和流量，20分钟跨度有关系的训练集
def divide_train_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    # entry
    volume_all_entry = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'entry')].copy()
    volume_all_entry['y'] = 1
    volume_all_entry.index = volume_all_entry["time"]
    del volume_all_entry["time"]
    del volume_all_entry["tollgate_id"]
    del volume_all_entry["direction"]
    del volume_all_entry["vehicle_type"]
    del volume_all_entry["has_etc"]
    del volume_all_entry["vehicle_model"]
    volume_all_entry = volume_all_entry.resample("20T").sum()
    volume_all_entry = volume_all_entry.fillna(0)


    # exit
    volume_all_exit = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'exit')].copy()
    if len(volume_all_exit) > 0:
        volume_all_exit["y"] = 1
        volume_all_exit.index = volume_all_exit["time"]
        del volume_all_exit["time"]
        del volume_all_exit["tollgate_id"]
        del volume_all_exit["direction"]
        del volume_all_exit["vehicle_type"]
        del volume_all_exit["has_etc"]
        del volume_all_exit["vehicle_model"]
        volume_all_exit = volume_all_exit.resample("20T").sum()
        volume_all_exit = volume_all_exit.fillna(0)

    if entry_file_path:
        volume_all_entry.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        volume_all_exit.to_csv(exit_file_path, encoding="utf8")
    return volume_all_entry, volume_all_exit


# 在train_df的index基础上加上当前的时间特征（注意不是20分钟为单位的偏移量了）
def generate_time_features(data_df, offset, file_path=None):
    time_str_se = pd.Series(data_df.index)
    time_se = time_str_se.apply(lambda x: pd.Timestamp(x))
    time_se.index = time_se.values
    data_df["time"] = time_se + DateOffset(minutes=offset * 20)
    data_df["day_str"] = data_df["time"].apply(lambda x: str(x.day) + "D")
    data_df["hour_str"] = data_df["time"].apply(lambda x: str(x.hour) + "H")
    data_df["is_eight"] = data_df["time"].apply(lambda x: 1 if x.hour == 8 else 0)
    data_df["is_nine"] = data_df["time"].apply(lambda x: 1 if x.hour == 9 else 0)
    data_df["is_eighteen"] = data_df["time"].apply(lambda x: 1 if x.hour == 18 else 0)
    data_df["is_seventeen"] = data_df["time"].apply(lambda x: 1 if x.hour == 17 else 0)
    data_df["minute_str"] = data_df["time"].apply(lambda x: str(x.minute) + "M")
    data_df["week_str"] = data_df["time"].apply(lambda x: str(x.dayofweek) + "W")


    data_df["day"] = data_df["time"].apply(lambda x: x.day)
    data_df["hour"] = data_df["time"].apply(lambda x: x.hour)
    data_df["minute"] = data_df["time"].apply(lambda x: x.minute)
    data_df["week"] = data_df["time"].apply(lambda x: x.dayofweek)
    data_df["weekend"] = data_df["week"].apply(lambda x: 1 if x >= 5 else 0)
    del data_df["time"]
    if file_path:
        data_df.to_csv(file_path + ".csv")
    return data_df


# 创建车流量预测集，20分钟跨度有关系的预测集
def divide_test_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    entry_test = pd.DataFrame()
    # for date in range(18, 25, 1):
    #     entry_temp = pd.DataFrame()
    #     entry_arr = pd.period_range(start="2016-10-"+str(date)+" 8:00:00", end="2016-10-"+str(date)+" 10:00:00", freq="20T")
    #     entry_temp["time"] = entry_arr
    #     entry_test = entry_test.append(entry_temp, ignore_index=True)
    #
    #     entry_temp2 = pd.DataFrame()
    #     entry_arr2 = pd.period_range(start="2016-10-"+str(date)+" 17:00:00", end="2016-10-"+str(date)+" 19:00:00", freq="20T")
    #     entry_temp2["time"] = entry_arr2
    #     entry_test = entry_test.append(entry_temp2, ignore_index=True)
    time_lst1 = [pd.Timestamp("2016-10-"+str(i)+" 8:00:00") for i in range(18, 25, 1)]
    time_lst2 = [pd.Timestamp("2016-10-"+str(i)+" 17:00:00") for i in range(18, 25, 1)]
    time_lst = []
    for i in range(len(time_lst1)):
        time_lst.append(time_lst1[i])
        time_lst.append(time_lst2[i])
    entry_test["time"] = time_lst
    entry_test.index = entry_test["time"].apply(lambda x: str(x))

    exit_test = pd.DataFrame()
    if tollgate_id != "2S":
        exit_test["time"] = time_lst
        exit_test.index = exit_test["time"].apply(lambda x: str(x))

    return entry_test, exit_test

def generate_features():
    volume_train, volume_test = preprocessing()
    tollgate_list = ["1S", "2S", "3S"]
    for tollgate_id in tollgate_list:
        print tollgate_id

        record_entry_train, record_exit_train = divide_train_by_direction(volume_train, tollgate_id)
        record_entry_train = generate_time_features(record_entry_train, 0)
        record_entry_train.to_csv("./train&test5_zjw/volume_entry_train_" + tollgate_id + ".csv")
        if tollgate_id != "2S":
            record_exit_train = generate_time_features(record_exit_train, 0)
            record_exit_train.to_csv("./train&test5_zjw/volume_exit_train_" + tollgate_id + ".csv")

        record_entry_test, record_exit_test = divide_test_by_direction(volume_train, tollgate_id)
        for i in range(6):
            entry_test = generate_time_features(record_entry_test, i)
            entry_test.to_csv("./train&test5_zjw/volume_entry_test_" + tollgate_id + "offset" + str(i) + ".csv")
            if tollgate_id != "2S":
                exit_test = generate_time_features(record_exit_test, i)
                exit_test.to_csv("./train&test5_zjw/volume_exit_test_" + tollgate_id + "offset" + str(i) + ".csv")

generate_features()

