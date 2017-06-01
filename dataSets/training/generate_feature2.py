#coding=utf-8
'''
构造纯时间特征的训练集合预测集

2015-05-08：时间特征的构造还要仔细斟酌一下，不需要再按20分钟为单位构造2个小时的特征了，就用当前时间特征预测流量就好

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
    volume_all_entry['volume'] = 1
    volume_all_entry.index = volume_all_entry["time"]
    del volume_all_entry["time"]
    del volume_all_entry["tollgate_id"]
    del volume_all_entry["direction"]
    del volume_all_entry["vehicle_type"]
    del volume_all_entry["has_etc"]
    volume_all_entry = volume_all_entry.resample("20T").sum()
    volume_all_entry = volume_all_entry.fillna(0)


    # exit
    volume_all_exit = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'exit')].copy()
    if len(volume_all_exit) > 0:
        volume_all_exit["volume"] = 1
        volume_all_exit.index = volume_all_exit["time"]
        del volume_all_exit["time"]
        del volume_all_exit["tollgate_id"]
        del volume_all_exit["direction"]
        del volume_all_exit["vehicle_type"]
        del volume_all_exit["has_etc"]
        volume_all_exit = volume_all_exit.resample("20T").sum()
        volume_all_exit = volume_all_exit.fillna(0)

    if entry_file_path:
        volume_all_entry.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        volume_all_exit.to_csv(exit_file_path, encoding="utf8")
    return volume_all_entry, volume_all_exit

# 在train_df的index基础上加上offset*20分钟的时间特征
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

# 整合每20分钟的特征，并计算以2个小时为单位的特征
def generate_train_features(data_df, new_index, offset, has_y=True, file_path=None):
    train_df = pd.DataFrame()
    for i in range(len(data_df) - 6 - offset):
        se_temp = pd.Series()
        # 删除9月和10月交界的数据，就是训练集的X和y所在时间点分别在两个月份的情况
        # month_left = data_df.index[i]
        # month_right = data_df.index[i + 6 + offset]
        # if month_left == 9 and month_right == 10:
        #     continue
        for k in range(6):
            se_temp = se_temp.append(data_df.iloc[i + k, :].copy())
        if has_y:
            se_temp = se_temp.append(pd.Series(data_df.iloc[i + 6 + offset, :]["volume"].copy()))
        se_temp.index = new_index
        se_temp.name = str(data_df.index[i])
        train_df = train_df.append(se_temp)
    return generate_time_features(train_df, 6 + offset, file_path)

# 创建训练集，总的要求就是以前两个小时数据为训练集，用迭代式预测方法
# 例如8点-10点的数据预测10点20,8点-10点20预测10点40……，每一次预测使用的都是独立的（可能模型一样）的模型
# 现在开始构建训练集
# 第一个训练集特征是所有两个小时（以20分钟为一个单位）的数据，因变量是该两小时之后20分钟的流量
# 第二个训练集，特征是所有两个小时又20分钟（以20分钟为一个单位）的数据，因变量是该两个小时之后20分钟的流量
# 以此类推训练12个GBDT模型，其中entry 6个，exit 6个
def generate_train(volume_entry, volume_exit, entry_file_path=None, exit_file_path=None):
    old_index = volume_entry.columns
    new_index = []
    for i in range(6):
        new_index += [item + "%d" % (i,) for item in old_index]
    new_index.append("y")

    entry_df_lst = []
    exit_df_lst = []
    for j in range(6):
        train_entry_df = generate_train_features(volume_entry.copy(), new_index, j, file_path=entry_file_path)
        entry_df_lst.append(train_entry_df)

    # 注意！！！！2号收费站只有entry方向没有exit方向
    if len(volume_exit) == 0:
        return entry_df_lst, [pd.DataFrame() for i in range(6)]

    for j in range(6):
        train_exit_df = generate_train_features(volume_exit.copy(), new_index, j, file_path=exit_file_path)
        exit_df_lst.append(train_exit_df)

    return entry_df_lst, exit_df_lst


# 创建车流量预测集，20分钟跨度有关系的预测集
def divide_test_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    entry_test = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "entry")].copy()
    entry_test["volume"] = 1
    entry_test.index = entry_test["time"]
    del entry_test["time"]
    del entry_test["tollgate_id"]
    del entry_test["direction"]
    del entry_test["vehicle_type"]
    del entry_test["has_etc"]
    entry_test = entry_test.resample("20T").sum()
    entry_test = entry_test.dropna()

    #exit
    exit_test = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "exit")].copy()
    if len(exit_test) > 0:
        exit_test["volume"] = 1
        exit_test.index = exit_test["time"]
        del exit_test["time"]
        del exit_test["tollgate_id"]
        del exit_test["direction"]
        del exit_test["vehicle_type"]
        del exit_test["has_etc"]
        exit_test = exit_test.resample("20T").sum()
        exit_test = exit_test.dropna()
    if entry_file_path:
        entry_test.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        exit_test.to_csv(exit_file_path, encoding="utf8")
    return entry_test, exit_test


def generate_test(volume_entry_test, volume_exit_test, tollgate_id, entry_file_path=None, exit_file_path=None):
    old_index = volume_entry_test.columns
    new_index = []
    for i in range(6):
        new_index += [item + "%d" % (i,) for item in old_index]

    # （entry方向）
    entry_df_lst = []
    test_entry_df = pd.DataFrame()
    i = 0
    while i < len(volume_entry_test) - 5:
        se_temp = pd.Series()
        for k in range(6):
            se_temp = se_temp.append(volume_entry_test.iloc[i + k, :])
        se_temp.index = new_index
        se_temp.name = str(volume_entry_test.index[i])
        test_entry_df = test_entry_df.append(se_temp)
        i += 6
    for i in range(6):
        if entry_file_path:
            test_entry_df = generate_time_features(test_entry_df, i + 6, entry_file_path + "offset" + str(i))
        else:
            test_entry_df = generate_time_features(test_entry_df, i + 6)
        entry_df_lst.append(test_entry_df.copy())

    # （exit方向）
    exit_df_lst = []
    test_exit_df = pd.DataFrame()
    if tollgate_id == "2S":
        return entry_df_lst, [pd.DataFrame() for i in range(6)]
    i = 0
    while i < len(volume_exit_test) - 5:
        se_temp = pd.Series()
        for k in range(6):
            se_temp = se_temp.append(volume_exit_test.iloc[i + k, :])
        se_temp.index = new_index
        se_temp.name = str(volume_exit_test.index[i])
        test_exit_df = test_exit_df.append(se_temp)
        i += 6
    for i in range(6):
        if exit_file_path:
            test_exit_df = generate_time_features(test_exit_df, i + 6, exit_file_path + "offset" + str(i))
        else:
            test_exit_df = generate_time_features(test_exit_df, i + 6)
        exit_df_lst.append(test_exit_df.copy())
    return entry_df_lst, exit_df_lst

def generate_features():
    volume_train, volume_test = preprocessing()
    tollgate_list = ["1S", "2S", "3S"]
    train_df_morning = [pd.DataFrame() for i in range(6)]
    train_df_afternoon = [pd.DataFrame() for i in range(6)]
    test_df_morning = [pd.DataFrame() for i in range(6)]
    test_df_afternoon = [pd.DataFrame() for i in range(6)]
    for tollgate_id in tollgate_list:
        print tollgate_id
        def add_labels(data_df, direction):
            for item in data_df:
                item["tollgate_id"] = tollgate_id
                item["direction"] = direction

        # def add_history(data_df):


        def train_filter_morning(data_df, offset):
            if data_df.shape[0] == 0:
                return data_df
            temp_df = data_df.copy()
            hour_offset = offset / 3
            minute_offset = (offset % 3) * 20
            append_data = temp_df[(temp_df["hour"] == 8 + hour_offset) & (temp_df["minute"] == minute_offset)]
            for i in range(10):
                temp_df = temp_df.append(append_data, ignore_index=True)
            return temp_df

        def train_filter_afternoon(data_df, offset):
            if data_df.shape[0] == 0:
                return data_df
            temp_df = data_df.copy()
            hour_offset = offset / 3
            minute_offset = (offset % 3) * 20
            append_data = temp_df[(temp_df["hour"] == 17 + hour_offset) & (temp_df["minute"] == minute_offset)]
            for i in range(10):
                temp_df = temp_df.append(append_data, ignore_index=True)
            return temp_df

        record_entry_train, record_exit_train = divide_train_by_direction(volume_train, tollgate_id)
        volume_entry_train, volume_exit_train = generate_train(record_entry_train, record_exit_train)
        record_entry_test, record_exit_test = divide_test_by_direction(volume_test, tollgate_id)
        volume_entry_test, volume_exit_test = generate_test(record_entry_test, record_exit_test, tollgate_id)

        add_labels(volume_entry_train, "entry")
        add_labels(volume_exit_train, "exit")
        train_df_morning = [train_df_morning[i].append(train_filter_morning(volume_entry_train[i], i))
                            for i in range(6)]
        train_df_morning = [train_df_morning[i].append(train_filter_morning(volume_exit_train[i], i))
                            for i in range(6)]
        train_df_afternoon = [train_df_afternoon[i].append(train_filter_afternoon(volume_entry_train[i], i))
                              for i in range(6)]
        train_df_afternoon = [train_df_afternoon[i].append(train_filter_afternoon(volume_exit_train[i], i))
                              for i in range(6)]

        add_labels(volume_entry_test, "entry")
        add_labels(volume_exit_test, "exit")
        test_df_morning = [test_df_morning[i].append(volume_entry_test[i][volume_entry_test[i]["hour"] < 12])
                           for i in range(6)]

        test_df_afternoon = [test_df_afternoon[i].append(volume_entry_test[i][volume_entry_test[i]["hour"] > 12])
                             for i in range(6)]
        if volume_exit_test[0].shape[0] > 0:
            test_df_morning = [test_df_morning[i].append(volume_exit_test[i][volume_exit_test[i]["hour"] < 12])
                           for i in range(6)]
            test_df_afternoon = [test_df_afternoon[i].append(volume_exit_test[i][volume_exit_test[i]["hour"] > 12])
                             for i in range(6)]

    for i in range(6):
        # train_df[i].to_csv("./train&test3_zjw/train_offset%d.csv" % (i, ))
        # test_df[i].to_csv("./train&test3_zjw/test_offset%d.csv" % (i, ))
        train_df_morning[i].to_csv("./train&test3_zjw/train_offset%d_morning.csv" % (i, ))
        train_df_afternoon[i].to_csv("./train&test3_zjw/train_offset%d_afternoon.csv" % (i, ))
        # test_df_morning[i].to_csv("./train&test3_zjw/test2_offset%d_morning.csv" % (i, ))
        # test_df_afternoon[i].to_csv("./train&test3_zjw/test2_offset%d_afternoon.csv" % (i, ))


generate_features()