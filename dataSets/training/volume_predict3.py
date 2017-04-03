#coding=utf-8
'''
不同于volume_predict和volume_predict2的建模方式，现在用时间序列相似性考虑

建模思路：
从数据的单纯折线图上可以看出每一天的序列是非常相似的，也就是说每一天序列的趋势大体相同，可能会出现在趋势附近上下不稳定波动
而除了之前利用之前两个小时的非线性关系预测当前20分钟以外，我们还能从横向考虑，考虑前一周每天8点到8点20对但前有什么影响。
本来想直接用AIC估计出ARIMA参数预测的，但是考虑序列的时间跨度太短（只有29天），所以尝试其它的建模方式。
再考虑到每天的车流分布曲线大体相似（同一个收费站，同一个方向），我们可以尝试找到和当天6点到8点相似的历史数据（历史数据也只考虑6点到8点），
最相似的历史数据的后两个小时数据即为最终的预测数据。
例如：我们的到9月19日上午6点到8点（只考虑6点到8点的数据）和10月19日上午6点到8点相似度最高，那么可以用9月19日上午8点到10点的数据做为
10月19日上午8点到10点的预测值
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

    # 承载量：1-默认客车，2-默认货车，3-默认货车，4-默认客车
    # 承载量大于等于5的为货运汽车，所有承载量为0的车都类型不明
    volume_df = volume_df.sort_values(by="vehicle_model")
    vehicle_model0 = volume_df[volume_df['vehicle_model'] == 0].fillna("No")
    vehicle_model1 = volume_df[volume_df['vehicle_model'] == 1].fillna("passenger")
    vehicle_model2 = volume_df[volume_df['vehicle_model'] == 2].fillna("cargo")
    vehicle_model3 = volume_df[volume_df['vehicle_model'] == 3].fillna("cargo")
    vehicle_model4 = volume_df[volume_df['vehicle_model'] == 4].fillna("passenger")
    vehicle_model5 = volume_df[volume_df['vehicle_model'] >= 5].fillna("cargo")
    volume_df = pd.concat([vehicle_model0, vehicle_model1, vehicle_model2,
                           vehicle_model3, vehicle_model4, vehicle_model5])

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

    # 承载量：1-默认客车，2-默认货车，3-默认货车，4-默认客车
    # 承载量大于等于5的为货运汽车，所有承载量为0的车都类型不明
    volume_test = volume_test.sort_values(by="vehicle_model")
    vehicle_model0 = volume_test[volume_test['vehicle_model'] == 0].fillna("No")
    vehicle_model1 = volume_test[volume_test['vehicle_model'] == 1].fillna("passenger")
    vehicle_model2 = volume_test[volume_test['vehicle_model'] == 2].fillna("cargo")
    vehicle_model3 = volume_test[volume_test['vehicle_model'] == 3].fillna("cargo")
    vehicle_model4 = volume_test[volume_test['vehicle_model'] == 4].fillna("passenger")
    vehicle_model5 = volume_test[volume_test['vehicle_model'] >= 5].fillna("cargo")
    volume_test = pd.concat([vehicle_model0, vehicle_model1, vehicle_model2,
                             vehicle_model3, vehicle_model4, vehicle_model5])
    return volume_df, volume_test

def modeling():
    volume_train, volume_test = preprocessing()
    result_df = pd.DataFrame()
    tollgate_list = ["1S", "2S", "3S"]
    for tollgate_id in tollgate_list:
        print tollgate_id

        # 创建之和流量，20分钟跨度有关系的训练集
        def divide_train_by_direction(volume_df):
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
            del volume_all_entry["vehicle_model"]
            volume_all_entry = volume_all_entry.resample("20T").sum()
            volume_all_entry = volume_all_entry.dropna()

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
                del volume_all_exit["vehicle_model"]
                volume_all_exit = volume_all_exit.resample("20T").sum()
                volume_all_exit = volume_all_exit.dropna()
            return volume_all_entry, volume_all_exit

        # 创建车流量预测集，20分钟跨度有关系的预测集
        def divide_test_by_direction(volume_df):
            volume_entry_test = volume_df[
                (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "entry")].copy()
            volume_entry_test["volume"] = 1
            volume_entry_test.index = volume_entry_test["time"]
            del volume_entry_test["time"]
            del volume_entry_test["tollgate_id"]
            del volume_entry_test["direction"]
            del volume_entry_test["vehicle_type"]
            del volume_entry_test["has_etc"]
            del volume_entry_test["vehicle_model"]
            volume_entry_test = volume_entry_test.resample("20T").sum()
            volume_entry_test = volume_entry_test.dropna()

            volume_exit_test = volume_df[
                (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "exit")].copy()
            if len(volume_exit_test) > 0:
                volume_exit_test["volume"] = 1
                volume_exit_test.index = volume_exit_test["time"]
                del volume_exit_test["time"]
                del volume_exit_test["tollgate_id"]
                del volume_exit_test["direction"]
                del volume_exit_test["vehicle_type"]
                del volume_exit_test["has_etc"]
                del volume_exit_test["vehicle_model"]
                volume_exit_test = volume_exit_test.resample("20T").sum()
                volume_exit_test = volume_exit_test.dropna()
            return volume_entry_test, volume_exit_test

        # 将数据分成每天6点到8点，15点到17点两拨数据
        # 每一天是一个Series
        def divide_by_time_slot(data_df, train=True):
            if len(data_df) == 0:
                return [],[],[],[]
            data_df["time"] = data_df.index
            data_df["month"] = data_df["time"].apply(lambda x: x.month)
            data_df["day"] = data_df["time"].apply(lambda x: x.day)
            data_df["hour"] = data_df["time"].apply(lambda x: x.hour)
            data_df["minute"] = data_df["time"].apply(lambda x: x.minute)
            # 6-8点
            train_6_8 = data_df[(data_df["hour"] >= 6) & (data_df["hour"] < 8)][["volume", "day", "month"]]
            train_6_8.index = data_df[(data_df["hour"] >= 6) & (data_df["hour"] < 8)]["time"]

            # 按天拆分成无数个元素
            if (train):
                train_lst_6_8 = [train_6_8[(train_6_8["day"] == i) & (train_6_8["month"] == 9)]["volume"]
                                for i in range(19, 31, 1)
                                if len(train_6_8[(train_6_8["day"] == i) &
                                                (train_6_8["month"] == 9)]["volume"]) > 0]
                train_lst_6_8 += [train_6_8[(train_6_8["day"] == i) & (train_6_8["month"] == 10)]["volume"]
                                for i in range(1, 18, 1)
                                if len(train_6_8[(train_6_8["day"] == i) &
                                                (train_6_8["month"] == 10)]["volume"]) > 0]
            else:
                # 预测集日期是10月18－10月24
                train_lst_6_8 = [train_6_8[(train_6_8["day"] == i) & (train_6_8["month"] == 10)]["volume"]
                                 for i in range(18, 25, 1)
                                 if len(train_6_8[(train_6_8["day"] == i) &
                                                  (train_6_8["month"] == 10)]["volume"]) > 0]
            predict_lst_6_8 = []
            if (train):
                # 如果是训练集，还得做8点到10点的数据
                train_8_10 = data_df[(data_df["hour"] >= 8) & (data_df["hour"] < 10)][["volume", "day", "month"]]
                train_8_10.index = data_df[(data_df["hour"] >= 8) & (data_df["hour"] < 10)]["time"]
                predict_lst_6_8 += [train_8_10[(train_8_10["day"] == i) & (train_8_10["month"] == 9)]["volume"]
                                    for i in range(19, 31, 1)
                                    if len(train_8_10[(train_8_10["day"] == i) &
                                                      (train_8_10["month"] == 9)]["volume"]) > 0]
                predict_lst_6_8 += [train_8_10[(train_8_10["day"] == i) & (train_8_10["month"] == 10)]["volume"]
                                    for i in range(1, 18, 1)
                                    if len(train_8_10[(train_8_10["day"] == i) &
                                                      (train_8_10["month"] == 10)]["volume"]) > 0]


            # 15-17点
            train_15_17 = data_df[(data_df["hour"] >= 15) & (data_df["hour"] < 17)][["volume", "day", "month"]]
            train_15_17.index = data_df[(data_df["hour"] >= 15) & (data_df["hour"] < 17)]["time"]

            if (train):
                train_lst_15_17 = [train_15_17[(train_15_17["day"] == i) & (train_15_17["month"] == 9)]["volume"]
                                   for i in range(19, 31, 1)
                                   if len(train_15_17[(train_15_17["day"] == i) &
                                                      (train_15_17["month"] == 9)]["volume"]) > 0]
                train_lst_15_17 += [train_15_17[(train_15_17["day"] == i) & (train_15_17["month"] == 10)]["volume"]
                                    for i in range(1, 18, 1)
                                    if len(train_15_17[(train_15_17["day"] == i) &
                                                       (train_15_17["month"] == 10)]["volume"]) > 0]
            else:
                train_lst_15_17 = [train_15_17[(train_15_17["day"] == i) & (train_15_17["month"] == 10)]["volume"]
                                   for i in range(18, 25, 1)
                                   if len(train_15_17[(train_15_17["day"] == i) &
                                                      (train_15_17["month"] == 10)]["volume"]) > 0]
            predict_lst_15_17 = []
            if (train):
                # 如果是训练集，还得做17点到19点的数据
                train_17_19 = data_df[(data_df["hour"] >= 17) & (data_df["hour"] < 19)][["volume", "day", "month"]]
                train_17_19.index = data_df[(data_df["hour"] >= 17) & (data_df["hour"] < 19)]["time"]
                predict_lst_15_17 += [train_17_19[(train_17_19["day"] == i) & (train_17_19["month"] == 9)]["volume"]
                                      for i in range(19, 31, 1)
                                      if len(train_17_19[(train_17_19["day"] == i) &
                                                         (train_17_19["month"] == 9)]["volume"]) > 0]
                predict_lst_15_17 += [train_17_19[(train_17_19["day"] == i) & (train_17_19["month"] == 10)]["volume"]
                                      for i in range(1, 18, 1)
                                      if len(train_17_19[(train_17_19["day"] == i) &
                                                         (train_17_19["month"] == 10)]["volume"]) > 0]

            return train_lst_6_8, predict_lst_6_8, train_lst_15_17, predict_lst_15_17


        def predict(train_X, train_y, test_X):
            def scorer(series1, series2):
                temp1 = pd.Series(series1.values)
                temp2 = pd.Series(series2.values)
                return ((temp1 - temp2) ** 2).mean()
            result = pd.DataFrame()
            for test in test_X:
                score_list = pd.Series([scorer(test, train) for train in train_X])
                score_list.index = range(len(score_list))
                predict_result = train_y[score_list.idxmin()]
                predict_result.name = test.index[0]
                predict_result.index = range(len(predict_result))
                predict_result = predict_result + (test.mean() - predict_result.mean())
                result = result.append(predict_result)
            return result

        def transform_result(data_df, direction, tollgate_id):
            result = pd.DataFrame()
            for i in range(len(data_df)):
                time_basic = data_df.index[i]
                for j in range(6, 12, 1):
                    time_window = "[" + str(pd.Timestamp(time_basic) + DateOffset(minutes=j * 20)) + "," + str(
                        pd.Timestamp(time_basic) + DateOffset(minutes=(j + 1) * 20)) + ")"
                    series = pd.Series({"tollgate_id": tollgate_id,
                                        "time_window": time_window,
                                        "direction": direction,
                                        "volume": round(data_df.iloc[i, j - 6], 2)})
                    series.name = i + j - 6
                    result = result.append(series)
            return result

        def transform_predict(data_df1, data_df2):
            result = pd.DataFrame()
            for i in range(len(data_df1)):
                result = result.append(data_df1.iloc[i, :])
                result = result.append(data_df2.iloc[i, :])
            return result

        train_entry, train_exit = divide_train_by_direction(volume_train)
        test_entry, test_exit = divide_test_by_direction(volume_test)
        entry_6_8, entry_predict_6_8, entry_15_17, entry_predict_15_17 = divide_by_time_slot(train_entry)
        exit_6_8, exit_predict_6_8, exit_15_17, exit_predict_15_17 = divide_by_time_slot(train_exit)
        test_entry_6_8, a, test_entry_15_17, b = divide_by_time_slot(test_entry, False)
        test_exit_6_8, a, test_exit_15_17, b = divide_by_time_slot(test_exit, False)
        test_predict_entry_6_8 = predict(entry_6_8, entry_predict_6_8, test_entry_6_8)
        test_predict_entry_15_17 = predict(entry_15_17, entry_predict_15_17, test_entry_15_17)
        test_predict_exit_6_8 = predict(exit_6_8, exit_predict_6_8, test_exit_6_8)
        test_predict_exit_15_17 = predict(exit_15_17, exit_predict_15_17, test_exit_15_17)
        temp1 = transform_predict(test_predict_entry_6_8, test_predict_entry_15_17)
        temp2 = transform_predict(test_predict_exit_6_8, test_predict_exit_15_17)
        result_df = result_df.append(transform_result(temp1, "entry", tollgate_id))
        result_df = result_df.append(transform_result(temp2, "exit", tollgate_id))

    return result_df

result = modeling()
result_df = pd.DataFrame()
result_df["tollgate_id"] = result["tollgate_id"].replace({"1S": 1, "2S": 2, "3S": 3})
result_df["time_window"] = result["time_window"]
result_df["direction"] = result["direction"].replace({"entry": 0, "exit": 1})
result_df['volume'] = result["volume"]
result_df.to_csv("volume_predict3_result.csv", encoding="utf8", index=None)