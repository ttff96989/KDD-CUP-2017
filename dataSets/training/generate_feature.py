#coding=utf-8
'''
仿照volume_predict2代码的特征构造代码

不区分tollgate和direction，将这些做为1维标签特征
但是还是要区分offset时段，即分成六个模型

待优化：

待提交优化:
1. 删除噪声点，将10月1号到6号的数据作为噪声点剔除掉，因为原始训练集和预测集的比例为10000：60，除掉10月1日到6日的数据
   应该能够提高模型的泛化能力，

已优化:
1. 加特征：使用电子桩的车的总重量，平均重量，不适用电子桩的总重量，平均重量

2. resample之后的dropna修改成fillna(0)，因为存在训练集中20分钟内车流量为空的情况，此时resanmple后的结果同样是NaN（已经优化，没运行）
'''
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from pandas.tseries.offsets import *
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor

# description of the feature:
# Traffic Volume through the Tollgates
# time           datatime        the time when a vehicle passes the tollgate
# tollgate_id    string          ID of the tollgate
# direction      string           0:entry, 1:exit
# vehicle_model  int             this number ranges from 0 to 7, which indicates the capacity of the vehicle(bigger the higher)
# has_etc        string          does the vehicle use ETC (Electronic Toll Collection) device? 0: No, 1: Yes
# vehicle_type   string          vehicle type: 0-passenger vehicle, 1-cargo vehicle

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
    vehicle_model0_train = volume_df[volume_df['vehicle_model'] == 0].fillna("No")
    vehicle_model1_train = volume_df[volume_df['vehicle_model'] == 1].fillna("passenger")
    vehicle_model2_train = volume_df[volume_df['vehicle_model'] == 2].fillna("cargo")
    vehicle_model3_train = volume_df[volume_df['vehicle_model'] == 3].fillna("cargo")
    vehicle_model4_train = volume_df[volume_df['vehicle_model'] == 4].fillna("passenger")
    vehicle_model5_train = volume_df[volume_df['vehicle_model'] >= 5].fillna("cargo")
    volume_df = pd.concat([vehicle_model0_train, vehicle_model1_train, vehicle_model2_train,
                           vehicle_model3_train, vehicle_model4_train, vehicle_model5_train])
    # volume_df["vehicle_type"] = volume_df["vehicle_type"].fillna("No")

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
    vehicle_model0_test = volume_test[volume_test['vehicle_model'] == 0].fillna("No")
    vehicle_model1_test = volume_test[volume_test['vehicle_model'] == 1].fillna("passenger")
    vehicle_model2_test = volume_test[volume_test['vehicle_model'] == 2].fillna("cargo")
    vehicle_model3_test = volume_test[volume_test['vehicle_model'] == 3].fillna("cargo")
    vehicle_model4_test = volume_test[volume_test['vehicle_model'] == 4].fillna("passenger")
    vehicle_model5_test = volume_test[volume_test['vehicle_model'] >= 5].fillna("cargo")
    volume_test = pd.concat(
        [vehicle_model0_test, vehicle_model1_test, vehicle_model2_test,
         vehicle_model3_test, vehicle_model4_test, vehicle_model5_test])

    return volume_df, volume_test


# 创建之和流量，20分钟跨度有关系的训练集
def divide_train_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    # entry
    volume_all_entry = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'entry')].copy()
    volume_all_entry['volume'] = 1
    volume_all_entry['cargo_count'] = volume_all_entry['vehicle_type'].apply(lambda x: 1 if x == "cargo" else 0)
    volume_all_entry['passenger_count'] = volume_all_entry['vehicle_type'].apply(
        lambda x: 1 if x == "passenger" else 0)
    volume_all_entry['no_count'] = volume_all_entry['vehicle_type'].apply(lambda x: 1 if x == "No" else 0)
    volume_all_entry["etc_count"] = volume_all_entry["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
    volume_all_entry["cargo_model"] = volume_all_entry["cargo_count"] * volume_all_entry["vehicle_model"]
    volume_all_entry["passenger_model"] = volume_all_entry["passenger_count"] * volume_all_entry[
        "vehicle_model"]
    volume_all_entry["etc_model"] = volume_all_entry["etc_count"] * volume_all_entry["vehicle_model"]
    volume_all_entry.index = volume_all_entry["time"]
    del volume_all_entry["time"]
    del volume_all_entry["tollgate_id"]
    del volume_all_entry["direction"]
    del volume_all_entry["vehicle_type"]
    del volume_all_entry["has_etc"]
    volume_all_entry = volume_all_entry.resample("20T").sum()
    volume_all_entry = volume_all_entry.fillna(0)
    volume_all_entry["cargo_model_avg"] = volume_all_entry["cargo_model"] / volume_all_entry["cargo_count"]
    volume_all_entry["passenger_model_avg"] = volume_all_entry["passenger_model"] / volume_all_entry[
        "passenger_count"]
    volume_all_entry["etc_model_avg"] = volume_all_entry["etc_model"] / volume_all_entry["etc_count"]
    volume_all_entry["vehicle_model_avg"] = volume_all_entry["vehicle_model"] / volume_all_entry["volume"]
    volume_all_entry = volume_all_entry.fillna(0)

    # exit
    volume_all_exit = volume_df[
           (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'exit')].copy()
    if len(volume_all_exit) > 0:
        volume_all_exit["volume"] = 1
        volume_all_exit["cargo_count"] = volume_all_exit['vehicle_type'].apply(
                    lambda x: 1 if x == "cargo" else 0)
        volume_all_exit["passenger_count"] = volume_all_exit['vehicle_type'].apply(
                    lambda x: 1 if x == "passenger" else 0)
        volume_all_exit["no_count"] = volume_all_exit['vehicle_type'].apply(lambda x: 1 if x == "No" else 0)
        volume_all_exit["etc_count"] = volume_all_exit["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
        volume_all_exit["cargo_model"] = volume_all_exit["cargo_count"] * volume_all_exit["vehicle_model"]
        volume_all_exit["passenger_model"] = volume_all_exit["passenger_count"] * \
                                                    volume_all_exit["vehicle_model"]
        volume_all_exit["etc_model"] = volume_all_exit["etc_count"] * volume_all_exit["vehicle_model"]
        volume_all_exit.index = volume_all_exit["time"]
        del volume_all_exit["time"]
        del volume_all_exit["tollgate_id"]
        del volume_all_exit["direction"]
        del volume_all_exit["vehicle_type"]
        del volume_all_exit["has_etc"]
        volume_all_exit = volume_all_exit.resample("20T").sum()
        volume_all_exit = volume_all_exit.fillna(0)
        volume_all_exit["cargo_model_avg"] = volume_all_exit["cargo_model"] / volume_all_exit["cargo_count"]
        volume_all_exit["passenger_model_avg"] = volume_all_exit["passenger_model"] / volume_all_exit[
                    "passenger_count"]
        volume_all_exit["etc_model_avg"] = volume_all_exit["etc_model"] / volume_all_exit["etc_count"]
        volume_all_exit["vehicle_model_avg"] = volume_all_exit["vehicle_model"] / volume_all_exit["volume"]
        volume_all_exit = volume_all_exit.fillna(0)
    if entry_file_path:
        volume_all_entry.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        volume_all_exit.to_csv(exit_file_path, encoding="utf8")
    return volume_all_entry, volume_all_exit


# 计算2个小时为单位的特征
# train_df就是整合后的特征，
# offset是从index开始偏移多少个单位
def generate_2hours_features(train_df, offset, file_path=None):
    train_df["vehicle_all_model"] = train_df["vehicle_model0"] + train_df["vehicle_model1"] + \
                                    train_df["vehicle_model2"] + train_df["vehicle_model3"] + \
                                    train_df["vehicle_model4"] + train_df["vehicle_model5"]
    train_df["cargo_all_model"] = train_df["cargo_model0"] + train_df["cargo_model1"] + \
                                    train_df["cargo_model2"] + train_df["cargo_model3"] + \
                                    train_df["cargo_model4"] + train_df["cargo_model5"]
    train_df["passenger_all_model"] = train_df["passenger_model0"] + train_df["passenger_model1"] + \
                                        train_df["passenger_model2"] + train_df["passenger_model3"] + \
                                        train_df["passenger_model4"] + train_df["passenger_model5"]
    train_df["no_all_count"] = train_df["no_count0"] + train_df["no_count1"] + train_df["no_count2"] + \
                                train_df["no_count3"] + train_df["no_count4"] + train_df["no_count5"]
    train_df["cargo_all_count"] = train_df["cargo_count0"] + train_df["cargo_count1"] + \
                                    train_df["cargo_count2"] + train_df["cargo_count3"] + \
                                    train_df["cargo_count4"] + train_df["cargo_count5"]
    train_df["passenger_all_count"] = train_df["passenger_count0"] + train_df["passenger_count1"] + \
                                        train_df["passenger_count2"] + train_df["passenger_count3"] + \
                                        train_df["passenger_count4"] + train_df["passenger_count5"]
    train_df["volume_all"] = train_df["volume0"] + train_df["volume1"] + train_df["volume2"] + \
                                     train_df["volume3"] + train_df["volume4"] + train_df["volume5"]
    train_df["etc_all_count"] = train_df["etc_count0"] + train_df["etc_count1"] + train_df["etc_count2"] + \
                                    train_df["etc_count3"] + train_df["etc_count4"] + train_df["etc_count5"]
    train_df["vehicle_all_model_avg"] = train_df["vehicle_all_model"] / train_df["volume_all"]
    train_df["cargo_all_model_avg"] = train_df["cargo_all_model"] / train_df["cargo_all_count"]
    train_df["passenger_all_model_avg"] = train_df["passenger_all_model"] / train_df["passenger_all_count"]
    # 二次方 三次方 开方 运算
    train_df["vehicle_all_model_avg_S2"] = train_df["vehicle_all_model_avg"] * train_df["vehicle_all_model_avg"]
    train_df["vehicle_all_model_avg_S3"] = train_df["vehicle_all_model_avg"] * \
                                                   train_df["vehicle_all_model_avg"] * train_df["vehicle_all_model_avg"]
    train_df["vehicle_all_model_avg_sqrt"] = np.sqrt(train_df["vehicle_all_model_avg"])
    train_df["vehicle_model_avg5_S2"] = train_df["vehicle_model_avg5"] * train_df["vehicle_model_avg5"]
    train_df["vehicle_model_avg5_S3"] = train_df["vehicle_model_avg5"] * \
                                                train_df["vehicle_model_avg5"] * train_df["vehicle_model_avg5"]
    train_df["vehicle_model_avg5_sqrt"] = np.sqrt(train_df["vehicle_model_avg5"])
    train_df["vehicle_model_avg4_S2"] = train_df["vehicle_model_avg4"] * train_df["vehicle_model_avg4"]
    train_df["vehicle_model_avg4_S3"] = train_df["vehicle_model_avg4"] *\
                                                train_df["vehicle_model_avg4"] * train_df["vehicle_model_avg4"]
    train_df["vehicle_model_avg4_sqrt"] = np.sqrt(train_df["vehicle_model_avg4"])
    train_df["vehicle_model_avg3_S2"] = train_df["vehicle_model_avg3"] * train_df["vehicle_model_avg3"]
    train_df["vehicle_model_avg3_S3"] = train_df["vehicle_model_avg3"] * \
                                                train_df["vehicle_model_avg3"] * train_df["vehicle_model_avg3"]
    train_df["vehicle_model_avg3_sqrt"] = np.sqrt(train_df["vehicle_model_avg3"])
    train_df["vehicle_model_avg2_S2"] = train_df["vehicle_model_avg2"] * train_df["vehicle_model_avg2"]
    train_df["vehicle_model_avg2_S3"] = train_df["vehicle_model_avg2"] * \
                                                train_df["vehicle_model_avg2"] * train_df["vehicle_model_avg2"]
    train_df["vehicle_model_avg2_sqrt"] = np.sqrt(train_df["vehicle_model_avg2"])
    train_df["vehicle_model_avg1_S2"] = train_df["vehicle_model_avg1"] * train_df["vehicle_model_avg1"]
    train_df["vehicle_model_avg1_S3"] = train_df["vehicle_model_avg1"] * \
                                                train_df["vehicle_model_avg1"] * train_df["vehicle_model_avg1"]
    train_df["vehicle_model_avg1_sqrt"] = np.sqrt(train_df["vehicle_model_avg1"])
    train_df["vehicle_model_avg0_S2"] = train_df["vehicle_model_avg0"] * train_df["vehicle_model_avg0"]
    train_df["vehicle_model_avg0_S3"] = train_df["vehicle_model_avg0"] * \
                                                train_df["vehicle_model_avg0"] * train_df["vehicle_model_avg0"]
    train_df["passenger_all_model_avg_S2"] = train_df["passenger_all_model_avg"] * train_df["passenger_all_model_avg"]
    train_df["passenger_all_model_avg_S3"] = train_df["passenger_all_model_avg"]\
                                                     * train_df["passenger_all_model_avg"] * train_df["passenger_all_model_avg"]
    train_df["no_all_count_S2"] = train_df["no_all_count"] * train_df["no_all_count"]
    train_df["no_all_count_S3"] = train_df["no_all_count"] * train_df["no_all_count"] * train_df["no_all_count"]
    train_df["no_all_count_sqrt"] = np.sqrt(train_df["no_all_count"])
    train_df["no_count4_S2"] = train_df["no_count4"] * train_df["no_count4"]
    train_df["no_count4_S3"] = train_df["no_count4"] * train_df["no_count4"] * train_df["no_count4"]
    train_df["no_count4_sqrt"] = np.sqrt(train_df["no_count4"])
    train_df["volume1_S2"] = train_df["volume1"] * train_df["volume1"]
    train_df["volume1_S3"] = train_df["volume1"] * train_df["volume1"] * train_df["volume1"]
    train_df["volume1_sqrt"] = np.sqrt(train_df["volume1"])
    train_df["no_count5_S2"] = train_df["no_count5"] * train_df["no_count5"]
    train_df["no_count5_S3"] = train_df["no_count5"] * train_df["no_count5"] * train_df["no_count5"]
    train_df["no_count5_sqrt"] = np.sqrt(train_df["no_count5"])
    train_df["volume2_S2"] = train_df["volume2"] * train_df["volume2"]
    train_df["volume2_S3"] = train_df["volume2"] * train_df["volume2"] * train_df["volume2"]
    train_df["volume2_sqrt"] = np.sqrt(train_df["volume2"])
    train_df["volume3_S2"] = train_df["volume3"] * train_df["volume3"]
    train_df["volume3_S3"] = train_df["volume3"] * train_df["volume3"] * train_df["volume3"]
    train_df["volume3_sqrt"] = np.sqrt(train_df["volume3"])
    train_df["volume_all_S2"] = train_df["volume_all"] * train_df["volume_all"]
    train_df["volume_all_S3"] = train_df["volume_all"] * train_df["volume_all"] * train_df["volume_all"]
    train_df["volume_all_sqrt"] = np.sqrt(train_df["volume_all"])
    train_df["volume4_S2"] = train_df["volume4"] * train_df["volume4"]
    train_df["volume4_S3"] = train_df["volume4"] * train_df["volume4"] * train_df["volume4"]
    train_df["volume4_sqrt"] = np.sqrt(train_df["volume4"])
    train_df["volume5_S2"] = train_df["volume5"] * train_df["volume5"]
    train_df["volume5_S3"] = train_df["volume5"] * train_df["volume5"] * train_df["volume5"]
    train_df["volume5_sqrt"] = np.sqrt(train_df["volume5"])
    if offset >= 6:
        if file_path:
            train_df = generate_time_features(train_df, offset, file_path + "offset" + str(offset - 6))
        else:
            train_df = generate_time_features(train_df, offset)
    elif file_path:
        train_df.to_csv(file_path + ".csv")
    return train_df.fillna(0)


# 在train_df的index基础上加上offset*20分钟的时间特征
def generate_time_features(data_df, offset, file_path=None):
    time_str_se = pd.Series(data_df.index)
    time_se = time_str_se.apply(lambda x: pd.Timestamp(x))
    time_se.index = time_se.values
    data_df["time"] = time_se + DateOffset(minutes=offset * 20)
    # data_df["day"] = data_df["time"].apply(lambda x: str(x.day) + "D")
    # data_df["hour"] = data_df["time"].apply(lambda x: str(x.hour) + "H")
    data_df["is_eight"] = data_df["time"].apply(lambda x: 1 if x.hour == 8 else 0)
    data_df["is_nine"] = data_df["time"].apply(lambda x: 1 if x.hour == 9 else 0)
    data_df["is_eighteen"] = data_df["time"].apply(lambda x: 1 if x.hour == 18 else 0)
    data_df["is_seventeen"] = data_df["time"].apply(lambda x: 1 if x.hour == 17 else 0)
    data_df["minute"] = data_df["time"].apply(lambda x: str(x.minute) + "M")
    data_df["week"] = data_df["time"].apply(lambda x: str(x.dayofweek) + "W")
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
    return generate_2hours_features(train_df, 6 + offset, file_path)


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
        train_entry_df = generate_train_features(volume_entry, new_index, j, file_path=entry_file_path)
        entry_df_lst.append(train_entry_df)

    # 注意！！！！2号收费站只有entry方向没有exit方向
    if len(volume_exit) == 0:
        return entry_df_lst, [pd.DataFrame() for i in range(6)]

    for j in range(6):
        train_exit_df = generate_train_features(volume_exit, new_index, j, file_path=exit_file_path)
        exit_df_lst.append(train_exit_df)

    return entry_df_lst, exit_df_lst


# 创建车流量预测集，20分钟跨度有关系的预测集
def divide_test_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    entry_test = volume_df[
            (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "entry")].copy()
    entry_test["volume"] = 1
    entry_test["cargo_count"] = entry_test["vehicle_type"].apply(lambda x: 1 if x == "cargo" else 0)
    entry_test["passenger_count"] = entry_test["vehicle_type"].apply(
        lambda x: 1 if x == "passenger" else 0)
    entry_test["no_count"] = entry_test["vehicle_type"].apply(lambda x: 1 if x == "No" else 0)
    entry_test["etc_count"] = entry_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
    entry_test["cargo_model"] = entry_test["cargo_count"] * entry_test["vehicle_model"]
    entry_test["passenger_model"] = entry_test["passenger_count"] * entry_test[
        "vehicle_model"]
    entry_test["etc_model"] = entry_test["etc_count"] * entry_test["vehicle_model"]
    entry_test.index = entry_test["time"]
    del entry_test["time"]
    del entry_test["tollgate_id"]
    del entry_test["direction"]
    del entry_test["vehicle_type"]
    del entry_test["has_etc"]
    entry_test = entry_test.resample("20T").sum()
    entry_test = entry_test.dropna()
    entry_test["cargo_model_avg"] = entry_test["cargo_model"] / entry_test["cargo_count"]
    entry_test["passenger_model_avg"] = entry_test["passenger_model"] / entry_test[
        "passenger_count"]
    entry_test["etc_model_avg"] = entry_test["etc_model"] / entry_test["etc_count"]
    entry_test["vehicle_model_avg"] = entry_test["vehicle_model"] / entry_test["volume"]
    entry_test = entry_test.fillna(0)

    exit_test = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "exit")].copy()
    if len(exit_test) > 0:
        exit_test["volume"] = 1
        exit_test["cargo_count"] = exit_test["vehicle_type"].apply(
            lambda x: 1 if x == "cargo" else 0)
        exit_test["passenger_count"] = exit_test["vehicle_type"].apply(
            lambda x: 1 if x == "passenger" else 0)
        exit_test["no_count"] = exit_test["vehicle_type"].apply(lambda x: 1 if x == "No" else 0)
        exit_test["etc_count"] = exit_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
        exit_test["cargo_model"] = exit_test["cargo_count"] * exit_test["vehicle_model"]
        exit_test["passenger_model"] = exit_test["passenger_count"] * exit_test[
            "vehicle_model"]
        exit_test["etc_model"] = exit_test["etc_count"] * exit_test["vehicle_model"]
        exit_test.index = exit_test["time"]
        del exit_test["time"]
        del exit_test["tollgate_id"]
        del exit_test["direction"]
        del exit_test["vehicle_type"]
        del exit_test["has_etc"]
        exit_test = exit_test.resample("20T").sum()
        exit_test = exit_test.dropna()
        exit_test["cargo_model_avg"] = exit_test["cargo_model"] / exit_test["cargo_count"]
        exit_test["passenger_model_avg"] = exit_test["passenger_model"] / exit_test[
           "passenger_count"]
        exit_test["etc_model_avg"] = exit_test["etc_model"] / exit_test["etc_count"]
        exit_test["vehicle_model_avg"] = exit_test["vehicle_model"] / exit_test["volume"]
        exit_test = exit_test.fillna(0)
    if entry_file_path:
        entry_test.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        exit_test.to_csv(exit_file_path, encoding="utf8")
    return entry_test, exit_test


# 转换预测集，将预测集转换成与训练集格式相同的格式
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
    test_entry_df = generate_2hours_features(test_entry_df, 0)
    for i in range(6):
        if entry_file_path:
            test_entry_df = generate_time_features(test_entry_df, i + 6, entry_file_path + "offset" + str(i))
        else:
            test_entry_df = generate_time_features(test_entry_df, i + 6)
        entry_df_lst.append(test_entry_df)

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
    test_exit_df = generate_2hours_features(test_exit_df, 0)
    for i in range(6):
        if exit_file_path:
            test_exit_df = generate_time_features(test_exit_df, i + 6, exit_file_path + "offset" + str(i))
        else:
            test_exit_df = generate_time_features(test_exit_df, i + 6)
        exit_df_lst.append(test_exit_df)
    return entry_df_lst, exit_df_lst


def generate_features():
    volume_train, volume_test = preprocessing()
    tollgate_list = ["1S", "2S", "3S"]
    train_df = [pd.DataFrame() for i in range(6)]
    test_df = [pd.DataFrame() for i in range(6)]
    for tollgate_id in tollgate_list:
        print tollgate_id

        def add_labels(data_df, direction):
            for item in data_df:
                item["tollgate_id"] = tollgate_id
                item["direction"] = direction

        record_entry_train, record_exit_train = divide_train_by_direction(volume_train, tollgate_id)
        volume_entry_train, volume_exit_train = generate_train(record_entry_train, record_exit_train)
        add_labels(volume_entry_train, "entry")
        add_labels(volume_exit_train, "exit")
        train_df = [train_df[i].append(volume_entry_train[i]) for i in range(6)]
        train_df = [train_df[i].append(volume_exit_train[i]) for i in range(6)]

        record_entry_test, record_exit_test = divide_test_by_direction(volume_test, tollgate_id)
        volume_entry_test, volume_exit_test = generate_test(record_entry_test, record_exit_test, tollgate_id)
        add_labels(volume_entry_test, "entry")
        add_labels(volume_exit_test, "exit")
        test_df = [test_df[i].append(volume_entry_test[i]) for i in range(6)]
        test_df = [test_df[i].append(volume_exit_test[i]) for i in range(6)]

    for i in range(6):
        train_df[i].to_csv("./train&test4_zjw/train_offset%d.csv" % (i, ))
        test_df[i].to_csv("./train&test4_zjw/test_offset%d.csv" % (i, ))


generate_features()