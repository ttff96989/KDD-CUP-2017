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
    # volume_df = volume_df.sort_values(by="vehicle_model")
    # vehicle_model0_train = volume_df[volume_df['vehicle_model'] == 0].fillna("No")
    # vehicle_model1_train = volume_df[volume_df['vehicle_model'] == 1].fillna("passenger")
    # vehicle_model2_train = volume_df[volume_df['vehicle_model'] == 2].fillna("cargo")
    # vehicle_model3_train = volume_df[volume_df['vehicle_model'] == 3].fillna("cargo")
    # vehicle_model4_train = volume_df[volume_df['vehicle_model'] == 4].fillna("passenger")
    # vehicle_model5_train = volume_df[volume_df['vehicle_model'] >= 5].fillna("cargo")
    # volume_df = pd.concat([vehicle_model0_train, vehicle_model1_train, vehicle_model2_train,
    #                        vehicle_model3_train, vehicle_model4_train, vehicle_model5_train])
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
    # volume_test = volume_test.sort_values(by="vehicle_model")
    # vehicle_model0_test = volume_test[volume_test['vehicle_model'] == 0].fillna("No")
    # vehicle_model1_test = volume_test[volume_test['vehicle_model'] == 1].fillna("passenger")
    # vehicle_model2_test = volume_test[volume_test['vehicle_model'] == 2].fillna("cargo")
    # vehicle_model3_test = volume_test[volume_test['vehicle_model'] == 3].fillna("cargo")
    # vehicle_model4_test = volume_test[volume_test['vehicle_model'] == 4].fillna("passenger")
    # vehicle_model5_test = volume_test[volume_test['vehicle_model'] >= 5].fillna("cargo")
    # volume_test = pd.concat(
    #     [vehicle_model0_test, vehicle_model1_test, vehicle_model2_test,
    #      vehicle_model3_test, vehicle_model4_test, vehicle_model5_test])

    return volume_df, volume_test


# 创建之和流量，20分钟跨度有关系的训练集
def divide_train_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    # entry
    volume_all_entry = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df['direction'] == 'entry')].copy()
    volume_all_entry['volume'] = 1
    volume_all_entry["etc_count"] = volume_all_entry["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
    volume_all_entry["etc_model"] = volume_all_entry["etc_count"] * volume_all_entry["vehicle_model"]
    volume_all_entry["no_etc_count"] = volume_all_entry["has_etc"].apply(lambda x: 1 if x == "No" else 0)
    volume_all_entry["no_etc_model"] = volume_all_entry["no_etc_count"] * volume_all_entry["vehicle_model"]
    # entry方向不记录车辆类型，所以比exit少一些特征
    volume_all_entry["model02_count"] = volume_all_entry["vehicle_model"].apply(
        lambda x: 1 if x >= 0 and x <= 2 else 0)
    volume_all_entry["model02_model"] = volume_all_entry["vehicle_model"] * volume_all_entry["model02_count"]
    volume_all_entry["model35_count"] = volume_all_entry["vehicle_model"].apply(
        lambda x: 1 if x >= 3 and x <= 5 else 0)
    volume_all_entry["model35_model"] = volume_all_entry["vehicle_model"] * volume_all_entry["model35_count"]
    volume_all_entry["model67_count"] = volume_all_entry["vehicle_model"].apply(
        lambda x: 1 if x == 6 or x == 7 else 0)
    volume_all_entry["model67_model"] = volume_all_entry["vehicle_model"] * volume_all_entry["model67_count"]
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
        volume_all_exit["etc_count"] = volume_all_exit["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
        volume_all_exit["etc_model"] = volume_all_exit["etc_count"] * volume_all_exit["vehicle_model"]
        volume_all_exit["no_etc_count"] = volume_all_exit["has_etc"].apply(lambda x: 1 if x == "No" else 0)
        volume_all_exit["no_etc_model"] = volume_all_exit["no_etc_count"] * volume_all_exit["vehicle_model"]
        # 注意！！！！！！！！！！！
        # 只有exit方向才记录车辆类型
        volume_all_exit["cargo_count"] = volume_all_exit['vehicle_type'].apply(
            lambda x: 1 if x == "cargo" else 0)
        volume_all_exit["passenger_count"] = volume_all_exit['vehicle_type'].apply(
            lambda x: 1 if x == "passenger" else 0)
        # volume_all_exit["no_count"] = volume_all_exit['vehicle_type'].apply(lambda x: 1 if x == "No" else 0)
        volume_all_exit["cargo_model"] = volume_all_exit["cargo_count"] * volume_all_exit["vehicle_model"]
        volume_all_exit["passenger_model"] = volume_all_exit["passenger_count"] * \
                                             volume_all_exit["vehicle_model"]
        volume_all_exit.index = volume_all_exit["time"]
        del volume_all_exit["time"]
        del volume_all_exit["tollgate_id"]
        del volume_all_exit["direction"]
        del volume_all_exit["vehicle_type"]
        del volume_all_exit["has_etc"]
        volume_all_exit = volume_all_exit.resample("20T").sum()
        volume_all_exit = volume_all_exit.dropna(0)

        volume_all_exit["cargo_model_avg"] = volume_all_exit["cargo_model"] / volume_all_exit["cargo_count"]
        volume_all_exit["passenger_model_avg"] = volume_all_exit["passenger_model"] / volume_all_exit[
            "passenger_count"]
        volume_all_exit["vehicle_model_avg"] = volume_all_exit["vehicle_model"] / volume_all_exit["volume"]
        # volume_all_exit = volume_all_exit.fillna(0)
    if entry_file_path:
        volume_all_entry.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        volume_all_exit.to_csv(exit_file_path, encoding="utf8")
    return volume_all_entry, volume_all_exit


# 计算2个小时为单位的特征
# train_df就是整合后的特征，
# offset是从index开始偏移多少个单位
def generate_2hours_features(train_df, offset, file_path=None, has_type=False):
    # 加之前一定要判断空值，不然空值和数字相加还是空
    train_df["vehicle_all_model"] = train_df["vehicle_model0"] + train_df["vehicle_model1"] + \
                                    train_df["vehicle_model2"] + train_df["vehicle_model3"] + \
                                    train_df["vehicle_model4"] + train_df["vehicle_model5"]
    train_df["etc_all_model"] = train_df["etc_model0"] + train_df["etc_model1"] + train_df["etc_model2"] + \
                                train_df["etc_model3"] + train_df["etc_model4"] + train_df["etc_model5"]
    train_df["etc_all_count"] = train_df["etc_count0"] + train_df["etc_count1"] + train_df["etc_count2"] + \
                                train_df["etc_count3"] + train_df["etc_count4"] + train_df["etc_count5"]
    train_df["etc_avg_model"] = train_df["etc_all_model"] / train_df["etc_all_count"]

    if has_type:
        train_df["cargo_all_model"] = train_df["cargo_model0"] + train_df["cargo_model1"] + \
                                      train_df["cargo_model2"] + train_df["cargo_model3"] + \
                                      train_df["cargo_model4"] + train_df["cargo_model5"]
        train_df["passenger_all_model"] = train_df["passenger_model0"] + train_df["passenger_model1"] + \
                                          train_df["passenger_model2"] + train_df["passenger_model3"] + \
                                          train_df["passenger_model4"] + train_df["passenger_model5"]
        train_df["cargo_all_count"] = train_df["cargo_count0"] + train_df["cargo_count1"] + \
                                      train_df["cargo_count2"] + train_df["cargo_count3"] + \
                                      train_df["cargo_count4"] + train_df["cargo_count5"]
        train_df["passenger_all_count"] = train_df["passenger_count0"] + train_df["passenger_count1"] + \
                                          train_df["passenger_count2"] + train_df["passenger_count3"] + \
                                          train_df["passenger_count4"] + train_df["passenger_count5"]
        train_df["cargo_avg_model"] = train_df["cargo_all_model"] / train_df["cargo_all_count"]
        train_df["passenger_avg_model"] = train_df["passenger_all_model"] / train_df["passenger_all_count"]
    else:
        train_df["model02_all_count"] = train_df["model02_count0"] + train_df["model02_count1"] + \
                                        train_df["model02_count2"] + train_df["model02_count3"] + \
                                        train_df["model02_count4"] + train_df["model02_count5"]
        train_df["model35_all_count"] = train_df["model35_count0"] + train_df["model35_count1"] + \
                                        train_df["model35_count2"] + train_df["model35_count3"] + \
                                        train_df["model35_count4"] + train_df["model35_count5"]
        train_df["model67_all_count"] = train_df["model67_count0"] + train_df["model67_count1"] + \
                                        train_df["model67_count2"] + train_df["model67_count3"] + \
                                        train_df["model67_count4"] + train_df["model67_count5"]
        train_df["model02_all_model"] = train_df["model02_model0"] + train_df["model02_model1"] + \
                                        train_df["model02_model2"] + train_df["model02_model3"] + \
                                        train_df["model02_model4"] + train_df["model02_model5"]
        train_df["model35_all_model"] = train_df["model35_model0"] + train_df["model35_model1"] + \
                                        train_df["model35_model1"] + train_df["model35_model2"] + \
                                        train_df["model35_model3"] + train_df["model35_model4"]
        train_df["model67_all_model"] = train_df["model67_model0"] + train_df["model67_model1"] + \
                                        train_df["model67_model2"] + train_df["model67_model3"] + \
                                        train_df["model67_model4"] + train_df["model67_model5"]
        train_df["model02_avg_model"] = train_df["model02_all_model"] / train_df["model02_all_count"]
        train_df["model35_avg_model"] = train_df["model35_all_model"] / train_df["model35_all_count"]
        train_df["model67_avg_model"] = train_df["model67_all_model"] / train_df["model67_all_count"]
        train_df = train_df.fillna(0)

    train_df["volume_all"] = train_df["volume0"] + train_df["volume1"] + train_df["volume2"] + \
                             train_df["volume3"] + train_df["volume4"] + train_df["volume5"]
    train_df["vehicle_avg_model"] = train_df["vehicle_all_model"] / train_df["volume_all"]
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
def generate_train_features(data_df, new_index, offset, has_y=True, file_path=None, has_type=False):
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
    return generate_2hours_features(train_df, 6 + offset, file_path, has_type)


# 创建训练集，总的要求就是以前两个小时数据为训练集，用迭代式预测方法
# 例如8点-10点的数据预测10点20,8点-10点20预测10点40……，每一次预测使用的都是独立的（可能模型一样）的模型
# 现在开始构建训练集
# 第一个训练集特征是所有两个小时（以20分钟为一个单位）的数据，因变量是该两小时之后20分钟的流量
# 第二个训练集，特征是所有两个小时又20分钟（以20分钟为一个单位）的数据，因变量是该两个小时之后20分钟的流量
# 以此类推训练12个GBDT模型，其中entry 6个，exit 6个
def generate_train(volume_entry, volume_exit, entry_file_path=None, exit_file_path=None):
    old_index_entry = volume_entry.columns
    new_index_entry = []
    for i in range(6):
        new_index_entry += [item + "%d" % (i,) for item in old_index_entry]
    new_index_entry.append("y")

    entry_df_lst = []
    exit_df_lst = []

    def filter_error3(data_df):
        temp_df = data_df.copy()
        temp_df["time"] = temp_df.index
        temp_df["time"] = temp_df["time"].apply(pd.Timestamp)
        temp_df = temp_df[(temp_df["time"] < pd.Timestamp("2016-09-30 22:20:00")) |
                          (temp_df["time"] > pd.Timestamp("2016-10-07 00:00:00"))]
        del temp_df["time"]
        return temp_df

    def multi_sample(data_df, offset):
        temp_df = data_df.copy()
        hour_offset = offset / 3
        minute_offset = (offset % 3) * 20
        # 增加filter，只要下午的数据
        # temp_df = temp_df[temp_df["hour"] >= 14]
        append_data = temp_df[(temp_df["hour"] == 17 + hour_offset) & (temp_df["minute"] == minute_offset)]
        # print 'before appending : ' + str(temp_df.shape)
        for i in range(10):
            temp_df = temp_df.append(append_data, ignore_index=True)
        # print "after appending : " + str(temp_df.shape)
        return temp_df

    def multi_sample_morning(data_df, offset):
        temp_df = data_df.copy()
        hour_offset = offset / 3
        minute_offset = (offset % 3) * 20
        # 增加filter，只要上午的数据
        # temp_df = temp_df[temp_df["hour"] < 14]
        append_data = temp_df[(temp_df["hour"] == 8 + hour_offset) & (temp_df["minute"] == minute_offset)]
        # print 'before appending : ' + str(temp_df.shape)
        for i in range(10):
            temp_df = temp_df.append(append_data, ignore_index=True)
        # print "after appending : " + str(temp_df.shape)
        return temp_df

    for j in range(6):
        train_entry_df = generate_train_features(volume_entry, new_index_entry, j, file_path=None, has_type=False)
        train_entry_df = filter_error3(train_entry_df.fillna(0))
        train_morning = multi_sample_morning(train_entry_df, j)
        train_afternoon = multi_sample(train_entry_df, j)
        entry_df_lst.append([train_morning, train_afternoon])

    # 注意！！！！2号收费站只有entry方向没有exit方向
    if len(volume_exit) == 0:
        return entry_df_lst, [[pd.DataFrame(), pd.DataFrame()] for i in range(6)]

    old_index_exit = volume_exit.columns
    new_index_exit = []
    for i in range(6):
        new_index_exit += [item + "%d" % (i,) for item in old_index_exit]
    new_index_exit.append("y")
    for j in range(6):
        train_exit_df = generate_train_features(volume_exit, new_index_exit, j, file_path=None, has_type=True)
        train_exit_df = filter_error3(train_exit_df.fillna(0))
        train_morning = multi_sample_morning(train_exit_df, j)
        train_afternoon = multi_sample(train_exit_df, j)
        exit_df_lst.append([train_morning, train_afternoon])

    return entry_df_lst, exit_df_lst


# 创建车流量预测集，20分钟跨度有关系的预测集
def divide_test_by_direction(volume_df, tollgate_id, entry_file_path=None, exit_file_path=None):
    volume_entry_test = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "entry")].copy()
    volume_entry_test["volume"] = 1

    volume_entry_test["etc_count"] = volume_entry_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
    volume_entry_test["etc_model"] = volume_entry_test["etc_count"] * volume_entry_test["vehicle_model"]
    volume_entry_test["no_etc_count"] = volume_entry_test["has_etc"].apply(lambda x: 1 if x == "No" else 0)
    volume_entry_test["no_etc_model"] = volume_entry_test["no_etc_count"] * volume_entry_test["vehicle_model"]
    # 这里entry方向数据不记录车辆类型，所以特征稍微少一点
    volume_entry_test["model02_count"] = volume_entry_test["vehicle_model"].apply(
        lambda x: 1 if x >= 0 and x <= 2 else 0)
    volume_entry_test["model02_model"] = volume_entry_test["vehicle_model"] * volume_entry_test["model02_count"]
    volume_entry_test["model35_count"] = volume_entry_test["vehicle_model"].apply(
        lambda x: 1 if x >= 3 and x <= 5 else 0)
    volume_entry_test["model35_model"] = volume_entry_test["vehicle_model"] * volume_entry_test["model35_count"]
    volume_entry_test["model67_count"] = volume_entry_test["vehicle_model"].apply(
        lambda x: 1 if x == 6 or x == 7 else 0)
    volume_entry_test["model67_model"] = volume_entry_test["vehicle_model"] * volume_entry_test["model67_count"]
    volume_entry_test.index = volume_entry_test["time"]
    del volume_entry_test["time"]
    del volume_entry_test["tollgate_id"]
    del volume_entry_test["direction"]
    del volume_entry_test["vehicle_type"]
    del volume_entry_test["has_etc"]
    volume_entry_test = volume_entry_test.resample("20T").sum()
    volume_entry_test = volume_entry_test.dropna()

    volume_exit_test = volume_df[
        (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "exit")].copy()
    if len(volume_exit_test) > 0:
        volume_exit_test["volume"] = 1
        volume_exit_test["etc_count"] = volume_exit_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
        volume_exit_test["etc_model"] = volume_exit_test["etc_count"] * volume_exit_test["vehicle_model"]
        volume_exit_test["no_etc_count"] = volume_exit_test["has_etc"].apply(lambda x: 1 if x == "No" else 0)
        volume_exit_test["no_etc_model"] = volume_exit_test["no_etc_count"] * volume_exit_test[
            "vehicle_model"]
        volume_exit_test["cargo_count"] = volume_exit_test["vehicle_type"].apply(
            lambda x: 1 if x == "cargo" else 0)
        volume_exit_test["passenger_count"] = volume_exit_test["vehicle_type"].apply(
            lambda x: 1 if x == "passenger" else 0)
        # volume_exit_test["no_count"] = volume_exit_test["vehicle_type"].apply(lambda x: 1 if x == "No" else 0)
        volume_exit_test["cargo_model"] = volume_exit_test["cargo_count"] * volume_exit_test["vehicle_model"]
        volume_exit_test["passenger_model"] = volume_exit_test["passenger_count"] * volume_exit_test[
            "vehicle_model"]
        # volume_exit_test["model02_count"] = volume_exit_test["vehicle_model"].apply(
        #     lambda x: 1 if x >= 0 and x <= 2 else 0)
        # volume_exit_test["model35_count"] = volume_exit_test["vehicle_model"].apply(
        #     lambda x: 1 if x >= 3 and x <= 5 else 0)
        # volume_exit_test["model67_count"] = volume_exit_test["vehicle_model"].apply(
        #     lambda x: 1 if x == 6 or x == 7 else 0)
        volume_exit_test.index = volume_exit_test["time"]
        del volume_exit_test["time"]
        del volume_exit_test["tollgate_id"]
        del volume_exit_test["direction"]
        del volume_exit_test["vehicle_type"]
        del volume_exit_test["has_etc"]
        volume_exit_test = volume_exit_test.resample("20T").sum()
        volume_exit_test = volume_exit_test.dropna()
        volume_exit_test["cargo_model_avg"] = volume_exit_test["cargo_model"] / volume_exit_test["cargo_count"]
        volume_exit_test["passenger_model_avg"] = volume_exit_test["passenger_model"] / volume_exit_test[
            "passenger_count"]
        volume_exit_test["vehicle_model_avg"] = volume_exit_test["vehicle_model"] / volume_exit_test["volume"]
        volume_exit_test = volume_exit_test.fillna(0)

    if entry_file_path:
        volume_entry_test.to_csv(entry_file_path, encoding="utf8")
    if exit_file_path:
        volume_exit_test.to_csv(exit_file_path, encoding="utf8")
    return volume_entry_test, volume_exit_test


# 转换预测集，将预测集转换成与训练集格式相同的格式
def generate_test(volume_entry_test, volume_exit_test, tollgate_id, entry_file_path=None, exit_file_path=None):
    old_index_entry = volume_entry_test.columns
    new_index_entry = []
    for i in range(6):
        new_index_entry += [item + "%d" % (i,) for item in old_index_entry]

    # （entry方向）
    entry_df_lst = []
    test_entry_df = pd.DataFrame()
    i = 0
    while i < len(volume_entry_test) - 5:
        se_temp = pd.Series()
        for k in range(6):
            se_temp = se_temp.append(volume_entry_test.iloc[i + k, :])
        se_temp.index = new_index_entry
        se_temp.name = str(volume_entry_test.index[i])
        test_entry_df = test_entry_df.append(se_temp)
        i += 6
    test_entry_df = generate_2hours_features(test_entry_df, 0, has_type=False)
    for i in range(6):
        # if entry_file_path:
        #     test_entry_df = generate_time_features(test_entry_df, i + 6, entry_file_path + "offset" + str(i))
        # else:
        #     test_entry_df = generate_time_features(test_entry_df, i + 6)
        test_entry_df = generate_time_features(test_entry_df, i + 6)
        test_entry_df1 = test_entry_df[test_entry_df["hour"] < 12]
        test_entry_df2 = test_entry_df[test_entry_df["hour"] > 12]
        entry_df_lst.append([test_entry_df1, test_entry_df2])

    # （exit方向）
    exit_df_lst = []
    test_exit_df = pd.DataFrame()
    if tollgate_id == "2S":
        return entry_df_lst, [[pd.DataFrame(), pd.DataFrame()] for i in range(6)]

    old_index_exit = volume_exit_test.columns
    new_index_exit = []
    for i in range(6):
        new_index_exit += [item + "%d" % (i,) for item in old_index_exit]
    i = 0
    while i < len(volume_exit_test) - 5:
        se_temp = pd.Series()
        for k in range(6):
            se_temp = se_temp.append(volume_exit_test.iloc[i + k, :])
        se_temp.index = new_index_exit
        se_temp.name = str(volume_exit_test.index[i])
        test_exit_df = test_exit_df.append(se_temp)
        i += 6
    test_exit_df = generate_2hours_features(test_exit_df, 0, has_type=True)

    for i in range(6):
        # if exit_file_path:
        #     test_exit_df = generate_time_features(test_exit_df, i + 6, exit_file_path + "offset" + str(i))
        # else:
        #     test_exit_df = generate_time_features(test_exit_df, i + 6)
        test_exit_df = generate_time_features(test_exit_df, i + 6)
        test_exit_df1 = test_exit_df[test_exit_df["hour"] < 12]
        test_exit_df2 = test_exit_df[test_exit_df["hour"] > 12]
        exit_df_lst.append([test_exit_df1, test_exit_df2])
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
                for i in range(2):
                    item[i]["tollgate_id"] = tollgate_id
                    item[i]["direction"] = direction

        record_entry_train, record_exit_train = divide_train_by_direction(volume_train, tollgate_id)
        volume_entry_train, volume_exit_train = generate_train(record_entry_train, record_exit_train)
        add_labels(volume_entry_train, "entry")
        add_labels(volume_exit_train, "exit")
        train_df_morning = [train_df_morning[i].append(volume_entry_train[i][0], i)
                            for i in range(6)]
        train_df_morning = [train_df_morning[i].append(volume_exit_train[i][0], i)
                            for i in range(6)]
        train_df_afternoon = [train_df_afternoon[i].append(volume_entry_train[i][1], i)
                              for i in range(6)]
        train_df_afternoon = [train_df_afternoon[i].append(volume_exit_train[i][1], i)
                              for i in range(6)]

        record_entry_test, record_exit_test = divide_test_by_direction(volume_test, tollgate_id)
        volume_entry_test, volume_exit_test = generate_test(record_entry_test, record_exit_test, tollgate_id)
        add_labels(volume_entry_test, "entry")
        add_labels(volume_exit_test, "exit")
        test_df_morning = [test_df_morning[i].append(volume_entry_test[i][0])
                           for i in range(6)]

        test_df_afternoon = [test_df_afternoon[i].append(volume_entry_test[i][0])
                             for i in range(6)]
        if volume_exit_test[0][0].shape[0] > 0:
            test_df_morning = [test_df_morning[i].append(volume_exit_test[i][1])
                               for i in range(6)]
            test_df_afternoon = [test_df_afternoon[i].append(volume_exit_test[i][1])
                                 for i in range(6)]

    for i in range(6):
        # train_df[i].to_csv("./train&test4_zjw/train_offset%d.csv" % (i, ))
        # test_df[i].to_csv("./train&test4_zjw/test_offset%d.csv" % (i, ))
        train_df_morning[i].to_csv("./train&test4_zjw/train_offset%d_morning.csv" % (i,))
        train_df_afternoon[i].to_csv("./train&test4_zjw/train_offset%d_afternoon.csv" % (i,))
        test_df_morning[i].to_csv("./train&test4_zjw/test_offset%d_morning.csv" % (i, ))
        test_df_afternoon[i].to_csv("./train&test4_zjw/test_offset%d_afternoon.csv" % (i, ))


generate_features()