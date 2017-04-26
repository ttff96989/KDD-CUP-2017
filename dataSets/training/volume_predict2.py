#coding=utf-8
'''
在volume_predict的基础上改进模型

建模思路：
创建训练集，总的要求就是以前两个小时数据为训练集，用迭代式预测方法
例如8点-10点的数据预测10点20,8点-10点20预测10点40……，每一次预测使用的都是独立的（可能模型一样）的模型
现在开始构建训练集
第一个训练集特征是所有两个小时（以20分钟为一个单位）的数据，因变量是该两小时之后20分钟的流量
第二个训练集，特征是所有两个小时又20分钟（以20分钟为一个单位）的数据，因变量是该两个小时之后20分钟的流量
以此类推训练12个GBDT模型，其中entry 6个，exit 6个

待优化思路：
1. 该模型想说明的问题是当前待预测时段的车流只和之前两小时车流有线性（或非线性）关系，这个认识其实比较局限，可以尝试
   换一个角度思考，当前时刻的车流量也可能和之前一个月同一时段车流量呈线性（或非线性）关系

2. 如何证明分开考虑收费站比将收费站全部整合到一起效果好，如果将收费站整合到一起的话，那么就不对收费站id，出入方向做分类

3. 时间序列相似度不靠谱，将所有特征丢到GBDT模型中训练，从特征评分中我们发现：当前客流还是和之前客流有非常大的相关性，而
   日期只是个简单的调整参数，我感觉这是因为训练集规模小导致了某些特征被弱化。因此把所有特征都丢进GBDT里不能单独体现时间
   对车流的影响。解决办法有两种：第一种是使用特征权重，这种方法暂时不太会用，而且权重值大小是个问题；第二种是单纯将时间
   特征提取出来，这样得到的GBDT相当于考虑了大量的时间因素。再将两种模型线性结合（或者stacking）结合得到最终结果

4. 可以用样本自相关函数判断下一时段的车流到底和之前几个20分钟线性相关（因为不一定要前2个小时来预测下一个20分钟），样本自
   相关函数用statsmodel库（感觉不一定靠谱）

5. 尝试构造线性模型的特征（平方，三次方，开方等），最大化线性模型准确率

6. 尝试使用神经网络建模

7. 尝试将所有模型用stacking融合

8. 不需要删除国庆7天的数据，只需要删除国庆和国庆前后连接点的数据就可以了。原因：考虑到删除国庆数据会造成大幅度的过拟合，
   所以删除噪声点可以提高成绩，但是删除的方式需要斟酌。

优化思路：
1. 根据题目所给评价函数，如果将y转换成log(y)，那么损失函数可以朝lad方向梯度下降（过程已经大致证明了），而特征的log处理
   不影响CART的回归结果，所以对所有车流量（不论特征还是因变量都做log计算）。如果使用其他非树形结构模型需要考虑是否要对
   所有数据做log计算

2. 增加特征，之前只考虑20分钟内的车流量情况，现在加上在20分钟内的总载重量，平均载重量；货车数量，货车总载重量，货车平均
   载重量；客车数量，客车总载重量，客车平均载重量；使用电子桩的车数（2个小时6个时段，每个时段有10维特征，一共60维）；
   2小时内总载重量，平均载重量，货车数量，货车总载重量，货车平均载重量，客车数量，客车总载重量，客车平均载重量（9维）；
   。再加上待预测时段的时间信息，包括月，日，时，分

3. 从观察数据得出来的结论，在10月1日附近（具体哪几天不记得了）全天的数据相对于其他日期的数据很像噪声点，可以尝试剔除那
   几天的数据

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
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

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
        def divide_train_by_direction(volume_df, entry_file_path=None, exit_file_path=None):
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
            volume_all_entry.index = volume_all_entry["time"]
            del volume_all_entry["time"]
            del volume_all_entry["tollgate_id"]
            del volume_all_entry["direction"]
            del volume_all_entry["vehicle_type"]
            del volume_all_entry["has_etc"]
            volume_all_entry = volume_all_entry.resample("20T").sum()
            volume_all_entry = volume_all_entry.dropna()
            volume_all_entry["cargo_model_avg"] = volume_all_entry["cargo_model"] / volume_all_entry["cargo_count"]
            volume_all_entry["passenger_model_avg"] = volume_all_entry["passenger_model"] / volume_all_entry[
                "passenger_count"]
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
                volume_all_exit.index = volume_all_exit["time"]
                del volume_all_exit["time"]
                del volume_all_exit["tollgate_id"]
                del volume_all_exit["direction"]
                del volume_all_exit["vehicle_type"]
                del volume_all_exit["has_etc"]
                volume_all_exit = volume_all_exit.resample("20T").sum()
                volume_all_exit = volume_all_exit.dropna()
                volume_all_exit["cargo_model_avg"] = volume_all_exit["cargo_model"] / volume_all_exit["cargo_count"]
                volume_all_exit["passenger_model_avg"] = volume_all_exit["passenger_model"] / volume_all_exit[
                    "passenger_count"]
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
            if offset >= 6:
                train_df = generate_time_features(train_df, offset, file_path)
            elif file_path:
                train_df.to_csv(file_path)
            return train_df.fillna(0)

        # 在train_df的index基础上加上offset*20分钟的时间特征
        def generate_time_features(data_df, offset, file_path=None):
            time_str_se = pd.Series(data_df.index)
            time_se = time_str_se.apply(lambda x: pd.Timestamp(x))
            time_se.index = time_se.values
            data_df["time"] = time_se + DateOffset(minutes=offset * 20)
            data_df["month"] = data_df["time"].apply(lambda x: x.month)
            data_df["day"] = data_df["time"].apply(lambda x: x.day)
            data_df["hour"] = data_df["time"].apply(lambda x: x.hour)
            data_df["minute"] = data_df["time"].apply(lambda x: x.minute)
            del data_df["time"]
            if file_path:
                data_df.to_csv(file_path)
            return data_df

        # 整合每20分钟的特征，并计算以2个小时为单位的特征
        def generate_features(data_df, new_index, offset, has_y=True, file_path=None):
            train_df = pd.DataFrame()
            for i in range(len(data_df) - 6 - offset):
                se_temp = pd.Series()
                for k in range(6):
                    se_temp = se_temp.append(data_df.iloc[i + k, :].copy())
                if has_y:
                    se_temp = se_temp.append(pd.Series(data_df.iloc[i + 6 + offset, :]["volume"].copy()))
                se_temp.index = new_index
                se_temp.name = str(data_df.index[i])
                train_df = train_df.append(se_temp)
            return generate_2hours_features(train_df, 6 + offset, file_path)

        # 生成gbdt模型
        def gbdt_model(train_X, train_y):
            best_rate = 0.1
            best_n_estimator = 3000
            param_grid = [
                {'max_depth': [3], 'min_samples_leaf': [10],
                 'learning_rate': [best_rate + 0.01 * i for i in range(-2, 4, 1)],
                 'loss': ['lad'],
                 'n_estimators': [best_n_estimator + i * 200 for i in range(-2, 3, 1)],
                 'max_features': [1.0]}
            ]
            # param_grid = [
            #     {'max_depth':[3], 'min_samples_leaf':[10],
            #      'learning_rate':[0.1], 'loss':['lad'], 'n_estimators':[3000], 'max_features':[1.0]}
            # ]

            # 这是交叉验证的评分函数
            def scorer(estimator, X, y):
                predict_arr = estimator.predict(X)
                y_arr = y
                result = (np.abs(1 - np.exp(predict_arr - y_arr))).sum() / len(y)
                return result

            model = GradientBoostingRegressor()
            clf = GridSearchCV(model, param_grid, refit=True, scoring=scorer)
            clf.fit(train_X, train_y)
            print "Best GBDT param is :", clf.best_params_
            return clf.best_estimator_

        # 生成Lasso线性模型
        def Lasso_model(train_X, train_y):
            skewness = train_X.apply(lambda x: skew(x))
            skewness = skewness[abs(skewness) > 0.5]
            # print(str(skewness.shape[0]) + " skewed numerical features to log transform")
            skewed_features = skewness.index
            train_X[skewed_features] = np.log1p(train_X[skewed_features])
            stdSc = StandardScaler()
            train_X = stdSc.fit_transform(train_X)
            lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                                    0.3, 0.6, 1],
                            max_iter=50000, cv=10)
            lasso.fit(train_X, train_y)
            alpha = lasso.alpha_
            print("Best alpha :", alpha)
            return lasso, train_X

        # 生成Ridge线性模型
        def Ridge_model(train_X, train_y):
            skewness = train_X.apply(lambda x: skew(x))
            skewness = skewness[abs(skewness) > 0.5]
            # print(str(skewness.shape[0]) + " skewed numerical features to log transform")
            skewed_features = skewness.index
            train_X[skewed_features] = np.log1p(train_X[skewed_features])
            stdSc = StandardScaler()
            train_X = stdSc.fit_transform(train_X)
            ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
            ridge.fit(train_X, train_y)
            alpha = ridge.alpha_
            print("Best alpha :", alpha)

            print("Try again for more precision with alphas centered around " + str(alpha))
            ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                                    alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                                    alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                            cv=10)
            ridge.fit(train_X, train_y)
            alpha = ridge.alpha_
            print("Best alpha :", alpha)
            return ridge, train_X

        # 创建训练集，总的要求就是以前两个小时数据为训练集，用迭代式预测方法
        # 例如8点-10点的数据预测10点20,8点-10点20预测10点40……，每一次预测使用的都是独立的（可能模型一样）的模型
        # 现在开始构建训练集
        # 第一个训练集特征是所有两个小时（以20分钟为一个单位）的数据，因变量是该两小时之后20分钟的流量
        # 第二个训练集，特征是所有两个小时又20分钟（以20分钟为一个单位）的数据，因变量是该两个小时之后20分钟的流量
        # 以此类推训练12个GBDT模型，其中entry 6个，exit 6个
        def generate_models(volume_entry, volume_exit, entry_file_path=None, exit_file_path=None):
            old_index = volume_entry.columns
            new_index = []
            for i in range(6):
                new_index += [item + "%d" % (i, ) for item in old_index]
            new_index.append("y")

            # 这是用训练集做预测时的评分函数
            def scorer2(estimator, X, y):
                predict_arr = estimator.predict(X)
                result = (np.abs(1 - np.exp(predict_arr - y))).sum()
                return result

            models_entry = []
            train_entry_len = 0
            train_entry_score = 0
            for j in range(6):
                train_df = generate_features(volume_entry, new_index, j, file_path=entry_file_path)
                train_df = train_df[train_df["y"] > 0]
                train_y = np.log(1 + train_df["y"].fillna(0))
                del train_df["y"]
                train_X = train_df.fillna(0)
                train_entry_len += len(train_y)
                best_estimator = gbdt_model(train_X, train_y)
                train_entry_score += scorer2(best_estimator, train_X, train_y)
                models_entry.append(best_estimator)
            print "Best Score is :", train_entry_score / train_entry_len

            # 注意！！！！2号收费站只有entry方向没有exit方向
            if len(volume_exit) == 0:
                return models_entry, []

            models_exit = []
            train_exit_len = 0
            train_exit_score = 0
            for j in range(6):
                train_df = generate_features(volume_exit, new_index, j, file_path=exit_file_path)
                train_df = train_df[train_df["y"] > 0]
                train_y = np.log(1 + train_df["y"].fillna(0))
                del train_df["y"]
                train_X = train_df.fillna(0)
                best_estimator = gbdt_model(train_X, train_y)
                train_exit_len += len(train_y)
                train_exit_score += scorer2(best_estimator, train_X, train_y)
                models_exit.append(best_estimator)
            print "Best Score is :", train_exit_score / train_exit_len

            return models_entry, models_exit

        # 创建车流量预测集，20分钟跨度有关系的预测集
        def divide_test_by_direction(volume_df, entry_file_path=None, exit_file_path=None):
            volume_entry_test = volume_df[
                (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "entry")].copy()
            volume_entry_test["volume"] = 1
            volume_entry_test["cargo_count"] = volume_entry_test["vehicle_type"].apply(lambda x: 1 if x == "cargo" else 0)
            volume_entry_test["passenger_count"] = volume_entry_test["vehicle_type"].apply(
                lambda x: 1 if x == "passenger" else 0)
            volume_entry_test["no_count"] = volume_entry_test["vehicle_type"].apply(lambda x: 1 if x == "No" else 0)
            volume_entry_test["etc_count"] = volume_entry_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
            volume_entry_test["cargo_model"] = volume_entry_test["cargo_count"] * volume_entry_test["vehicle_model"]
            volume_entry_test["passenger_model"] = volume_entry_test["passenger_count"] * volume_entry_test[
                "vehicle_model"]
            volume_entry_test.index = volume_entry_test["time"]
            del volume_entry_test["time"]
            del volume_entry_test["tollgate_id"]
            del volume_entry_test["direction"]
            del volume_entry_test["vehicle_type"]
            del volume_entry_test["has_etc"]
            volume_entry_test = volume_entry_test.resample("20T").sum()
            volume_entry_test = volume_entry_test.dropna()
            volume_entry_test["cargo_model_avg"] = volume_entry_test["cargo_model"] / volume_entry_test["cargo_count"]
            volume_entry_test["passenger_model_avg"] = volume_entry_test["passenger_model"] / volume_entry_test[
                "passenger_count"]
            volume_entry_test["vehicle_model_avg"] = volume_entry_test["vehicle_model"] / volume_entry_test["volume"]
            volume_entry_test = volume_entry_test.fillna(0)

            volume_exit_test = volume_df[
                (volume_df['tollgate_id'] == tollgate_id) & (volume_df["direction"] == "exit")].copy()
            if len(volume_exit_test) > 0:
                volume_exit_test["volume"] = 1
                volume_exit_test["cargo_count"] = volume_exit_test["vehicle_type"].apply(
                    lambda x: 1 if x == "cargo" else 0)
                volume_exit_test["passenger_count"] = volume_exit_test["vehicle_type"].apply(
                    lambda x: 1 if x == "passenger" else 0)
                volume_exit_test["no_count"] = volume_exit_test["vehicle_type"].apply(lambda x: 1 if x == "No" else 0)
                volume_exit_test["etc_count"] = volume_exit_test["has_etc"].apply(lambda x: 1 if x == "Yes" else 0)
                volume_exit_test["cargo_model"] = volume_exit_test["cargo_count"] * volume_exit_test["vehicle_model"]
                volume_exit_test["passenger_model"] = volume_exit_test["passenger_count"] * volume_exit_test[
                    "vehicle_model"]
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
        def predict(volume_entry_test, volume_exit_test, models_entry, models_exit,
                    entry_file_path=None, exit_file_path=None):
            old_index = volume_entry_test.columns
            new_index = []
            for i in range(6):
                new_index += [item + "%d" % (i, ) for item in old_index]

            # （entry方向）
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
            predict_test_entry = pd.DataFrame()
            for i in range(6):
                test_entry_df = generate_time_features(test_entry_df, i + 6, entry_file_path)
                test_y = models_entry[i].predict(test_entry_df)
                predict_test_entry[i] = np.exp(test_y) - 1
            predict_test_entry.index = test_entry_df.index

            # （exit方向）
            test_exit_df = pd.DataFrame()
            if len(models_exit) == 0:
                return predict_test_entry, test_exit_df
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
            predict_test_exit = pd.DataFrame()
            for i in range(6):
                test_exit_df = generate_time_features(test_exit_df, i + 6, exit_file_path)
                test_y = models_exit[i].predict(test_exit_df)
                predict_test_exit[i] = np.exp(test_y) - 1
            predict_test_exit.index = test_exit_df.index
            return predict_test_entry, predict_test_exit

        # 将预测数据转换成输出文件的格式
        def transform_predict(predict_original, direction, tollgate_id):
            result = pd.DataFrame()
            for i in range(len(predict_original)):
                time_basic = predict_original.index[i]
                for j in range(6, 12, 1):
                    time_window = "[" + str(pd.Timestamp(time_basic) + DateOffset(minutes=j * 20)) + "," + str(
                        pd.Timestamp(time_basic) + DateOffset(minutes=(j + 1) * 20)) + ")"
                    series = pd.Series({"tollgate_id": tollgate_id,
                                        "time_window": time_window,
                                        "direction": direction,
                                        "volume": "%.2f" % (predict_original.iloc[i, j - 6])})
                    series.name = i + j - 6
                    result = result.append(series)
            return result

        entry_train_file = "./train&test_zjw/volume_entry_train_%s.csv" % (tollgate_id, )
        exit_train_file = "./train&test_zjw/volume_exit_train_%s.csv" % (tollgate_id, )
        volume_entry_train, volume_exit_train = divide_train_by_direction(volume_train)
        models_entry, models_exit = generate_models(volume_entry_train,
                                                    volume_exit_train)
        entry_test_file = "./train&test_zjw/volume_entry_test_%s.csv" % (tollgate_id, )
        exit_test_file = "./train&test_zjw/volume_exit_test_%s.csv" % (tollgate_id, )
        entry_test, exit_test = divide_test_by_direction(volume_test)
        predict_original_entry, predict_original_exit = predict(entry_test,
                                                                exit_test,
                                                                models_entry,
                                                                models_exit)
        result_df = result_df.append(transform_predict(predict_original_entry, "entry", tollgate_id))
        result_df = result_df.append(transform_predict(predict_original_exit, "exit", tollgate_id))

    return result_df

result = modeling()
result_df = pd.DataFrame()
result_df["tollgate_id"] = result["tollgate_id"].replace({"1S": 1, "2S": 2, "3S": 3})
result_df["time_window"] = result["time_window"]
result_df["direction"] = result["direction"].replace({"entry": 0, "exit": 1})
result_df['volume'] = result["volume"]
result_df.to_csv("volume_predict2_result.csv", encoding="utf8", index=None)