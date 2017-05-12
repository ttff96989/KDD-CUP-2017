#coding=utf-8
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso, LinearRegression
from math import sqrt
from pandas.tseries.offsets import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb

'''
和stack.py的区别是不区分方向和出口

2017-05-08: 发现不区分方向和出口效果并不理想，不论是用stacking还是在stacking的基础上加上均值模型
所以打算退回到分端口和方向的模型中，用stacking * （纯时间特征模型 + 常规模型）

待优化：

1. Adaboost + linerRegression

待提交优化：

1. 将5折提高到8折

2. 去掉均值模型，改用纯时间因子作为特征的训练集和预测集，用stacking后替换掉均值模型


已优化：

1. 优化均值模型，(volumn0 + ... + volumn5) / n 变成 (log1p(volumn0) + ... + log1p(volumn5)) / n

2. 增加两个惩罚粒度不同的XGBoost

结论：

1. 加上纯时间特征的模型，准确率确实有提高从0.1930到0.1882

2. 5折升级为10折效果也有提高，从0.1882到0.1847

3. 纯时间特征模型和不区分出口和方向的模型效果效果差不多，可见不分方向和出口是不可能的……

4. 不分方向和端口的纯时间特征模型准确率要比分端口和方向的时间模型准确率高很多

'''

TARGET = 'y'
NFOLDS = 5
SEED = 0
NROWS = None
SUBMISSION_FILE = '../input/sample_submission.csv'

et_params = {
        'n_jobs': 16,
        'n_estimators': 1000,
        'max_features': 0.5,
        'max_depth': 4,
        'min_samples_leaf': 2,
    }

rf_params = {
        'n_jobs': 16,
        'n_estimators': 1000,
        'max_features': 0.2,
        'max_depth': 4,
        'min_samples_leaf': 2,
    }

rd_params = {
        'alpha': 10
    }

ls_params = {
        'alpha': 0.002,
        'max_iter': 5000
}

gbdt_params = {
    'max_depth': 3,
    'min_samples_leaf': 1,
    'learning_rate': 0.1,
    'loss': 'lad',
    'n_estimators': 3000,
    'max_features': 1.0
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}

xgb_params2 = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500,
    'lambda': 0.3
}

ada_param = {
    'base_estimator': DecisionTreeRegressor(max_depth=4),
    'n_estimators': 300
}

ada_param2 = {
    'base_estimator': LinearRegression(),
    'n_estimators': 300
}

mean_param = {

}

result_df = pd.DataFrame()
model_score_dic = {}

model_used_name = ["GB", "RF", "XGB", "XGB2", "ADA", "LS", "ET", "RD"]

def predict1(offset):
    ## Load the data ##
    train = pd.read_csv("./train&test4_zjw/train_offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    test = pd.read_csv("./train&test4_zjw/test_offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    train = train.dropna()
    test_index = test.index
    test_tollgate = test.tollgate_id.values
    test_direction = test.direction.values

    ## Preprocessing ##

    y_train = np.log(train[TARGET] + 1)

    train.drop([TARGET], axis=1, inplace=True)

    all_data = pd.concat((train.copy(), test.copy()))

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    stdSc = StandardScaler()
    all_data[numeric_feats] = stdSc.fit_transform(all_data[numeric_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:

    # creating matrices for sklearn:

    x_train = np.array(all_data[:train.shape[0]])
    x_test = np.array(all_data[train.shape[0]:])

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    class Mean_Model(object):
        def __init__(self, random_state=None):
            pass

        def fit(self, X_train, y_train):
            pass

        def predict(self, X_test):
            volume_index = ["volume0", "volume1", "volume2", "volume3", "volume4", "volume5"]
            result = np.zeros(len(X_test))
            for index in volume_index:
                result += np.log1p(X_test.loc[:, index].values)
            result /= len(volume_index)
            return result

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def clf_type(self):
            return type(self.clf)

    class XgbWrapper(object):
        def __init__(self, seed=0, params=None):
            self.param = params
            self.param["seed"] = seed
            self.nrounds = params.pop("nrounds", 250)

        def train(self, x_train, y_train):
            dtrain = xgb.DMatrix(x_train, label=y_train)
            self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

        def predict(self, x):
            return self.gbdt.predict(xgb.DMatrix(x))

        def clf_type(self):
            return None

    def get_oof(clf):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            if clf.clf_type() and clf.clf_type() == Mean_Model:
                oof_train = clf.predict(train.copy())
                oof_test_skf[i, :] = clf.predict(test.copy())
            else:
                clf.train(x_tr, y_tr)

                oof_train[test_index] = clf.predict(x_te)
                oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        # print oof_train
        # print oof_test
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    # 可以无限增加元模型，然后增加模型组合的可能性
    model_name_lst = ["GB", "RF", "XGB", "XGB2", "ADA", "LS", "ET", "RD"]
    model_lst = [GradientBoostingRegressor, RandomForestRegressor, None, None,
                 AdaBoostRegressor, Lasso, ExtraTreesRegressor, Ridge]
    model_params = [gbdt_params, rf_params, xgb_params, xgb_params2,
                    ada_param, ls_params, et_params, rd_params]
    model2_name = ["ET", "RF", "GB", "ADA", "XGB", "XGB2"]
    model2_lst = [ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, None, None]
    model2_params = [et_params, rf_params, gbdt_params, ada_param, xgb_params, xgb_params2]
    model_used_idx = [[0, 1, 2],
                      [0, 1, 2, 3],
                      [0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 4, 5],
                      [0, 1, 2, 3, 4, 5, 6],
                      [0, 1, 2, 3, 4, 5, 6, 7],
                      [1, 2, 3, 4, 5, 6, 7],
                      [2, 3, 4, 5, 6, 7],
                      [3, 4, 5, 6, 7],
                      [4, 5, 6, 7],
                      [5, 6, 7]]

    y_test = np.zeros((ntest,))
    for i in range(len(model_used_idx)):

        def generate_wrapper(index, names, models, params):
            if names[index] == "XGB" or names[index] == "XGB2":
                return XgbWrapper(seed=SEED, params=params[index])
            else:
                return SklearnWrapper(clf=models[index], seed=SEED, params=params[index])

        # model_used = [model_lst[idx] for idx in model_used_idx[i]]
        # arams_used = [model_params[idx] for idx in model_used_idx[i]]
        wrapper_lst = [generate_wrapper(idx, model_name_lst, model_lst, model_params)
                       for idx in range(len(model_used_idx[i]))]
        train_test_lst = [get_oof(wrapper) for wrapper in wrapper_lst]
        train_lst = [train_temp for train_temp, test_temp in train_test_lst]
        test_lst = [test_temp for train_temp, test_temp in train_test_lst]

        def scorer(data_lst1, data_lst2):
            # print data_lst1
            # print data_lst2
            return (np.abs(1 - np.exp(data_lst1.values - data_lst2.reshape((1, -1))[0]))).mean()

        for j in range(len(train_lst)):
            score = scorer(y_train, train_lst[j])
            print model_name_lst[j] + "-CV".format(score)
            if model_name_lst[j] in model_score_dic:
                model_score_dic[model_name_lst[j]][0] += score
                model_score_dic[model_name_lst[j]][1] += 1
            else:
                model_score_dic[model_name_lst[j]] = [score, 1]
        x_train = np.concatenate(train_lst, axis=1)
        x_test = np.concatenate(test_lst, axis=1)

        print("{},{}".format(x_train.shape, x_test.shape))

        random_index = random.randint(0, 5)
        print "second floor use : " + model2_name[random_index]
        model2 = generate_wrapper(random_index, model2_name, model2_lst, model2_params)
        model2.train(x_train, y_train)
        y_test += model2.predict(x_test)
    return y_test, len(model_used_idx), test_index, test_tollgate, test_direction


def predict2(offset):
    ## Load the data ##
    train = pd.read_csv("./train&test3_zjw/train_offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    test = pd.read_csv("./train&test3_zjw/test_offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    train = train.dropna()
    test_index = test.index
    test_tollgate = test.tollgate_id.values
    test_direction = test.direction.values

    ## Preprocessing ##

    y_train = np.log(train[TARGET] + 1)

    train.drop([TARGET], axis=1, inplace=True)

    all_data = pd.concat((train.copy(), test.copy()))

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    stdSc = StandardScaler()
    all_data[numeric_feats] = stdSc.fit_transform(all_data[numeric_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:

    # creating matrices for sklearn:

    x_train = np.array(all_data[:train.shape[0]])
    x_test = np.array(all_data[train.shape[0]:])

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    class Mean_Model(object):
        def __init__(self, random_state=None):
            pass

        def fit(self, X_train, y_train):
            pass

        def predict(self, X_test):
            volume_index = ["volume0", "volume1", "volume2", "volume3", "volume4", "volume5"]
            result = np.zeros(len(X_test))
            for index in volume_index:
                result += np.log1p(X_test.loc[:, index].values)
            result /= len(volume_index)
            return result

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def clf_type(self):
            return type(self.clf)

    class XgbWrapper(object):
        def __init__(self, seed=0, params=None):
            self.param = params
            self.param["seed"] = seed
            self.nrounds = params.pop("nrounds", 250)

        def train(self, x_train, y_train):
            dtrain = xgb.DMatrix(x_train, label=y_train)
            self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

        def predict(self, x):
            return self.gbdt.predict(xgb.DMatrix(x))

        def clf_type(self):
            return None

    def get_oof(clf):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            if clf.clf_type() and clf.clf_type() == Mean_Model:
                oof_train = clf.predict(train.copy())
                oof_test_skf[i, :] = clf.predict(test.copy())
            else:
                clf.train(x_tr, y_tr)

                oof_train[test_index] = clf.predict(x_te)
                oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        # print oof_train
        # print oof_test
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    model_name = ["GB", "RF", "XGB", "XGB2", "ET"]
    model_lst = [GradientBoostingRegressor, RandomForestRegressor, None, None, ExtraTreesRegressor]
    model_params = [gbdt_params, rf_params, xgb_params, xgb_params2, et_params]
    model2_name = ["GB", "ADA", "XGB", "XGB2"]
    model2_lst = [GradientBoostingRegressor, None, None]
    model2_params = [gbdt_params, xgb_params, xgb_params2]
    model_used_idx = [[0, 1, 2],
                       [0, 1, 2, 3],
                       [0, 1, 2, 3, 4],
                       [1, 2, 3, 4],
                       [2, 3, 4]]

    y_test = np.zeros((ntest,))
    for i in range(len(model_used_idx)):

        def generate_wrapper(index, names, models, params):
            if names[index] == "XGB" or names[index] == "XGB2":
                return XgbWrapper(seed=SEED, params=params[index])
            else:
                return SklearnWrapper(clf=models[index], seed=SEED, params=params[index])

        # model_used = [model_lst[idx] for idx in model_used_idx[i]]
        # arams_used = [model_params[idx] for idx in model_used_idx[i]]
        wrapper_lst = [generate_wrapper(idx, model_name, model_lst, model_params)
                       for idx in range(len(model_used_idx[i]))]
        train_test_lst = [get_oof(wrapper) for wrapper in wrapper_lst]
        train_lst = [train_temp for train_temp, test_temp in train_test_lst]
        test_lst = [test_temp for train_temp, test_temp in train_test_lst]

        def scorer(data_lst1, data_lst2):
            # print data_lst1
            # print data_lst2
            return (np.abs(1 - np.exp(data_lst1.values - data_lst2.reshape((1, -1))[0]))).mean()

        for j in range(len(train_lst)):
            score = scorer(y_train, train_lst[j])
            print model_name[j] + "-CV".format(score)
            if model_name[j] in model_score_dic:
                model_score_dic[model_name[j]][0] += score
                model_score_dic[model_name[j]][1] += 1
            else:
                model_score_dic[model_name[j]] = [score, 1]
        x_train = np.concatenate(train_lst, axis=1)
        x_test = np.concatenate(test_lst, axis=1)

        print("{},{}".format(x_train.shape, x_test.shape))

        random_index = random.randint(0, 3)
        print "second floor use : " + model2_name[random_index]
        model2 = generate_wrapper(random_index, model2_name, model2_lst, model2_params)
        model2.train(x_train, y_train)
        y_test += model2.predict(x_test)
    return y_test, len(model_used_idx), test_index, test_tollgate, test_direction


def main():
    global result_df
    for offset in range(6):

        # index, tollgate, direction在测试集中的顺序就按照predict1里读取到的文件里的顺序来
        # y_test1, len1, test_index, test_tollgate, test_direction = predict1(offset)
        y_test2, len2, test_index, test_tollgate, test_direction = predict2(offset)
        # y_test = (y_test1 + y_test2) / (len1 + len2)
        y_test = y_test2 / len2

        y_predict = pd.DataFrame()
        y_predict["volume_float"] = np.exp(y_test)
        y_predict.index = test_index
        y_predict["tollgate_id"] = test_tollgate
        y_predict["time_window"] = y_predict.index
        y_predict["time_window"] = y_predict["time_window"].apply(lambda time_basic: "[" + str(pd.Timestamp(time_basic) + DateOffset(minutes=(6 + offset) * 20)) + "," + str(
                    pd.Timestamp(time_basic) + DateOffset(minutes=((6 + offset) + 1) * 20)) + ")")
        y_predict["direction"] = test_direction
        y_predict["volume"] = y_predict["volume_float"].apply(lambda x: "%.2f" % x)
        del y_predict["volume_float"]
        result_df = result_df.append(y_predict)

    try:
        for i in range(len(model_used_name)):
            name = model_used_name[i]
            score = model_score_dic[name][0] / model_score_dic[name][1]
            print name + "-stacking : %.5f" % (score,)
        print result_df.sort_values(["tollgate_id", "direction"])
    except Exception as e:
        print e

    result_df["tollgate_id"] = result_df["tollgate_id"].replace({"1S": 1, "2S": 2, "3S": 3})
    result_df["direction"] = result_df["direction"].replace({"entry": 0, "exit": 1})
    result_df.to_csv("./train&test3_zjw/volume_predict_stacking_pure.csv", index=None, encoding="utf8")


main()
