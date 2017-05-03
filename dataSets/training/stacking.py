#coding=utf-8
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt
from pandas.tseries.offsets import *
import random
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

'''
待优化点：
1. 在stack里加一个均值模型
2. 在stack里加Adaboost套liner模型，Adaboost套DecisionTree模型
'''

TARGET = 'y'
NFOLDS = 5
SEED = 0
NROWS = None
SUBMISSION_FILE = '../input/sample_submission.csv'

id_direction_lst = [("1S", "entry"), ("1S", "exit"), ("2S", "entry"), ("3S", "entry"), ("3S", "exit")]
tuple_lst = []
for id, direction in id_direction_lst:
    for i in range(6):
        tuple_lst.append((id, direction, i))

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
    'max_depth': 3, 'min_samples_leaf': 1,
    'learning_rate': 0.1, 'loss': 'lad', 'n_estimators': 3000, 'max_features': 1.0
    }

ada_param = {
    'base_estimator': DecisionTreeRegressor(max_depth=4), 'n_estimators': 300
}

mean_param = {

}

result_df = pd.DataFrame()
model_score_dic = {}

for tollgate_id, direction, offset in tuple_lst:

    ## Load the data ##
    train = pd.read_csv("./train&test_zjw/volume_" + direction + "_train_" + tollgate_id + "offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    test = pd.read_csv("./train&test_zjw/volume_" + direction + "_test_" + tollgate_id + "offset" + str(offset) + ".csv", index_col="Unnamed: 0")
    train = train.dropna()
    test_index = test.index

    ## Preprocessing ##

    y_train = np.log(train[TARGET] + 1)

    train.drop([TARGET], axis=1, inplace=True)

    all_data = pd.concat((train.copy(), test.copy()))

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    stdSc = StandardScaler()
    all_data[numeric_feats] = stdSc.fit_transform(all_data[numeric_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:
    # all_data = all_data.fillna(all_data.mean())

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
            volume_index = [60, 61, 62, 63, 64, 65]
            result = np.zeros(len(X_test))
            for index in volume_index:
                result += X_test.iloc[:, index].values
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

    def get_oof(clf):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            if clf.clf_type() == Mean_Model:
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
    model_name_lst = ["MEAN", "RD", "ET", "GB", "LS", "RF", "ADA"]
    model_lst = [Mean_Model, Ridge, ExtraTreesRegressor, GradientBoostingRegressor, Lasso,
                 RandomForestRegressor, AdaBoostRegressor]
    model_params = [mean_param, rd_params, et_params, gbdt_params, ls_params, rf_params, ada_param]
    model2_name = ["ET", "RF", "GB", "ADA"]
    model2_lst = [ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor]
    model2_params = [et_params, rf_params, gbdt_params, ada_param]
    model_used_idx = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
                      [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [3, 4, 5, 6], [4, 5, 6]]

    y_test = np.zeros((ntest,))
    for i in range(len(model_used_idx)):
        model_used = [model_lst[idx] for idx in model_used_idx[i]]
        params_used = [model_params[idx] for idx in model_used_idx[i]]
        wrapper_lst = [SklearnWrapper(clf=model_used[i], seed=SEED, params=params_used[i])
                       for i in range(len(model_used))]
        train_test_lst = [get_oof(wrapper) for wrapper in wrapper_lst]
        train_lst = [train_temp for train_temp, test_temp in train_test_lst]
        test_lst = [test_temp for train_temp, test_temp in train_test_lst]


        def scorer(data_lst1, data_lst2):
            # print data_lst1
            # print data_lst2
            return (np.abs(1 - np.exp(data_lst1.values - data_lst2.reshape((1, -1))[0]))).sum()


        for j in range(len(train_lst)):
            score = scorer(y_train, train_lst[j])
            print model_name_lst[j] + "-CV".format(score)
            if model_name_lst[j] in model_score_dic:
                model_score_dic[model_name_lst[j]] += score
            else:
                model_score_dic[model_name_lst[j]] = score
        x_train = np.concatenate(train_lst, axis=1)
        x_test = np.concatenate(test_lst, axis=1)

        # et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
        # rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
        # rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
        # ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
        # gb = SklearnWrapper(clf=GradientBoostingRegressor, seed=SEED, params=gbdt_params)
        #
        # et_oof_train, et_oof_test = get_oof(et)
        # rf_oof_train, rf_oof_test = get_oof(rf)
        # rd_oof_train, rd_oof_test = get_oof(rd)
        # ls_oof_train, ls_oof_test = get_oof(ls)
        # gb_oof_train, gb_oof_test = get_oof(gb)
        #
        # print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
        # print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
        # print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
        # print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
        # print("GB-CV: {}".format(sqrt(mean_squared_error(y_train, gb_oof_train))))
        #
        # x_train = np.concatenate((et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train, gb_oof_train), axis=1)
        # x_test = np.concatenate((et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test, gb_oof_test), axis=1)

        print("{},{}".format(x_train.shape, x_test.shape))

        random_index = random.randint(0, 3)
        print "second floor use : " + model2_name[random_index]
        model2_param = model2_params[random_index]
        model2 = model2_lst[random_index](**model2_param)
        model2.fit(x_train, y_train)
        y_test += model2.predict(x_test)
    y_test /= len(model_used_idx)

    y_predict = pd.DataFrame()
    y_predict["volume_float"] = np.exp(y_test)
    y_predict.index = test_index
    y_predict["tollgate_id"] = tollgate_id
    y_predict["time_window"] = y_predict.index
    y_predict["time_window"] = y_predict["time_window"].apply(lambda time_basic: "[" + str(pd.Timestamp(time_basic) + DateOffset(minutes=(6 + offset) * 20)) + "," + str(
                pd.Timestamp(time_basic) + DateOffset(minutes=((6 + offset) + 1) * 20)) + ")")
    y_predict["direction"] = direction
    y_predict["volume"] = y_predict["volume_float"].apply(lambda x: "%.2f" % x)
    del y_predict["volume_float"]
    result_df = result_df.append(y_predict)

for i in range(len(model_name_lst)):
    name = model_name_lst[i]
    model_score_dic[name] /= 30 * (5 - abs(i - 2))


result_df["tollgate_id"] = result_df["tollgate_id"].replace({"1S": 1, "2S": 2, "3S": 3})
result_df["direction"] = result_df["direction"].replace({"entry": 0, "exit": 1})
result_df.to_csv("./train&test_zjw/volume_predict_stacking.csv", index=None, encoding="utf8")
print result_df
