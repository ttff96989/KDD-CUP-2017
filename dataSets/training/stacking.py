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

result_df = pd.DataFrame()
for tollgate_id, direction, offset in tuple_lst:

    ## Load the data ##
    train = pd.read_csv("./train&test_zjw/volume_" + direction + "_train_" + tollgate_id + "offset"+str(offset)+".csv", index_col="Unnamed: 0")
    test = pd.read_csv("./train&test_zjw/volume_" + direction + "_test_"+tollgate_id+".csv", index_col="Unnamed: 0")
    test_index = test.index

    ntrain = train.shape[0]
    ntest = test.shape[0]

    ## Preprocessing ##

    y_train = np.log(train[TARGET] + 1)


    train.drop([TARGET], axis=1, inplace=True)


    all_data = pd.concat((train.copy(),test.copy()))

    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    #filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    #creating matrices for sklearn:

    x_train = np.array(all_data[:train.shape[0]])
    x_test = np.array(all_data[train.shape[0]:])

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

    def get_oof(clf):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    et_params = {
        'n_jobs': 16,
        'n_estimators': 100,
        'max_features': 0.5,
        'max_depth': 12,
        'min_samples_leaf': 2,
    }

    rf_params = {
        'n_jobs': 16,
        'n_estimators': 100,
        'max_features': 0.2,
        'max_depth': 12,
        'min_samples_leaf': 2,
    }

    rd_params={
        'alpha': 10
    }

    ls_params={
        'alpha': 0.005
    }

    gbdt_params={

    }

    # 可以无限增加元模型，然后增加模型组合的可能性
    model_name_lst = ["RD", "ET", "RF", "GB", "LS"]
    model_lst = [Ridge, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, Lasso]
    model2_lst = [ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor]
    model_params = [rd_params, et_params, rf_params, gbdt_params, ls_params]
    model_used_idx = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4]]
    y_test = np.zeros((ntest,))
    for i in range(len(model_used_idx)):
        model_used = [model_lst[idx] for idx in model_used_idx[i]]
        params_used = [model_params[idx] for idx in model_used_idx[i]]
        wrapper_lst = [SklearnWrapper(clf=model_used[i], seed=SEED, params=params_used[i]) for i in range(len(model_used))]
        train_test_lst = [get_oof(wrapper) for wrapper in wrapper_lst]
        train_lst = [train for train, test in train_test_lst]
        test_lst = [test for train, test in train_test_lst]
        for i in range(len(train_lst)):
            print model_name_lst[i] + "-CV".format(sqrt(mean_squared_error(y_train, train_lst[i])))
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

        gbdt = model2_lst[random.randint(0, 2)]()
        gbdt.fit(x_train, y_train)
        y_test += gbdt.predict(x_test)
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

result_df["tollgate_id"] = result_df["tollgate_id"].replace({"1S": 1, "2S": 2, "3S": 3})
result_df["direction"] = result_df["direction"].replace({"entry": 0, "exit": 1})
result_df.to_csv("volume_predict_stacking.csv")
print result_df