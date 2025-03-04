import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import inv_boxcox
from scipy.stats import alpha
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.gaussian_process.kernels import (
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel,
)
from sklearn.linear_model import (
    LinearRegression,
    Lars,
    LassoLars,
    LassoLarsIC,
    OrthogonalMatchingPursuit,
    MultiTaskElasticNet,
    Ridge,
    SGDRegressor,
    ElasticNet,
    HuberRegressor,
    QuantileRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    TweedieRegressor,
    BayesianRidge,
)
from imblearn.over_sampling import ADASYN
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import CARS
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy import stats
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import boxcox

rng = np.random.RandomState(1)


def create_type(val, bound):
    """
    Give different labels to normal values or outliers
    :param val: the value to be labeled
    :param bound: the bound to define outliers
    :return: a vector of label
    """
    res_type = []
    for i in range(len(val)):
        if val[i] >= bound:
            res_type.append(2)
        else:
            res_type.append(1)
    return np.array(res_type)


def res_lower_cap(lower_bound, ranges, res):
    """
    Replace the values lower than a threshold with some bigger random values
    :param lower_bound: the threshold
    :param ranges: the bounds to define the sample belong to which type of water(check 3838 2002:
    https://english.mee.gov.cn/standards_reports/standards/water_environment/quality_standard/200710/W020061027509896672057.pdf)
    :param res: the predicted concentration vector
    :return: the processed concentration vector
    """
    for i in range(len(res)):
        if res[i] <= lower_bound:
            res[i] = random.uniform(
                min(lower_bound, ranges[0]), max(lower_bound, ranges[0])
            )
    return res


def res_upper_cap(upper_bound, bound1, bound2, res):
    """
    Replace the values bigger than a threshold with some smaller random values
    :param upper_bound: the threshold
    :param bound1: the lower bound for random values
    :param bound2: the upper bound for random values
    :param res: the predicted concentration vector
    :return: the processed concentration vector
    """
    for i in range(len(res)):
        if res[i] > upper_bound * 1.5:
            res[i] = random.uniform(bound1, bound2)
    return res


def resample_meta(X_train, y_train, data_type):
    """
    To create balanced datasets between normal values and outliers
    :param X_train: the spectrum matrix for training
    :param y_train: the target to predict
    :param data_type: the group label created by function create_type
    :return: resampled training data and its target
    """
    ada = ADASYN()
    try:
        X_res, y_res = ada.fit_resample(
            np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1), data_type
        )
        X_train = X_res[:, :-1]
        y_train = X_res[:, -1]

        print("resampled", X_train.shape, y_train.shape)
    except:
        pass
        # print('failed')
    return X_train, y_train


def pls_meta(in_data, target, bound, days, select_wave, location, label, model_type):
    """
    To train a pls model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :return: the predicted value and two models with their scores
    """

    if len(target.shape) == 1:
        data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model = None
    for i in range(len(unique_day)):
        print("training on day", i)
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        if select_wave:
            lis, minRMSECV, rindex = CARS.CARS_Cloud(
                X_train,
                y_train,
                days[train_idx],
                in_data.shape[1],
                location,
                label,
                model_type,
                N=20,
            )
            print("RMSE of CARS", minRMSECV)
            X_train = X_train[:, lis]
            # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
            pls = PLSRegression(
                n_components=rindex + 1, scale=True, max_iter=500, tol=1e-06, copy=True
            )
            Xtest = in_data[test_idx, :][:, lis]
        else:
            # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
            pls = PLSRegression(
                n_components=30, scale=True, max_iter=500, tol=1e-06, copy=True
            )
            Xtest = in_data[test_idx, :]
        pls.fit(X_train, y_train)
        mapping = {"TUR": 0, "AN": 1, "COD": 2, "TN": 3, "TP": 4, "KMNO": 5}
        if len(target.shape) == 1:
            res_train.extend(pls.predict(Xtest).reshape(1, -1).tolist()[0])
        else:
            res_train.extend(
                pls.predict(Xtest)[:, mapping[label]].reshape(1, -1).tolist()[0]
            )
        if pls.score(X_train, y_train) > best_score1:
            best_score1 = pls.score(X_train, y_train)
            best_model_fit = pls
        if r2_score(pls.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(pls.predict(Xtest), target[test_idx])
            best_model_predict = pls
        print(
            "predicted r2 score at day",
            i,
            r2_score(pls.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(pls.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def reg_meta(in_data, target, days, boost):
    """
    To train a linear regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param days: the day index vector
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]

        clf = LinearRegression()

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]

        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def lars_meta(in_data, target, days, boost):
    """
    To train a lars regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param days: the day index vector
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]

        clf = Lars()

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]

        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def lasso_lars_ic_meta(in_data, target, days, boost, type):
    """
    To train a lars regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param days: the day index vector
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]

        clf = LassoLarsIC(criterion=type)

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]

        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def lasso_lars_meta(in_data, target, days, boost, configs, label):
    """
    To train a lars regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param days: the day index vector
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]

        clf = LassoLars(alpha=configs[label + "2"].values[0])

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]

        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def OrthogonalMatchingPursuit_meta(in_data, target, days, boost):
    """
    To train a lars regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param days: the day index vector
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]

        clf = OrthogonalMatchingPursuit()

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]

        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def multi_elst_meta(in_data, days, configs, config, label, boost):
    """
    To train a MultiTaskLasso regression model
    :param in_data: the spectrum matrix for training
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None

    target = [
        [i, j]
        for i, j, k, p, q, z in zip(
            config["COD"],
            config["KMNO"],
            config["AN"],
            config["TP"],
            config["TN"],
            config["TUR"],
        )
    ]
    mapping = {"COD": 0, "KMNO": 1}
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]

        clf = MultiTaskElasticNet(
            max_iter=1000, tol=1e-3, alpha=configs[label + "1"].values[0]
        )

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(
            [float(i[mapping[label]]) for i in clf.predict(in_data[test_idx, :])]
        )

        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf

    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def ridge_meta(in_data, target, bound, days, configs, label, boost):
    """
    To train a ridge regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        parametersGrid = {"alpha": [0.1**i for i in range(1, 10)]}
        clf = Ridge()
        logo = LeaveOneGroupOut()
        tmp = [
            (train_index, test_index)
            for _, (train_index, test_index) in enumerate(
                logo.split(X_train, y_train, days[train_idx])
            )
        ]
        # logo.get_n_splits(X_train, y_train, days[train_idx])
        # logo.get_n_splits(groups=days[train_idx])
        # clf = GridSearchCV(
        #     clf, parametersGrid, scoring="neg_root_mean_squared_error", cv=tmp
        # )

        clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        # clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        # print('LEN coef', clf.best_params_)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        # if clf.score(X_train, y_train) > best_score1:
        #     best_score1 = clf.score(X_train, y_train)
        #     best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def lasso_meta(in_data, target, bound, days, configs, label, boost):
    """
    To train a lasso regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :param boost: whether to use ada boosting
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = linear_model.Lasso(alpha=0.1 * configs[label + "2"].values[0])
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_predict = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def gpr_meta(in_data, target, bound, days, kernel_):
    """
    To train a gaussian process regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param kernel_: the kernel used for gaussian process
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    sigma = []
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        gpr = GaussianProcessRegressor(
            kernel=kernel_,
            random_state=0,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
        )
        gpr.fit(X_train, y_train)
        r, s = gpr.predict(in_data[test_idx, :], return_std=True)
        res_train.extend(r)
        sigma.extend(s)
        Xtest = in_data[test_idx, :]
        if gpr.score(X_train, y_train) > best_score1:
            best_score1 = gpr.score(X_train, y_train)
            best_model_fit = gpr
        if r2_score(gpr.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(gpr.predict(Xtest), target[test_idx])
            best_model_predict = gpr
        print(
            "predicted r2 score at day",
            i,
            r2_score(gpr.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(gpr.predict(Xtest), target[test_idx])),
        )

    return (
        res_train,
        best_score1,
        best_score2,
        best_model_fit,
        best_model_predict,
        sigma,
    )


def lgbm_meta(in_data, target, bound, days, configs, label):
    """
    To train a lgbm regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = in_data[test_idx, :]
        X_val = scaler.transform(in_data[test_idx, :])
        model = LGBMRegressor(
            reg_alpha=configs[label + "1"].values[0],
            reg_lambda=configs[label + "2"].values[0],
            boosting_type="dart",
            metric="rmse",
            n_estimators=300,
        ).fit(X_train, y_train)
        res_train.extend(model.predict(X_val))
        # if model.score(model.predict(X_val), target[test_idx]) > best_score1:
        #     best_score1 = r2_score(model.predict(X_val), target[test_idx])
        #     best_model_fit = model
        # if r2_score(model.predict(X_val), target[test_idx]) > best_score2:
        #     best_score2 = r2_score(model.predict(X_val), target[test_idx])
        #     best_model_predict = model
        print(
            "predicted r2 score at day",
            i,
            r2_score(model.predict(X_val), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(model.predict(X_val), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def bayes_ridge_meta(in_data, target, bound, days, configs, label):
    """
    To train a bayesian ridge regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        # clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        # clf = linear_model.BayesianRidge(alpha_1=0.01*configs[label + "1"].values[0])
        clf = linear_model.BayesianRidge(
            max_iter=1000, tol=1e-6, alpha_init=1, lambda_init=0.001
        )

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def ARD_meta(in_data, target, bound, days, configs, label):
    """
    To train an ARD regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        clf = linear_model.ARDRegression()

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def elastic_meta(in_data, target, bound, days, configs, label):
    """
    To train a Linear regression model with combined L1 and L2 priors as regularizer.
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        days_train = days[train_idx]
        parametersGrid = {
            "alpha": [0.1**i for i in range(1, 10)],
            "l1_ratio": np.arange(0.0, 1.0, 0.1),
        }
        eNet = ElasticNet()
        logo = LeaveOneGroupOut()
        # logo.get_n_splits(X_train, y_train, days[train_idx])
        # logo.get_n_splits(groups=days[train_idx])
        tmp = [
            (train_index, test_index)
            for _, (train_index, test_index) in enumerate(
                logo.split(X_train, y_train, days[train_idx])
            )
        ]
        # print(len(train_idx))
        # print(len(days[train_idx]))
        # clf = GridSearchCV(
        #     eNet, parametersGrid, scoring="neg_root_mean_squared_error", cv=tmp
        # )

        # X_train0, y_train = resample_meta(np.concatenate((X_train,days[train_idx].reshape(-1,1)),axis=1), y_train, data_type[train_idx])
        # X_train = X_train0[:,:-1]
        # days_train = X_train0[:,-1]

        clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)

        # clf = ElasticNet(alpha=configs[label + "2"].values[0], fit_intercept=True)
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train)
        # print("LEN coef", clf.best_params_)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        # if clf.score(X_train, y_train) > best_score1:
        # if clf.best_score_ > best_score1:
        #     # best_score1 = clf.score(X_train, y_train)
        #     best_score1 = clf.best_score_
        #     best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def lasso_lars_meta(in_data, target, bound, days, configs, label):
    """
    To train a Lasso model fit with Least Angle Regression a.k.a. Lars.
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        # clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        clf = linear_model.LassoLars(
            alpha=configs[label + "2"].values[0], fit_intercept=True
        )
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def huber_meta(in_data, target, bound, days, configs, label):
    """
    To train a L2-regularized linear regression model that is robust to outliers
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        # clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        clf = HuberRegressor(alpha=configs[label + "1"].values[0], fit_intercept=True)
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def quantile_meta(in_data, target, bound, days, configs, label):
    """
    To train a Linear regression model that predicts conditional quantiles.
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        # clf = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        clf = QuantileRegressor(
            alpha=configs[label + "2"].values[0], fit_intercept=True, quantile=0.7
        )
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def ransar_meta(in_data, target, bound, days, configs, label):
    """
    To train a RANSAC (RANdom SAmple Consensus) algorithm
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        # c = Ridge(alpha=configs[label + "1"].values[0], solver="lbfgs", positive=True)
        kernel_ = (
            1.0 * RationalQuadratic(length_scale=1, alpha=1)
            + WhiteKernel(1e-1)
            + ConstantKernel(constant_value=np.mean(target))
        )
        c = GaussianProcessRegressor(
            kernel=kernel_,
            random_state=0,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
        )
        clf = RANSACRegressor(estimator=c, min_samples=0.8)
        # clf = make_pipeline(StandardScaler(),
        #                     SGDRegressor(max_iter=1000, tol=1e-3))

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def theilsen_meta(in_data, target, bound, days, configs, label):
    """
    To train a Theil-Sen Estimator
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = TheilSenRegressor()

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def gamma_meta(in_data, target, bound, days, configs, label, boost):
    """
    To train a Generalized Linear Model with a Gamma distribution
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        Xtest = in_data[test_idx, :]
        # Xtest = scaler.transform(in_data[test_idx, :])
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        parametersGrid = {
            "alpha": [0.1**i for i in range(1, 10)]
        }  # AN: 0.1, TP: 0.0002, COD: 0.000003
        gamma = linear_model.GammaRegressor()
        logo = LeaveOneGroupOut()
        # logo.get_n_splits(X_train, y_train, days[train_idx])
        # logo.get_n_splits(groups=days[train_idx])
        tmp = [
            (train_index, test_index)
            for _, (train_index, test_index) in enumerate(
                logo.split(X_train, y_train, days[train_idx])
            )
        ]
        clf = linear_model.GammaRegressor(
            alpha=configs[label + "2"].values[0], warm_start=False
        )
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)
        # clf = linear_model.GammaRegressor(alpha=0.0002, warm_start=False, max_iter=1000,tol=0.000000001)
        # clf = GridSearchCV(gamma, parametersGrid, scoring='neg_mean_absolute_percentage_error', cv=tmp)
        clf.fit(X_train, y_train)
        # clf.fit(X_train, y_train)
        # print('LEN coef', clf.best_params_)

        res_train.extend(clf.predict(Xtest))
        # Xtest = in_data[test_idx, :]
        # if clf.score(X_train, y_train) > best_score1:
        #     best_score1 = clf.score(X_train, y_train)
        #     best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


class GammaHuberModel(torch.nn.Module):
    def __init__(self, n_features, link='log'):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.randn(n_features, dtype=torch.float64))
        self.link = link  # 'log', 'inverse', 'identity'

    def forward(self, X):
        linear = X @ self.beta
        if self.link == 'log':
            return torch.exp(linear)
        elif self.link == 'inverse':
            return 1.0 / (linear + 1e-6)  # 防止除零
        elif self.link == 'identity':
            return torch.clamp(linear, min=1e-6)  # 确保输出为正


# 自定义损失函数
def gamma_huber_loss(y_pred, y_true, delta=1.0):
    residuals = y_true - y_pred
    mask = torch.abs(residuals) <= delta
    loss = torch.where(mask, 0.5 * residuals ** 2, delta * (torch.abs(residuals) - 0.5 * delta))
    return loss.mean()


def gamma_huber(in_data, target, days, configs, label):
    """
        To train a Gamma Huber Model
        :param in_data: the spectrum matrix for training
        :param target: the target to predict
        :param days: the day index vector
        :param configs: hyperparameters for ridge/lasso model
        :param label: which element
        :return: the predicted value and two models with their scores
        """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        print(type(X_train))
        print(type(y_train))
        X_test = in_data[test_idx, :]
        y_test = target[test_idx]
        X_tensor = torch.tensor(X_train.astype("float64"))
        y_tensor = torch.tensor(y_train.astype("float64"))

        X_test_tensor = torch.tensor(X_test.astype("float64"))
        y_test_tensor = torch.tensor(y_test.astype("float64"))

        print(X_tensor.shape)
        model = GammaHuberModel(n_features=X_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(100000):
            optimizer.zero_grad()
            y_pred = model(X_tensor)
            loss = gamma_huber_loss(y_pred, y_tensor, delta=1.5)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.3f}")

        # print("估计系数:", model.beta.detach().numpy())

        print(model(X_test_tensor).tolist())
        res_train.extend(model(X_test_tensor).tolist())
        # if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
        #     best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
        #     best_model_predict = clf
        # print(
        #     "predicted r2 score at day",
        #     i,
        #     r2_score(clf.predict(Xtest), target[test_idx]),
        # )
        # print(
        #     "predicted RMSE at day",
        #     i,
        #     np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        # )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict



def poisson_meta(in_data, target, bound, days, configs, label, boost):
    """
    To train a Generalized Linear Model with a Poisson distribution
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    mean = np.mean(target)
    var = np.var(target)
    # print("标签方差", var)
    # print("标签均值", mean)
    # print(f"方差/均值比: {var / mean:.2f}")
    if var > mean:
        print("target过离散，建议使用负二项回归")
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = linear_model.PoissonRegressor(alpha=configs[label + "2"].values[0])
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def tweedie_meta(in_data, target, bound, days, configs, label, boost):
    """
    To train a Generalized Linear Model with a Tweedie distribution
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        days_train = days[train_idx]
        parametersGrid = {
            "alpha": [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
            "power": [0, 1, 1.25, 1.5, 1.75, 2, 3],
        }
        tw = TweedieRegressor(power=1.5)
        logo = LeaveOneGroupOut()
        tmp = [
            (train_index, test_index)
            for _, (train_index, test_index) in enumerate(
                logo.split(X_train, y_train, days[train_idx])
            )
        ]

        # X_train0, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = linear_model.TweedieRegressor(
            power=1.5, alpha=configs[label + "1"].values[0], max_iter=100
        )

        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)
        # clf = linear_model.TweedieRegressor(power=1.5, alpha=0.1,
        #                                     max_iter=100)
        # clf.fit(X_train, y_train)

        # clf = GridSearchCV(
        #     tw, parametersGrid, scoring="neg_root_mean_squared_error", cv=tmp
        # )
        clf.fit(X_train, y_train)
        # print("LEN coef", clf.best_params_)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        # if clf.score(X_train, y_train) > best_score1:
        #     best_score1 = clf.score(X_train, y_train)
        #     best_model_fit = clf
        # if clf.best_score_ > best_score1:
        #     # best_score1 = clf.score(X_train, y_train)
        #     best_score1 = clf.best_score_
        #     best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def passive_meta(in_data, target, bound, days, configs, label):
    """
    To train a PassiveAggressive regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = linear_model.PassiveAggressiveRegressor(epsilon=0.01, n_iter_no_change=50)

        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def SDG_meta(in_data, target, bound, days, configs, label):
    """
    To train a SDG regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = target[train_idx]
        # X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = SGDRegressor(
            max_iter=1000,
            tol=1e-3,
            alpha=configs[label + "1"].values[0],
            n_iter_no_change=50,
        )
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(in_data[test_idx, :])
        clf.fit(X_train, y_train)

        res_train.extend(clf.predict(X_val))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def multi_meta(in_data, days, configs, config, label):
    """
    To train a MultiTaskLasso regression model
    :param in_data: the spectrum matrix for training
    :param target: the target to predict
    :param bound: the bound to define outliers
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    # target = [[i,j,k,p,q,z] for i,j,k,p,q,z in zip(config["COD"], config["KMNO"], config["AN"], config["TP"],config["TN"],config["TUR"])]
    # mapping = {"COD":0,"KMNO":1,"AN":2,"TP":3,"TN":4,"TUR":5}
    # target = [
    #     [i, j, k, p, q]
    #     for i, j, k, p, q in zip(
    #         config["COD"], config["KMNO"], config["AN"], config["TP"], config["TN"]
    #     )
    # ]
    # target = [
    #     [k, p]
    #     for i, j, k, p, q in zip(
    #         config["COD"], config["KMNO"], config["AN"], config["TP"], config["TN"]
    #     )
    # ]
    target = [
        [i,k,p,q, z]
        for i, j, k, p, q, z in zip(
            config["COD"],
            config["KMNO"],
            config["AN"],
            config["TP"],
            config["TN"],
            config["TUR"],
        )
    ]
    mapping = {"COD": 0, "AN": 1,"TP": 2, "TN": 3, "TUR":4}
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]

        clf = linear_model.MultiTaskLasso(
            max_iter=1000, tol=1e-3, alpha=configs[label + "2"].values[0]
        )
        # clf = linear_model.MultiTaskLasso(
        #     max_iter=1000, tol=1e-3, alpha=0.000001
        # )
        clf.fit(X_train, y_train)

        res_train.extend(
            [float(i[mapping[label]]) for i in clf.predict(in_data[test_idx, :])]
        )
        # Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        # if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
        #     best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
        #     best_model_predict = clf
        # print(
        #     "predicted r2 score at day",
        #     i,
        #     r2_score(clf.predict(Xtest), target[test_idx]),
        # )
        # print(
        #     "predicted RMSE at day",
        #     i,
        #     np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        # )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


import statsmodels.api as sm
def WLS_meta(in_data, days, configs, config, label):
    """
    To train a weighted least squares model
    :param in_data: the spectrum matrix for training
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    target = config[label]
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]

        res_ols = sm.OLS(y_train, X_train).fit()
        # print(res_ols.summary())

        # print(res_ols.resid)
        # print(type(res_ols.resid))

        mod_wls = sm.WLS(y_train, X_train, weights=1.0 / abs(res_ols.resid))
        res_wls = mod_wls.fit()
        # print(res_wls.summary())
        # res_train.extend(mod_wls.predict(np.transpose(in_data[test_idx, :])))
        res_train.extend(np.dot(in_data[test_idx, :], res_wls.params))
    #     print(np.dot(in_data[test_idx, :], res_wls.params))
    #
    # print(res_train)

    return res_train, best_score1, best_score2, best_model_fit, best_model_predict

def box_cox(in_data, days, config, label):
    """
    To train a box cox model
    :param in_data: the spectrum matrix for training
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    target = config[label]
    for i in range(len(unique_day)):
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]
        X_test = in_data[test_idx, :]
        y_test = [target[i] for i in test_idx]
        # print([i + 1e-6 for i in y_train])

        # y_train_trans, lambda_ = boxcox(np.array([np.log(i + 1e-6) for i in y_train]))
        # print('lambda', lambda_)
        pt = PowerTransformer(method='yeo-johnson',standardize=False)
        y_train_trans = pt.fit_transform(np.array(y_train).reshape(-1,1))
        print("拟合的lambda:", pt.lambdas_[0])
        model = sm.OLS(y_train_trans, sm.add_constant(X_train)).fit()
        y_pred_trans = model.predict(sm.add_constant(X_test)).reshape(-1,1)

        y_pred = [float(i) for i in pt.inverse_transform(y_pred_trans).reshape(-1,)]
        print(y_test, y_pred_trans, y_pred)
        res_train.extend(y_pred)

    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


def TL_meta(in_data, days, configs, config, label, base_model_type):
    """
    To train a weighted least squares model
    :param in_data: the spectrum matrix for training
    :param days: the day index vector
    :param configs: hyperparameters for ridge/lasso model
    :param label: which element
    :return: the predicted value and two models with their scores
    """
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    target = config[label]
    for i in range(len(unique_day)):
        with open(
            "models\\"
            + "Futian_gaolitong_base"
            + "_"
            + label
            + "_"
            + base_model_type
            + "_best_fit"
            + ".pkl",
            "rb",
        ) as f:
            clf = pickle.load(f)

        train_idx = [
            j for j in range(len(days)) if days[j] != unique_day[i] and j < 751
        ]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i] and j < 751]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]

        clf.fit(X_train, y_train)
        res_train.extend(clf.predict(in_data[test_idx, :]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print(
            "predicted r2 score at day",
            i,
            r2_score(clf.predict(Xtest), target[test_idx]),
        )
        print(
            "predicted RMSE at day",
            i,
            np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])),
        )
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict


class DynamicWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(reduction='none'), alpha=0.1):
        """
        :param alpha: 权重衰减系数，控制对高损失样本的抑制强度（越大则抑制越强）
        """
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha

    def forward(self, inputs, targets, reduction='mean'):
        # 计算每个样本的损失（shape: [batch_size]）
        losses = self.base_loss(inputs.squeeze(), targets)

        # 动态计算权重（指数衰减函数）
        weights = torch.exp(-self.alpha * losses.detach())

        # 归一化权重（保持梯度稳定性）
        weights = weights / (weights.sum() + 1e-8) * len(weights)

        # 加权损失
        weighted_loss = (weights * losses).sum()

        return weighted_loss if reduction == 'sum' else weighted_loss / len(losses)


# 定义神经网络模型（输入601维，输出1维）
class SpectralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(591, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

def dynamic_meta(in_data, days, configs, config, label):
    # 训练配置
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SpectralModel().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 使用动态加权损失（alpha=0.5）
    criterion = DynamicWeightedLoss(alpha=0.1)

    # 数据加载（假设本地实验室数据）
    # train_dataset = SyntheticDataset(num_samples=2000)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model_fit = None
    best_model_predict = None
    target = config[label]
    for i in range(len(unique_day)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpectralModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = DynamicWeightedLoss(alpha=0.5)
        train_idx = [j for j in range(len(days)) if days[j] != unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j] == unique_day[i]]
        X_train = in_data[train_idx, :]
        y_train = [target[i] for i in train_idx]
        X_test = in_data[test_idx, :]
        y_test = [target[i] for i in test_idx]



    # 训练循环
        for epoch in range(100):
            model.train()
            total_loss = 0.0

            # for batch_X, batch_y_noisy, _ in train_loader:  # 注意：训练时使用含噪声标签
            #     batch_X, batch_y_noisy = batch_X.to(device), batch_y_noisy.to(device)

            # 前向传播
            preds = model(torch.tensor(X_train.astype("float32"))).squeeze()

            # 计算动态加权损失
            loss = criterion(preds, torch.tensor(y_train))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 验证（假设有少量干净标签）
            model.eval()
            with torch.no_grad():
                # 使用合成数据中的真实标签验证
                # X_val, _, y_val = train_dataset[:100]  # 取100个样本验证
                val_preds = model(torch.tensor(X_test.astype("float32")).to(device)).squeeze()
                val_loss = nn.MSELoss()(val_preds, torch.tensor(y_test).to(device)).item()

            print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(X_train):.4f} | Val Loss: {val_loss:.4f}")

        # Xtest = in_data[test_idx, :]
            if total_loss / len(X_train) > best_score1:
                best_score1 = total_loss / len(X_train)
                best_model_fit = model
        res_train.extend(
            list(val_preds)
        )
            # if r2_score(val_preds, y_test) > best_score2:
            #     best_score2 = r2_score(val_preds, y_test)
            #     best_model_predict = model
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict




def plot_gpr(x, y, p, idx, sigma, label):
    """
    To plot gaussian process plots with 95% confidence intervals: check this example:
    https://www.geeksforgeeks.org/quick-start-to-gaussian-process-regression/?ref=oin_asr5
    :param x: a vector data at any wavelength
    :param y: the target
    :param p: the predicted value
    :param sigma: Standard deviation of predictive distribution at query points.
    :param label: which element
    :return: a figure will be saved to figs
    """
    x0 = [x[i] for i in idx]
    y0 = [y[i] for i in idx]
    p0 = [p[i] for i in idx]
    sigma0 = [sigma[i] for i in idx]
    ###sort by x0 or the figure will be a mess
    tmp = list(enumerate(x0))
    l1 = [x0[i[0]] for i in sorted(tmp, key=lambda x: (x[1], x[0]))]
    l2 = [
        [j - k for j, k in zip(p0, [1.96 * i for i in sigma0])][i[0]]
        for i in sorted(tmp, key=lambda x: (x[1], x[0]))
    ]
    l3 = [
        [j + k for j, k in zip(p0, [1.96 * i for i in sigma0])][i[0]]
        for i in sorted(tmp, key=lambda x: (x[1], x[0]))
    ]
    plt.scatter(x0, y0, c="r", marker=".", label="Observations")
    plt.fill_between(
        l1,
        l2,
        l3,
        alpha=0.2,
        color="blue",
        label="95% Confidence Interval",
        interpolate=True,
    )
    plt.title("Gaussian Process Regression")
    plt.xlabel("Input")
    plt.ylabel(label)
    plt.legend()
    plt.savefig("figs\\" + label + "_gpr.png")
