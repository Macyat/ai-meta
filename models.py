import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from imblearn.over_sampling import ADASYN
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import CARS
from sklearn.metrics import mean_squared_error

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
        if val[i] >=bound:
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
            res[i] = random.uniform(min(lower_bound, ranges[0]),max(lower_bound, ranges[0]))
    return res

    
def resample_meta(X_train, y_train,data_type):
    """
        To create balanced datasets between normal values and outliers
        :param X_train: the spectrum matrix for training
        :param y_train: the target to predict
        :param data_type: the group label created by function create_type
        :return: resampled training data and its target
    """
    ada = ADASYN()
    try:
        X_res, y_res = ada.fit_resample(np.concatenate((X_train,y_train.reshape(-1,1)),axis=1),
                                        data_type)
        X_train = X_res[:,:-1]
        y_train = X_res[:,-1]

        print('resampled', X_train.shape,y_train.shape)
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
    data_type = create_type(target, bound)
    unique_day = np.unique(days)
    res_train = []
    best_score1 = -100
    best_score2 = -100
    best_model = None
    for i in range(len(unique_day)):
        print('training on day', i)
        train_idx = [j for j in range(len(days)) if days[j]!=unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j]==unique_day[i]]
        X_train = in_data[train_idx,:]
        y_train = target[train_idx]
        if select_wave:
            lis, minRMSECV, rindex = CARS.CARS_Cloud(X_train, y_train, days[train_idx], 800, location, label, model_type, N=20)
            print('RMSE of CARS', minRMSECV)
            X_train = X_train[:, lis]
            X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
            pls = PLSRegression(n_components=rindex+1, scale=True, max_iter=500, tol=1e-06, copy=True)
            Xtest = in_data[test_idx, :][:, lis]
        else:
            X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
            pls = PLSRegression(n_components= 30, scale=True, max_iter=500, tol=1e-06, copy=True)
            Xtest = in_data[test_idx, :]
        pls.fit(X_train, y_train)

        res_train.extend(pls.predict(Xtest).reshape(1,-1).tolist()[0])
        if pls.score(X_train, y_train) > best_score1:
            best_score1 = pls.score(X_train, y_train)
            best_model_fit = pls
        if r2_score(pls.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(pls.predict(Xtest), target[test_idx])
            best_model_predict = pls
        print('predicted r2 score at day', i, r2_score(pls.predict(Xtest), target[test_idx]))
        print('predicted RMSE at day', i, np.sqrt(mean_squared_error(pls.predict(Xtest), target[test_idx])))
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
        train_idx = [j for j in range(len(days)) if days[j]!=unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j]==unique_day[i]]
        X_train = in_data[train_idx,:]
        y_train = target[train_idx]
        X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = Ridge(alpha=configs[label + '1'].values[0],solver='lbfgs',positive=True)
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)
            
        clf.fit(X_train, y_train)
                
        res_train.extend(clf.predict(in_data[test_idx,:]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_fit = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print('predicted r2 score at day', i, r2_score(clf.predict(Xtest), target[test_idx]))
        print('predicted RMSE at day', i, np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])))
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
        train_idx = [j for j in range(len(days)) if days[j]!=unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j]==unique_day[i]]
        X_train = in_data[train_idx,:]
        y_train = target[train_idx]
        X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])

        clf = linear_model.Lasso(alpha=configs[label + '2'].values[0])
        if boost:
            clf = AdaBoostRegressor(clf, n_estimators=300, random_state=rng)
            
        clf.fit(X_train, y_train)
                
        res_train.extend(clf.predict(in_data[test_idx,:]))
        Xtest = in_data[test_idx, :]
        if clf.score(X_train, y_train) > best_score1:
            best_score1 = clf.score(X_train, y_train)
            best_model_predict = clf
        if r2_score(clf.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(clf.predict(Xtest), target[test_idx])
            best_model_predict = clf
        print('predicted r2 score at day', i, r2_score(clf.predict(Xtest), target[test_idx]))
        print('predicted RMSE at day', i, np.sqrt(mean_squared_error(clf.predict(Xtest), target[test_idx])))
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
        train_idx = [j for j in range(len(days)) if days[j]!=unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j]==unique_day[i]]
        X_train = in_data[train_idx,:]
        y_train = target[train_idx]
        X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        
        gpr = GaussianProcessRegressor(
            kernel=kernel_,
            random_state=0,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
        )
        gpr.fit(X_train, y_train)
        r, s = gpr.predict(in_data[test_idx,:], return_std=True)
        res_train.extend(r)
        sigma.extend(s)
        Xtest = in_data[test_idx, :]
        if gpr.score(X_train, y_train) > best_score1:
            best_score1 = gpr.score(X_train, y_train)
            best_model_fit = gpr
        if r2_score(gpr.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(gpr.predict(Xtest), target[test_idx])
            best_model_predict = gpr
        print('predicted r2 score at day', i, r2_score(gpr.predict(Xtest), target[test_idx]))
        print('predicted RMSE at day', i, np.sqrt(mean_squared_error(gpr.predict(Xtest), target[test_idx])))
    return res_train, best_score1, best_score2, best_model_fit, best_model_predict, sigma


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
        train_idx = [j for j in range(len(days)) if days[j]!=unique_day[i]]
        test_idx = [j for j in range(len(days)) if days[j]==unique_day[i]]
        X_train = in_data[train_idx,:]
        y_train = target[train_idx]
        X_train, y_train = resample_meta(X_train, y_train, data_type[train_idx])
        scaler = StandardScaler() 
        scaler.fit(X_train) 
        X_train = scaler.transform(X_train) 
        X_val = scaler.transform(in_data[test_idx,:])
        model = LGBMRegressor(reg_alpha = configs[label + '1'].values[0],reg_lambda = configs[label + '2'].values[0], boosting_type =
                              'dart', metric = 'rmse',n_estimators = 300).fit(X_train, y_train)
        res_train.extend(model.predict(X_val))
        Xtest = in_data[test_idx, :]
        if r2_score(model.predict(X_val),target[test_idx]) > best_score1:
            best_score1 = r2_score(model.predict(X_val),target[test_idx])
            best_model_fit = model
        if r2_score(model.predict(Xtest), target[test_idx]) > best_score2:
            best_score2 = r2_score(model.predict(Xtest), target[test_idx])
            best_model_predict = model
        print('predicted r2 score at day', i, r2_score(model.predict(Xtest), target[test_idx]))
        print('predicted RMSE at day', i, np.sqrt(mean_squared_error(model.predict(Xtest), target[test_idx])))
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
    y0= [y[i] for i in idx]
    p0 = [p[i] for i in idx]
    sigma0 = [sigma[i] for i in idx]
    ###sort by x0 or the figure will be a mess
    tmp = list(enumerate(x0))
    l1 = [x0[i[0]] for i in sorted(tmp,key = lambda x:(x[1],x[0]))]
    l2 = [[j-k for j,k in zip(p0,[1.96 * i for i in sigma0])][i[0]] for i in sorted(tmp,key = lambda x:(x[1],x[0]))]
    l3 = [[j+k for j,k in zip(p0,[1.96 * i for i in sigma0])][i[0]] for i in sorted(tmp,key = lambda x:(x[1],x[0]))]
    plt.scatter(x0, y0, c='r', marker='.', label='Observations')
    plt.fill_between(l1, l2, l3, alpha=0.2, color='blue', label='95% Confidence Interval',interpolate=True)
    plt.title('Gaussian Process Regression')
    plt.xlabel('Input')
    plt.ylabel(label)
    plt.legend()
    plt.savefig("figs\\" + label + '_gpr.png')
    
    



































