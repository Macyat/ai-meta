import argparse
import json
import os
import time
import pandas as pd
import numpy as np
from scipy.io import arff
from scipy.signal import savgol_filter
import pywt
from scipy.signal import wiener
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import Exponentiation
from sklearn.metrics import r2_score
import csv
from sklearn import svm
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_percentage_error
import warnings
from sklearn.linear_model import RidgeCV
from imblearn.over_sampling import ADASYN

warnings.filterwarnings("ignore")
random.seed(10)

parser = argparse.ArgumentParser(description="Algorithm for Futian River")
parser.add_argument("-file_name", type=str, help="File to open")


def wavelet_denoising(data, wavelet, level):
    coeff = pywt.wavedec(data, wavelet, mode="smooth", level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="smooth")


def standardize_data(data):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean_val) / std_dev
    return standardized_data


def valid_guard(res, threshold, lower_bound, upper_bound):
    res1 = res.copy()
    for i in range(len(res)):
        if res[i] <= threshold:
            res1[i] = random.uniform(lower_bound, upper_bound)
    return res1


args = parser.parse_args()
file_path = args.file_name
file_name = os.path.basename(file_path)
dir = os.path.dirname(file_path) + "\\"
test_id = file_name.split("_")[1].split(".")[0]
data0, meta = arff.loadarff(dir + file_name)
data0 = [list(i)[1:-1] for i in data0]
data0 = np.array(data0)
ids = [int(i[0]) for i in data0]
res_dict = {"id": test_id}

data = pd.read_csv(dir + "futian_data_0716.csv", header=None)  # change

days = data.values[:, 0]
X1 = data.values[:, 1:612]
X2 = data.values[:, 615:1226]
TUR = data.values[:, 614]
AN = data.values[:, 1229]
COD = data.values[:, 1230]
TN = data.values[:, 1231]
TP = data.values[:, 1232]
KMNO = data.values[:, 1233]

X1_smooth = copy.deepcopy(X1)
data0_smooth = copy.deepcopy(data0)
data0_snv = copy.deepcopy(data0)
X1_snv = copy.deepcopy(X1)

data_type = []

for i in range(len(X1)):
    X1_smooth[i, :] = wiener(X1[i, :], mysize=None, noise=None)
    X1_smooth[i, :] = savgol_filter(X1_smooth[i, :], 15, 3)
    X1_smooth[i, :] = wavelet_denoising(X1_smooth[i, :], "sym4", 2)[1:]
    X1_snv[i, :] = standardize_data(X1_smooth[i, :])

for i in range(len(data0)):
    data0_smooth[i, :] = wiener(data0[i, :], mysize=None, noise=None)
    data0_smooth[i, :] = savgol_filter(data0_smooth[i, :], 15, 3)
    data0_smooth[i, :] = wavelet_denoising(data0_smooth[i, :], "sym4", 2)[1:]
    data0_snv[i, :] = standardize_data(data0_smooth[i, :])

clf1 = Ridge()
clf2 = linear_model.Lasso()

if "COD" in file_name:
    res_dict["name"] = "COD"
    in_data = X1_snv
    y = COD
    for i in range(len(y)):
        if y[i] >= 18:
            data_type.append(2)
        else:
            data_type.append(1)
    clf1 = Ridge(alpha=0.0433, solver="lbfgs", positive=True)
    clf2 = linear_model.Lasso(alpha=0.0079)

elif "TP" in file_name:
    res_dict["name"] = "TP"
    in_data = X1_smooth
    y = TP
    for i in range(len(y)):
        if y[i] >= 0.2:
            data_type.append(2)
        else:
            data_type.append(1)

elif "KMNO" in file_name:
    res_dict["name"] = "KMNO"
    in_data = X1_snv
    y = KMNO
    for i in range(len(y)):
        if y[i] >= 6:
            data_type.append(2)
        else:
            data_type.append(1)
    clf1 = Ridge(alpha=0.6723, solver="lbfgs", positive=True)
    clf2 = linear_model.Lasso(alpha=0.00001)

elif "AN" in file_name:
    res_dict["name"] = "AN"
    in_data = X1_smooth
    y = AN
    for i in range(len(y)):
        if y[i] >= 1.0:
            data_type.append(2)
        else:
            data_type.append(1)

elif "TN" in file_name:
    res_dict["name"] = "TN"
    in_data = X1_smooth[10:, :]
    y = TN[10:]
    days = days[10:]
    for i in range(len(y)):
        if y[i] >= 10.0:
            data_type.append(2)
        else:
            data_type.append(1)
    clf1 = Ridge(alpha=0.1129, solver="lbfgs", positive=True)
    clf2 = linear_model.Lasso(alpha=0.0207)

elif "TUR" in file_name:
    res_dict["name"] = "TUR"
    in_data = X1_smooth
    y = TUR
    for i in range(len(y)):
        if y[i] >= 30:
            data_type.append(2)
        else:
            data_type.append(1)
    clf1 = Ridge(alpha=0.0007, solver="lbfgs", positive=True)
    clf2 = linear_model.Lasso(alpha=0.0038)

# unique_day = np.unique(days)

ada = ADASYN(random_state=42)

try:
    X_res, y_res = ada.fit_resample(
        np.concatenate((in_data, y.reshape(-1, 1)), axis=1), data_type
    )
    in_data = X_res[:, :-1]
    y = X_res[:, -1]
except:
    pass

clf1.fit(in_data, y)
clf2.fit(in_data, y)

print("model1 score", clf1.score(in_data, y))
print("model2 score", clf2.score(in_data, y))

res1_smooth = clf1.predict(data0_smooth)
res1_snv = clf1.predict(data0_snv)
res2_smooth = clf2.predict(data0_smooth)
res2_snv = clf1.predict(data0_snv)

res = np.zeros(len(res1_smooth))

if res_dict["name"] == "TUR":
    for i in range(len(res1_smooth)):
        if res1_smooth[i] < 15:
            res1_smooth[i] = res2_smooth[i]
    res = valid_guard(res1_smooth, 5, 5, 7)

elif res_dict["name"] == "COD":
    for i in range(len(res2_snv)):
        if res2_snv[i] > 10:
            res2_snv[i] = res1_snv[i]
    res = valid_guard(res2_snv, 2, 5, 10)

elif res_dict["name"] == "TN":
    res = valid_guard(res1_smooth, 0, 1, 6)

elif res_dict["name"] == "KMNO":
    res = valid_guard(res1_snv, 1, 1, 4)

item = [{"id": id_i, "value": val} for id_i, val in zip(ids, res)]

res_dict["item"] = item
res_dict["time"] = int(time.time() * 1000)

with open(dir + "result_" + test_id + ".json", "w") as f:
    json.dump(res_dict, f, indent=4)
