import copy
import math

import numpy as np
from scipy.signal import wiener, savgol_filter
import utils


def process(X1):
    # print(X1.shape)
    X1_smooth = copy.deepcopy(X1)
    X1_snv = copy.deepcopy(X1)
    for i in range(len(X1)):
        X1_smooth[i, :] = wiener(X1[i, :], mysize=None, noise=None)
        # print(X1_smooth.shape)
        X1_smooth[i, :] = savgol_filter(X1_smooth[i, :], 15, 3)
        # print(X1_smooth.shape)
        X1_smooth[i, :] = utils.wavelet_denoising(X1_smooth[i, :], "sym4", 2)[1:]
        # print(X1_smooth.shape)
        X1_snv[i, :] = utils.standardize_data(X1_smooth[i, :])  ## snv
    return X1_snv, X1_smooth


def get_configs(label, data, start, end, first_wave, location):
    days = data.values[start:end, 0]
    if "daojin" in location:
        L = 711
    else:
        L = 611
    print(L)
    TUR = data.values[start:end, L + 3]
    X1 = data.values[start:end, first_wave : (L + 1)].astype("float64")
    X2 = data.values[:, (L + 4) : (2 * L + 4)].astype("float64")
    AN = data.values[start:end, 2 * L + 7]
    COD = data.values[start:end, 2 * L + 8]
    TN = data.values[start:end, 2 * L + 9]
    TP = data.values[start:end, 2 * L + 10]
    KMNO = data.values[start:end, 2 * L + 11]
    comments = data.values[start:end, -1]
    tmp = [L + 3, 2 * L + 7, 2 * L + 8, 2 * L + 9, 2 * L + 10, 2 * L + 11]
    labels = data.values[start:end, tmp]

    if label == "KMNO":  # CODMn
        target = KMNO
        ranges = [2, 4, 6, 10, 15]
        cut_bound = 4  # the bound for outliers
        lower_bound = 0.5  # the detection limit
        upper_bound = 20
        abs_error_bound = 1  # absolute error bound for smaller concentration
        mape_bound = 0.15  # mape error bound for bigger concentration
        X1, _ = process(X1)
        # X2, _ = process(X2)
        if "daojin" in location:
            upper_cap = 100
            bound1 = 80
            bound2 = 100
        else:
            upper_cap = 30
            bound1 = 20
            bound2 = 30

    elif label == "COD":
        target = COD
        ranges = [15, 15, 20, 30, 40]
        cut_bound = 15
        lower_bound = 4
        upper_bound = 50
        abs_error_bound = 4
        mape_bound = 0.15
        X1, _ = process(X1)
        # X2, _ = process(X2)
        if "daojin" in location:
            upper_cap = 600
            bound1 = 500
            bound2 = 600
        else:
            upper_cap = 150
            bound1 = 100
            bound2 = 150

    elif label == "TN":
        target = TN
        ranges = [0.2, 0.5, 1, 1.5, 2]
        cut_bound = 2
        lower_bound = 0.5
        upper_bound = 10
        abs_error_bound = 0.1
        mape_bound = 0.15
        X1, _ = process(X1)
        # X2, _ = process(X2)
        if "daojin" in location:
            upper_cap = 40
            bound1 = 20
            bound2 = 40
        else:
            upper_cap = 40
            bound1 = 20
            bound2 = 40

    elif label == "TP":
        target = TP
        ranges = [0.02, 0.1, 0.2, 0.3, 0.4]
        cut_bound = 1
        lower_bound = 0.01
        upper_bound = 1
        abs_error_bound = 0.02
        mape_bound = 0.15
        X1, _ = process(X1)
        # X2, _ = process(X2)
        if "daojin" in location:
            upper_cap = 10
            bound1 = 5
            bound2 = 10
        else:
            upper_cap = 10
            bound1 = 5
            bound2 = 10

    elif label == "AN":
        target = AN
        ranges = [0.15, 0.5, 1, 1.5, 2]
        cut_bound = 1
        lower_bound = 0.025
        upper_bound = 5
        abs_error_bound = 0.5
        mape_bound = 0.15
        X1, _ = process(X1)
        # X2, _ = process(X2)
        if "daojin" in location:
            upper_cap = 30
            bound1 = 20
            bound2 = 30
        else:
            upper_cap = 30
            bound1 = 20
            bound2 = 30

    elif label == "TUR":
        target = TUR
        ranges = [30, 50, 1000]
        cut_bound = 30
        lower_bound = 1
        upper_bound = 100
        abs_error_bound = 5
        mape_bound = 0.15
        _, X1 = process(X1)
        # _, X2 = process(X2)
        if "daojin" in location:
            upper_cap = 350
            bound1 = 300
            bound2 = 350
        else:
            upper_cap = 150
            bound1 = 100
            bound2 = 150
    if "dankeng" not in location and "guanlan" not in location:
        valid_idx = [
            i
            for i in range(len(target))
            if not math.isnan(target[i])
            and "污" not in comments[i]
            and "纯" not in comments[i]
        ]
    else:
        valid_idx = [i for i in range(len(target)) if not math.isnan(target[i])]
    target = target[valid_idx]
    X1 = X1[valid_idx, :]
    # X2 = X2[valid_idx, :]
    days = days[valid_idx]
    TUR = TUR[valid_idx]
    TN = TN[valid_idx]
    COD = COD[valid_idx]
    KMNO = KMNO[valid_idx]
    TP = TP[valid_idx]
    AN = AN[valid_idx]
    labels = labels[valid_idx, :]

    return {
        "days": days,
        "X1": X1,
        "target": target,
        "TUR": TUR,
        "TN": TN,
        "KMNO": KMNO,
        "COD": COD,
        "TP": TP,
        "AN": AN,
        "labels": labels,
        "ranges": ranges,
        "cut_bound": cut_bound,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "abs_error_bound": abs_error_bound,
        "mape_bound": mape_bound,
        "upper_cap": upper_cap,
        "bound1": bound1,
        "bound2": bound2,
    }
