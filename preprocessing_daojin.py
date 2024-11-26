import copy
import math
from scipy.signal import wiener, savgol_filter
import utils


def process(X1):
    X1_smooth = copy.deepcopy(X1)
    X1_snv = copy.deepcopy(X1)
    for i in range(len(X1)):
        X1_smooth[i, :] = wiener(X1[i, :], mysize=None, noise=None)
        X1_smooth[i, :] = savgol_filter(X1_smooth[i, :], 15, 3)
        X1_smooth[i, :] = utils.wavelet_denoising(X1_smooth[i, :], "sym4", 2)[1:]
        X1_snv[i, :] = utils.standardize_data(X1_smooth[i, :])  ## snv
    return X1_snv, X1_smooth


def get_configs(label, data, start, end, first_wave):
    days = data.values[start:end, 0]
    TUR = data.values[start:end, 714]
    X1 = data.values[start:end, first_wave:712].astype("float64")
    X2 = data.values[:, 715:1426].astype("float64")
    TN = data.values[start:end, 1431]
    KMNO = data.values[start:end, 1433]
    COD = data.values[start:end, 1430]
    TP = data.values[start:end, 1432]
    AN = data.values[start:end, 1429]

    if label == "KMNO":  # CODMn
        target = KMNO
        ranges = [2, 4, 6, 10, 15]
        cut_bound = 4  # the bound for outliers
        lower_bound = 0.5  # the detection limit
        upper_bound = 20
        abs_error_bound = 1  # absolute error bound for smaller concentration
        mape_bound = 0.15  # mape error bound for bigger concentration
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "COD":
        target = COD
        ranges = [15, 15, 20, 30, 40]
        cut_bound = 15
        lower_bound = 4
        upper_bound = 50
        abs_error_bound = 4
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "TN":
        target = TN
        ranges = [0.2, 0.5, 1, 1.5, 2]
        cut_bound = 2
        lower_bound = 0.5
        upper_bound = 10
        abs_error_bound = 0.1
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "TP":
        target = TP
        ranges = [0.02, 0.1, 0.2, 0.3, 0.4]
        cut_bound = 1
        lower_bound = 0.01
        upper_bound = 1
        abs_error_bound = 0.02
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "AN":
        target = AN
        ranges = [0.15, 0.5, 1, 1.5, 2]
        cut_bound = 1
        lower_bound = 0.025
        upper_bound = 5
        abs_error_bound = 0.5
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "TUR":
        target = TUR
        ranges = [30, 50, 1000]
        cut_bound = 30
        lower_bound = 1
        upper_bound = 100
        abs_error_bound = 5
        mape_bound = 0.15
        _, X1 = process(X1)
        _, X2 = process(X2)

    valid_idx = [i for i in range(len(target)) if not math.isnan(target[i])]
    target = target[valid_idx]
    X1 = X1[valid_idx, :]
    X2 = X2[valid_idx, :]
    days = days[valid_idx]
    TUR = TUR[valid_idx]
    TN = TN[valid_idx]
    COD = COD[valid_idx]
    KMNO = KMNO[valid_idx]
    TP = TP[valid_idx]
    AN = AN[valid_idx]

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
        "ranges": ranges,
        "cut_bound": cut_bound,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "abs_error_bound": abs_error_bound,
        "mape_bound": mape_bound,
        "upper_bound": ranges[-1] * 1.5,
        "bound1": ranges[-1],
        "bound2": ranges[-1] * 1.5,
    }
