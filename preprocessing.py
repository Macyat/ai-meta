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
        X1_smooth[i, :] = utils.wavelet_denoising(X1_smooth[i, :], 'sym4', 2)[1:]
        X1_snv[i, :] = utils.standardize_data(X1_smooth[i, :])  ## snv
    return X1_snv, X1_smooth


def get_configs(label, data, start, end, first_wave):
    days = data.values[start:end, 0]
    TUR = data.values[start:end, 614]
    X1 = data.values[start:end, first_wave:612].astype('float64')
    X2 = data.values[:, 615:1226].astype('float64')

    if label == "KMNO":  # CODMn
        target = data.values[start:end,1233]
        ranges = [2, 4, 6, 10, 15]
        cut_bound = 4  # the bound for outliers
        lower_bound = 0.5  # the detection limit
        upper_bound = 20
        abs_error_bound = 0.3  # absolute error bound for smaller concentration
        mape_bound = 0.15  # mape error bound for bigger concentration
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "COD":
        target = data.values[start:end,1230]
        ranges = [15, 15, 20, 30, 40]
        cut_bound = 15
        lower_bound = 4
        upper_bound = 50
        abs_error_bound = 3
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "TN":
        target = data.values[start:end,1231]
        ranges = [0.2, 0.5, 1, 1.5, 2]
        cut_bound = 2
        lower_bound = 0.5
        upper_bound = 10
        abs_error_bound = 0.1
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "TP":
        target = data.values[start:end,1232]
        ranges = [0.02, 0.1, 0.2, 0.3, 0.4]
        cut_bound = 1
        lower_bound = 0.01
        upper_bound = 1
        abs_error_bound = 0.02
        mape_bound = 0.15
        X1, _ = process(X1)
        X2, _ = process(X2)

    elif label == "AN":
        target = data.values[start:end,1229]
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

    return {'days': days,
            'X1': X1,
            'target': target,
            'TUR': TUR,
            'ranges': ranges,
            'cut_bound': cut_bound,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'abs_error_bound': abs_error_bound,
            'mape_bound': mape_bound}