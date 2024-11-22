import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pywt
import csv
from sklearn.metrics import r2_score


def select_idx(target, TUR, TUR_bound, lower_bound, upper_bound):
    """
    To select a part of samples to evaluate the models
    :param target: the target to predict
    :param TUR: the turbidity data
    :param TUR_bound: the upper bound of turbidity to be selected
    :param lower_bound: the lower bound of the target data to be selected
    :param upper_bound: the upper bound of the target data to be selected
    :return: the vector of selected index
    """
    return [
        i
        for i in range(len(TUR))
        if TUR[i] <= TUR_bound
        and target[i] >= lower_bound
        and target[i] <= upper_bound
        and not math.isnan(target[i])
    ]


def evaluate(res, target, ranges, idx, abs_error_bound, mape_bound):
    """
    To evaluate the models and calculate the rate of results that reach some standard
    :param res: the predicted values
    :param target: the target to predict
    :param ranges: the bounds to define the sample belong to which type of water(check 3838 2002:
    https://english.mee.gov.cn/standards_reports/standards/water_environment/quality_standard/200710/W020061027509896672057.pdf)
    :param idx: the index of data to be evaluated
    :param abs_error_bound: the absolute error bound for smaller concentration
    :param mape_bound: the mape bound for bigger concentration
    :return: the vector of selected index and the index of data that do not meet the standard
    """
    count = 0
    bad_idx = []
    for i in range(len(idx)):
        if target[i] <= ranges[1] and res[i] <= ranges[1]:
            count += 1
        elif target[i] < ranges[2] and abs(target[i] - res[i]) <= abs_error_bound:
            count += 1
        elif ranges[1] < target[i] <= ranges[2] and ranges[1] < res[i] <= ranges[2]:
            count += 1
        elif ranges[2] < target[i] <= ranges[3] and ranges[2] < res[i] <= ranges[3]:
            count += 1
        elif ranges[3] < target[i] <= ranges[4] and ranges[3] < res[i] <= ranges[4]:
            count += 1
        elif ranges[4] < target[i] and ranges[4] < res[i]:
            count += 1
        # 三类以上误差
        elif (
            target[i] >= ranges[2] and abs(target[i] - res[i]) / target[i] <= mape_bound
        ):
            count += 1
        else:
            bad_idx.append(i)
    return count / len(idx), bad_idx


def plot(res, target, TUR, idx, label, model_type, location):
    """
    To create some plots supporting model evaluating
    :param res: the predicted values
    :param target: the original target values to be predicted
    :param TUR: the turbidity data
    :param idx: the index of data to be evaluated
    :param label: which element
    :param model_type: which model
    :param location: Futian/LAB, FUTIAN is for samples collected from the Futian River,
    LAB is for samples created in out lab by mixing dirty water and river water
    :return: mape and r2_score of evaluated data
    """
    l1 = [res[i] for i in idx]
    l2 = [target[i] for i in idx]
    print("MAPE", mean_absolute_percentage_error(l1, l2))
    print("r2 score", r2_score(l1, l2))
    fig = plt.figure()
    plt.subplot(311)
    plt.scatter(list(range(len(l1))), [abs(i - j) / j for i, j in zip(l1, l2)])
    plt.ylabel("MAPE")
    plt.subplot(312)
    plt.scatter(list(range(len(l1))), l1, s=2, edgecolors="r")
    plt.scatter(list(range(len(l2))), l2, s=2, edgecolors="b")
    plt.legend(["PREDICT", "TRUE"])
    plt.xlabel("Sample ID")
    plt.ylabel(label)
    plt.subplot(313)
    plt.scatter(
        list(range(len([TUR[i] for i in idx]))),
        [TUR[i] for i in idx],
        s=2,
        edgecolors="y",
    )
    plt.ylabel("TUR")
    plt.savefig("figs\\" + location + "_" + label + "_" + model_type + ".png")
    return (
        float(mean_absolute_percentage_error(l1, l2)),
        r2_score(l1, l2),
        np.sqrt(mean_squared_error(l1, l2)),
    )


def wavelet_denoising(data, wavelet, level):
    """
    To implement wavelet denoising on data
    :param data: the spectrum matrix
    :param wavelet: the wavelet to be used
    :param level: a parameter for wavelet denoising
    :return: the spectrum matrix after wavelet denoising processing
    """
    coeff = pywt.wavedec(data, wavelet, mode="smooth", level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="smooth")


def standardize_data(data):
    """
    To implement snv on data
    :param data: the spectrum matrix
    :return: the spectrum matrix after snv processing
    """
    mean_val = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean_val) / std_dev
    return standardized_data


def moving_cur(lis):
    """
    To implement moving average
    :param lis: a vector of values
    :return: a vector of values after moving averaging
    """
    if len(lis) <= 10:
        return lis
    return moving_cur(lis[:-1]) + [np.mean(lis[-10:])]


def write_res(res, location, label, model_type):
    """
    To write the predicted values to a csv file
    :param res: the predicted values
    :param location: Futian/LAB, FUTIAN is for samples collected from the Futian River,
    LAB is for samples created in out lab by mixing dirty water and river water
    :param label: which element
    :param model_type: which model
    :return: none
    """
    with open(
        "results\\" + location + "_" + label + "_" + model_type + ".csv",
        "w",
        newline="",
    ) as f:
        write = csv.writer(f)
        for r in res:
            write.writerow([r])
