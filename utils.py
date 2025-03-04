import math
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pywt
import csv
from sklearn.metrics import r2_score
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats


def select_idx(
    target,
    y_pred,
    TUR,
    TUR_bound,
    lower_bound,
    upper_bound,
    evaluate_start,
    evaluate_end,
    comments,
    man_made_evaluate,
):
    """
    To select a part of samples to evaluate the models
    :param target: the target to predict
    :param TUR: the turbidity data
    :param TUR_bound: the upper bound of turbidity to be selected
    :param lower_bound: the lower bound of the target data to be selected
    :param upper_bound: the upper bound of the target data to be selected
    :return: the vector of selected index
    """
    if not man_made_evaluate:
        return [
            i
            for i in range(len(TUR))
            if TUR[i] <= TUR_bound
            and lower_bound <= target[i] <= upper_bound
            and not math.isnan(target[i])
            and not math.isnan(y_pred[i])
            and evaluate_start <= i <= evaluate_end
            and "污" not in comments[i]
            and "纯" not in comments[i]
        ]
    else:
        return [
            i
            for i in range(len(TUR))
            if TUR[i] <= TUR_bound
            and lower_bound <= target[i] <= upper_bound
            and not math.isnan(target[i])
            and not math.isnan(y_pred)
            and evaluate_start <= i <= evaluate_end
        ]


def evaluate(
    X,
    res,
    target,
    ranges,
    idx,
    abs_error_bound,
    mape_bound,
    location,
    model_type,
    label,
):
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
    bad_grouped = 0
    for i in range(len(idx)):
        # if target[i] <= ranges[0] and res[i] > ranges[0]:
        #     bad_grouped += 1
        if target[i] <= ranges[1] and res[i] > ranges[1]:
            bad_grouped += 1
        elif ranges[1] < target[i] <= ranges[2] and res[i] > ranges[2]:
            bad_grouped += 1
        elif ranges[2] < target[i] <= ranges[3] and res[i] > ranges[3]:
            bad_grouped += 1
        elif (
            len(ranges) >= 5
            and ranges[3] < target[i] <= ranges[4]
            and res[i] > ranges[4]
        ):
            bad_grouped += 1

    # ref. https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202405/W020240511533747829144.pdf
    if label != "TUR":

        for i in range(len(idx)):
            if target[i] <= ranges[1] and res[i] <= ranges[1]:
                count += 1

            elif target[i] <= ranges[1] and abs(target[i] - res[i]) / target[i] <= 0.4:
                count += 1

            elif (
                ranges[1] < target[i] <= ranges[3]
                and abs(target[i] - res[i]) / target[i] <= 0.3
            ):
                count += 1

            elif target[i] > ranges[3] and abs(target[i] - res[i]) / target[i] <= 0.2:
                count += 1

            else:
                bad_idx.append(i)

    else:
        for i in range(len(idx)):
            if target[i] <= ranges[0] or target[i] >= ranges[2]:
                count += 1
            elif (
                ranges[0] < target[i] <= ranges[1]
                and abs(target[i] - res[i]) / target[i] <= 0.3
            ):
                count += 1
            elif (
                ranges[1] < target[i] < ranges[2]
                and abs(target[i] - res[i]) / target[i] <= 0.2
            ):
                count += 1
            else:
                bad_idx.append(i)

        # elif ranges[2] < target[i] <= ranges[3] and ranges[2] < res[i] <= ranges[3]:
        #     count += 1

        # elif (
        #     len(ranges) >= 5
        #     and ranges[3] < target[i] <= ranges[4]
        #     and ranges[3] < res[i] <= ranges[4]
        # ):
        #     count += 1
        #
        # elif len(ranges) >= 5 and ranges[4] < target[i] and ranges[4] < res[i]:
        #     count += 1
        #
        # # elif target[i] < ranges[2] and abs(target[i] - res[i]) <= abs_error_bound:
        # elif abs(target[i] - res[i]) <= abs_error_bound:
        #     count += 1
        # # 三类以上误差
        # # elif (
        # #     target[i] >= ranges[2] and abs(target[i] - res[i]) / target[i] <= mape_bound
        # # ):
        # elif abs(target[i] - res[i]) / target[i] <= mape_bound:
        #     count += 1
        # else:
        #     bad_idx.append(i)
    resid = [i - j for i, j in zip(target, res)]
    fig = plt.figure()
    # print(np.abs(residuals))
    plt.scatter(target, resid, c="blue", edgecolors="black")
    plt.xlabel("True values")
    plt.ylabel("Residuals")
    plt.title("Scatter plot of residuals")
    plt.savefig(
        "figs\\"
        + location
        + "\\"
        + label
        + "_"
        + model_type
        + "_residuals_vs_True_all.png"
    )
    fig = plt.figure()
    # print(np.abs(residuals))
    plt.scatter(res, resid, c="blue", edgecolors="black")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Scatter plot of residuals")
    plt.savefig(
        "figs\\"
        + location
        + "\\"
        + label
        + "_"
        + model_type
        + "_residuals_vs_Predict_all.png"
    )
    fig = plt.figure()
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Model1 Residuals Q-Q Plot")
    plt.savefig("figs\\" + location + "\\" + label + "_" + model_type + "_QQ.png")
    plt.figure()
    plt.hist(resid)
    plt.title("hist plot of resid")
    plt.savefig("figs\\" + location + "\\" + label + "_" + model_type + "_hist.png")
    try:
        plt.figure()
        plt.hist(np.log(resid))
        plt.title("hist plot of log resid")
        plt.savefig(
            "figs\\" + location + "\\" + label + "_" + model_type + "_hist_log.png"
        )
    except:
        if os.path.exists(
            "figs\\" + location + "\\" + label + "_" + model_type + "_hist_log.png"
        ):
            os.remove(
                "figs\\" + location + "\\" + label + "_" + model_type + "_hist_log.png"
            )
    # Assumption of Independent Errors
    durbin_watson_value = durbin_watson(resid)
    print("durbin_watson_value", durbin_watson_value)
    X = sm.add_constant(X)
    model = sm.OLS(target, X)
    # X = sm.add_constant(X)
    bp_test = het_breuschpagan(resid, model.exog)
    labels = [
        "LM - Statistic",
        "LM - Test p - value",
        "F - Statistic",
        "F - Test p - value",
    ]
    print(dict(zip(labels, bp_test)))
    name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
    test = jarque_bera(resid)
    print(lzip(name, test))
    skew = test[2]
    Kurtosis = test[3]
    return (
        count / len(idx),
        bad_idx,
        bad_grouped / len(idx),
        durbin_watson_value,
        bp_test[1],
        bp_test[3],
        Kurtosis,
        skew,
    )


def plot(
    res, target, TUR, compared_data, idx, label, compared_label, model_type, location
):
    """
    To create some plots supporting model evaluating
    :param res: the predicted values
    :param target: the original target values to be predicted
    :param TUR: the turbidity data
    :param TN: the TN data
    :param idx: the index of data to be evaluated
    :param label: which element
    :param model_type: which model
    :param location: Futian/LAB, FUTIAN is for samples collected from the Futian River,
    LAB is for samples created in out lab by mixing dirty water and river water
    :return: mape and r2_score of evaluated data
    """
    print(len(idx))
    print(len(res))
    l1 = [res[i] for i in idx]
    l2 = [target[i] for i in idx]
    l3 = [compared_data[i] for i in idx]
    l4 = [TUR[i] for i in idx]
    print("IDX LEN", len(idx))
    print("MAPE", mean_absolute_percentage_error(l1, l2))
    print("r2 score", r2_score(l1, l2))
    fig = plt.figure()

    plt.subplot(411)
    plt.scatter(list(range(len(l1))), [abs(i - j) / j for i, j in zip(l1, l2)])
    plt.ylabel("MAPE")
    plt.xticks([])

    plt.subplot(412)
    tmp = [(i, j, k, z) for i, j, k, z in zip(l1, l2, l3, l4)]
    tmp.sort(key=lambda x: x[1])
    plt.scatter(list(range(len(l1))), [i[0] for i in tmp], s=2, edgecolors="r")
    plt.scatter(list(range(len(l2))), [i[1] for i in tmp], s=2, edgecolors="b")
    # plt.scatter(list(range(len(l1))), l1, s=2, edgecolors="r")
    # plt.scatter(list(range(len(l2))), l2, s=2, edgecolors="b")
    plt.legend(["PREDICT", "TRUE"])
    plt.ylabel(label)
    plt.xticks([])

    plt.subplot(413)
    plt.scatter(
        list(range(len(idx))),
        [i[2] for i in tmp],
        s=2,
        edgecolors="y",
    )
    plt.ylabel(compared_label)
    plt.xticks([])

    plt.subplot(414)
    plt.scatter(
        list(range(len(idx))),
        [i[3] for i in tmp],
        s=2,
        edgecolors="y",
    )
    plt.ylabel("TUR")

    if not os.path.exists("figs\\" + location):
        os.makedirs("figs\\" + location)

    plt.savefig("figs\\" + location + "\\" + label + "_" + model_type + ".png")
    fig = plt.figure()
    residuals = [j - i for i, j in zip(l1, l2)]
    # print(np.abs(residuals))
    plt.scatter(l2, residuals, c="blue", edgecolors="black")
    plt.xlabel("True values")
    plt.ylabel("Residuals")
    plt.title("Scatter plot of residuals")
    plt.savefig(
        "figs\\" + location + "\\" + label + "_" + model_type + "_residuals_vs_True.png"
    )
    fig = plt.figure()
    # print(np.abs(residuals))
    plt.scatter(l1, residuals, c="blue", edgecolors="black")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Scatter plot of residuals")
    plt.savefig(
        "figs\\"
        + location
        + "\\"
        + label
        + "_"
        + model_type
        + "_residuals_vs_Predict.png"
    )

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
