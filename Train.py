import csv
from smtplib import LMTP_PORT

import pandas as pd
import pickle
import numpy as np
import random
import warnings
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import (
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel,
)
import models
import utils
import os
import argparse
import preprocessing

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Entry for training")
# parser.add_argument("-re_train", type=str,help="Retrain or not")
parser.add_argument("--label", "-l", type=str, help="element to predict")
parser.add_argument(
    "--compared_label", "-c", type=str, help="another element to be compared in plots"
)
parser.add_argument("--start", "-s", type=int, help="the first row to train")
parser.add_argument("--end", "-e", type=int, help="the last row to train")
parser.add_argument(
    "--evaluate_start", "-es", type=int, help="the first row to be evaluated"
)
parser.add_argument(
    "--evaluate_end", "-ee", type=int, help="the last row to be evaluated"
)
parser.add_argument(
    "--man_made_evaluate", "-mme", type=int, help="whether to evaluate man-made samples"
)
parser.add_argument(
    "--first_wave",
    "-f",
    type=int,
    help="the starting wavelength to be selected for training",
)
parser.add_argument("--model_type", "-m", type=str, help="the model selected")
parser.add_argument(
    "--cars_iterations", "-ca", type=int, help="the times for running cars"
)
parser.add_argument(
    "--location", "-lo", type=str, help="where the samples are collected"
)
parser.add_argument("--parent_folder", "-p", type=str, help="where the data locates")
parser.add_argument("--base_model_type", "-b", type=str, help="pretrained model type", default="pls")
# parser.add_argument("--filename", type=str, help="name of the data table")


random.seed(10)
rng = np.random.RandomState(1)

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("results"):
    os.makedirs("results")

if not os.path.exists("figs"):
    os.makedirs("figs")

if not os.path.exists("metrics"):
    os.makedirs("metrics")

if not os.path.exists("cars"):
    os.makedirs("cars")

### read the arguments from command line
args = parser.parse_args()
label = args.label
compared_label = args.compared_label
start = args.start
end = args.end
evaluate_start = args.evaluate_start
evaluate_end = args.evaluate_end
first_wave = args.first_wave
model_type = args.model_type
location = args.location
cars_iterations = args.cars_iterations
parent_folder = args.parent_folder
man_made_evaluate = args.man_made_evaluate
base_model_type = args.base_model_type

if "gaolitong" in location:
    if "select" in location:
        sub_dir = "gaolitong\\same_as_daojin"
    else:
        sub_dir = "gaolitong"
    # filename = os.path.join(parent_folder, sub_dir, "merge_data_gaolitong.csv")
    # filename = os.path.join(parent_folder, sub_dir, "tmp_1226.csv")
    if "all" not in location:
        configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
        filename = os.path.join(parent_folder, sub_dir, "merge_data_gaolitong.csv")
        # filename = os.path.join(parent_folder, sub_dir, "tmp_1226.csv")
    elif "base" in location:
        configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
        filename = os.path.join(parent_folder, sub_dir, "merge_data_gaolitong.csv")
    else:
        configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong_all.csv"))
        filename = os.path.join(parent_folder, sub_dir, "merge_data_gaolitong.csv")

elif "daojin" in location:
    if "select" in location:
        sub_dir = "daojin\\same_as_gaolitong"
    else:
        sub_dir = "daojin"
    filename = os.path.join(parent_folder, sub_dir, "merge_data_daojin.csv")
    configs = pd.read_csv(os.path.join(parent_folder, "configs_daojin.csv"))
elif "kukeng_common" in location:
    sub_dir = "gaolitong"
    configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
    filename = os.path.join(parent_folder, sub_dir, "merge_data_kukeng_common_avg.csv")
elif "kukeng_common" in location:
    sub_dir = "gaolitong"
    configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
    filename = os.path.join(parent_folder, sub_dir, "merge_data_kukeng_common.csv")
elif "kukeng" in location:
    sub_dir = "gaolitong"
    configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
    filename = os.path.join(parent_folder, sub_dir, "merge_data_kukeng.csv")
elif "dankeng" in location:
    sub_dir = "gaolitong"
    configs = pd.read_csv(os.path.join(parent_folder, "configs_gaolitong.csv"))
    filename = os.path.join(parent_folder, sub_dir, "merge_data_dankeng.csv")


data = pd.read_csv(filename, encoding="gbk")


if end == -1:
    end = len(data)
    evaluate_end = len(data)


### only evaluate results where the turbidity is lower the TUR_bound
if label == "TUR":
    TUR_bound = 200
else:
    TUR_bound = 20
print(evaluate_start, evaluate_end)
config = preprocessing.get_configs(label, data, start, end, first_wave, location)
print('LLLL',len(config["target"]))
# print(config["target"])
print(len([i for i in range(len(config["target"])) if config["target"][i] >= 0.5 and config["target"][i] <= 10 and config["TUR"][i] <= 20]))
print(config["lower_bound"],config["upper_bound"])
evaluate_idx = utils.select_idx(
    config["target"],
    config["TUR"],
    TUR_bound,
    config["lower_bound"],
    config["upper_bound"],
    evaluate_start,
    evaluate_end,
    config["comments"],
    man_made_evaluate,
)
print(len(evaluate_idx))

X1 = config["X1"]

print(
    "evaluation data size",
    config["lower_bound"],
    config["upper_bound"],
    len(evaluate_idx),
)
print(
    f"Train data shape:{X1.shape}, Target shape:{end - start}, days shape:{config['days'].shape}, "
    f"unique days shape:{np.unique(config['days']).shape}"
)


### default values for CARS
min_minRMSECV = 100
lis_best = list(range(611))
rindex_best = 30


###model training###
if model_type == "pls":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.pls_meta(
        X1,
        config["target"],
        config["cut_bound"],
        config["days"],
        False,
        location,
        label,
        model_type,
    )
if model_type == "pls_multi":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.pls_meta(
        X1,
        config["labels"],
        config["cut_bound"],
        config["days"],
        False,
        location,
        label,
        model_type,
    )
if model_type == "pls_multi_cars":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.pls_meta(
        X1,
        config["labels"],
        config["cut_bound"],
        config["days"],
        True,
        location,
        label,
        model_type,
    )
if model_type == "pls_cars":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.pls_meta(
        X1,
        config["target"],
        config["cut_bound"],
        config["days"],
        True,
        location,
        label,
        model_type,
    )
elif model_type == "ridge":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.ridge_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label, False
    )
elif model_type == "ada_ridge":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.ridge_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label, True
    )
elif model_type == "lasso":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label, False
    )
elif model_type == "ada_lasso":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label, True
    )
elif model_type == "gpr":
    kernel_ = (
        1.0 * RationalQuadratic(length_scale=1, alpha=1)
        + WhiteKernel(1e-1)
        + ConstantKernel(constant_value=np.mean(config["target"]))
    )
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
        sigma,
    ) = models.gpr_meta(
        X1, config["target"], config["cut_bound"], config["days"], kernel_
    )
elif model_type == "gpr_pca":
    kernel_ = (
        1.0 * RationalQuadratic(length_scale=1, alpha=1)
        + WhiteKernel(1e-1)
        + ConstantKernel(constant_value=np.mean(config["target"]))
    )
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
        sigma,
    ) = models.gpr_meta(
        PCA(n_components=10).fit_transform(X1),
        config["target"],
        config["cut_bound"],
        config["days"],
        kernel_,
    )
elif model_type == "lgbm":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lgbm_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "bayes_ridge":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.bayes_ridge_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "ARD":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.ARD_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "lasso_lars":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_lars_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "elst":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.elastic_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "huber":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.huber_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "quantile":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.quantile_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "ransar":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.ransar_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "theilsen":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.theilsen_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "gamma":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.gamma_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "poisson":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.poisson_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "tweedie":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.tweedie_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "passive":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.passive_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "SDG":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.SDG_meta(
        X1, config["target"], config["cut_bound"], config["days"], configs, label
    )
elif model_type == "multi":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.multi_meta(X1, config["days"], configs, config, label)
elif model_type == "WLS":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.WLS_meta(X1, config["days"], configs, config, label)
elif model_type == "Basic":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.reg_meta(X1, config[label], config["days"], False)
elif model_type == "ada_basic":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.reg_meta(X1, config[label], config["days"], True)
elif model_type == "lars":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lars_meta(X1, config[label], config["days"], True)
elif model_type == "lassolars_bic":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_lars_ic_meta(X1, config[label], config["days"], False, "bic")
elif model_type == "lassolars_aic":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_lars_ic_meta(X1, config[label], config["days"], False, "aic")
elif model_type == "lasso_lars":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.lasso_lars_meta(X1, config[label], config["days"], False, configs, label)
elif model_type == "orth":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.OrthogonalMatchingPursuit_meta(X1, config[label], config["days"], False)
elif model_type == "multi_elst":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.multi_elst_meta(X1, config["days"], configs, config, label, False)
elif model_type == "TL":
    (
        res_train,
        best_score_fit,
        best_score_predict,
        best_fit_model,
        best_predict_model,
    ) = models.TL_meta(X1, config["days"], configs, config, label, base_model_type)



### processing the outcomes
res_train_cap = models.res_lower_cap(
    config["lower_bound"], config["ranges"], res_train
)  ### remove invalid values
res_train_cap = models.res_upper_cap(
    config["upper_cap"], config["bound1"], config["bound2"], res_train_cap
)  ### remove invalid values
print(min(res_train_cap))
utils.write_res(res_train_cap, location, label, model_type)
print('plot location',location)
mape, r2_score, rmse = utils.plot(
    res_train_cap,
    config["target"],
    config["TUR"],
    config[compared_label],
    evaluate_idx,
    label,
    compared_label,
    model_type,
    location,
)
adj_r2_score = 1 - (1 - r2_score) * (len(res_train) - 1) / (
    len(res_train) - X1.shape[1] - 1
)
print('len res train cap', len(res_train_cap))
(
    good_predict_percentage,
    bad_idx,
    bad_grouped_ratio,
    durbin_watson_value,
    LM_p,
    F_p,
    Kurtosis,
    skew,
) = utils.evaluate(
    X1,
    res_train_cap,
    config["target"],
    config["ranges"],
    evaluate_idx,
    config["abs_error_bound"],
    config["mape_bound"],
    location,
    model_type,
    label,
)
print("rate of reaching the standard", good_predict_percentage)
print("bad_grouped_ratio", bad_grouped_ratio)
print("adjusted r2 score", adj_r2_score)


### update model performances
if os.path.exists("metrics\\" + location + "_" + label + ".csv"):
    with open("metrics\\" + location + "_" + label + ".csv", "r", newline="") as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_STRINGS)
        metrics_data = [row for row in reader]
    flag = False
    for row in metrics_data:
        if row["model_type"] == model_type:
            flag = True
            if mape < float(row["mape"]):
                row["mape"] = mape
                row["r2_score"] = r2_score
                row["rmse"] = rmse
                row["rate of reaching the standard"] = good_predict_percentage
                row["bad grouped ratio"] = bad_grouped_ratio
                row["durbin_watson_value"] = durbin_watson_value
                row["Kurtosis"] = Kurtosis
                row["skew"] = skew
                row["LM_p"] = LM_p
                row["F_p"] = F_p
    if not flag:
        metrics_data.append(
            {
                "model_type": model_type,
                "mape": mape,
                "r2_score": r2_score,
                "rmse": rmse,
                "rate of reaching the standard": good_predict_percentage,
                "bad grouped ratio": bad_grouped_ratio,
                "durbin_watson_value": durbin_watson_value,
                "Kurtosis": Kurtosis,
                "skew": skew,
                "LM_p": LM_p,
                "F_p": F_p,
            }
        )

else:
    metrics_data = []
    metrics_data.append(
        {
            "model_type": model_type,
            "mape": mape,
            "r2_score": r2_score,
            "rmse": rmse,
            "rate of reaching the standard": good_predict_percentage,
            "bad grouped ratio": bad_grouped_ratio,
            "durbin_watson_value": durbin_watson_value,
            "Kurtosis": Kurtosis,
            "skew": skew,
            "LM_p": LM_p,
            "F_p": F_p,
        }
    )

for i in range(len(metrics_data)):
    metrics_data[i]["mape"] = float(metrics_data[i]["mape"])
    metrics_data[i]["r2_score"] = float(metrics_data[i]["r2_score"])
    metrics_data[i]["rmse"] = float(metrics_data[i]["rmse"])
    metrics_data[i]["rate of reaching the standard"] = float(
        metrics_data[i]["rate of reaching the standard"]
    )
    metrics_data[i]["bad grouped ratio"] = float(metrics_data[i]["bad grouped ratio"])
    metrics_data[i]["durbin_watson_value"] = float(
        metrics_data[i]["durbin_watson_value"]
    )
    metrics_data[i]["Kurtosis"] = float(metrics_data[i]["Kurtosis"])
    metrics_data[i]["skew"] = float(metrics_data[i]["skew"])
    metrics_data[i]["LM_p"] = float(metrics_data[i]["LM_p"])
    metrics_data[i]["F_p"] = float(metrics_data[i]["F_p"])

with open("metrics\\" + location + "_" + label + ".csv", "w", newline="") as f:
    fieldnames = [
        "model_type",
        "mape",
        "r2_score",
        "rmse",
        "rate of reaching the standard",
        "bad grouped ratio",
        "durbin_watson_value",
        "LM_p",  # LM test p value for Breusch-Pagan test (for Heteroskedasticity)
        "F_p",  # F test p value for Breusch-Pagan test (for Heteroskedasticity)
        "Kurtosis",
        "skew",
        "rank",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    l0 = [i["mape"] for i in metrics_data.copy()]
    l0_sorted = sorted(l0)
    rank1 = [l0_sorted.index(j) for j in l0]

    rank2 = np.argsort([i["r2_score"] for i in metrics_data.copy()])[::-1]

    l1 = [i["rmse"] for i in metrics_data.copy()]
    l1_sorted = sorted(l1)
    rank3 = [l1_sorted.index(j) for j in l1]

    l2 = [i["rate of reaching the standard"] for i in metrics_data.copy()]
    l2_sorted = sorted(l2, reverse=True)
    rank4 = [l2_sorted.index(j) for j in l2]

    l3 = [i["bad grouped ratio"] for i in metrics_data.copy()]
    l3_sorted = sorted(l3)
    rank5 = [l3_sorted.index(j) for j in l3]

    l4 = [np.abs(i["durbin_watson_value"] - 2) for i in metrics_data.copy()]
    l4_sorted = sorted(l4)
    rank6 = [l4_sorted.index(j) for j in l4]

    tmp = [
        [data, float(i) + float(k) + float(p) + float(q) + float(z)]
        for data, i, j, k, p, q, z in zip(
            metrics_data, rank1, rank2, rank3, rank4, rank5, rank6
        )
    ]
    for i in range(len(tmp)):
        tmp[i][0]["rank"] = tmp[i][1]
    tmp = sorted(tmp, key=lambda x: x[1])

    writer.writerows([pair[0] for pair in tmp])


## save models
## these files can cost a lot of disk space, use them only when you want to bypass the training process
with open(
    "models\\" + location + "_" + label + "_" + model_type + "_best_fit" + ".pkl", "wb"
) as f:
    pickle.dump(best_fit_model, f)
with open(
    "models\\" + location + "_" + label + "_" + model_type + "_best_predict" + ".pkl",
    "wb",
) as f:
    pickle.dump(best_predict_model, f)
