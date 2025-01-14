import os
import subprocess
import sys
import argparse
from fileinput import filename

parser = argparse.ArgumentParser(description="Algorithm for gaolitong")
parser.add_argument(
    "--location",
    "-lo",
    type=str,
    help="gaolitong data or daojin data",
)
parser.add_argument(
    "--select",
    "-s",
    type=str,
    help="whether to select the samples that both have gaolitong data and daojin data",
)
parser.add_argument(
    "--compared_label",
    "-c",
    type=str,
    help="another element to be compared with",
)


args = parser.parse_args()
select_same_period = args.select in ["True", "Yes", "Y", "1", "true", "y", "yes"]
compared_label = args.compared_label
location = args.location

parent_folder = "E:\\Program Files\\MATLAB\\Matlab\\futian\\futian\\futian1\\raw_data"

if location == "gaolitong":
    if select_same_period:
        end = "349"
        evaluate_end = "349"
        location = "Futian_gaolitong_select"
    else:
        end = "751"
        evaluate_end = "751"
        location = "Futian_gaolitong"
elif location == "daojin":
    if select_same_period:
        end = "349"
        evaluate_end = "349"
        location = "Futian_daojin_select"
    else:
        end = "364"
        evaluate_end = "364"
        location = "Futian_daojin"
elif location == "gaolitong_all":
    end = "-1"
    evaluate_end = "751"
    location = "Futian_gaolitong_all"
elif location == "kukeng":
    end = "52"
    evaluate_end = "52"
else:
    end = "-1"
    evaluate_end = "-1"


model_types = [
    "multi",
    "SDG",
    "Basic",
    "ada_basic",
    "lars",
    "lassolars_bic",
    "lassolars_aic",
    "lasso_lars",
    "orth",
    "multi_elst",
    "WLS",
    "tweedie",
    "gamma",
    "passive",
    "poisson",
    "quantile",
    "huber",
    "elst",
    "bayes_ridge",
    "lgbm",
    "lasso",
    "ada_lasso",
    "pls",
    "gpr_pca",
    "gpr",
    "ridge",
    "ada_ridge",
    "pls_cars",
    "pls_multi",
    "pls_multi_cars",
    "ARD",
    "ransar",
    "theilsen",
]
labels = ["KMNO", "TP", "AN", "TUR", "COD"]
# labels = ["TN"]

for t in model_types:
    for l in labels:
        start = "0"
        evaluate_start = "0"
        if (
            l == "TN" and "gaolitong" in location
        ):  # for TN, the first 10 gaolitong labels are nonsense
            start = "10"
            evaluate_start = "10"
        command = "".join(
            [
                "python Train.py -l ",
                l,
                " -s ",
                start,
                " -e ",
                end,
                " -es ",
                evaluate_start,
                " -ee ",
                evaluate_end,
                " -mme ",
                "0",
                " -f 11 -m ",
                t,
                " -ca 1 -lo ",
                location,
                " -c ",
                compared_label,
                " -p ",
                '"',
                parent_folder,
                '"',
            ]
        )
        print(command)
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # print(process.stdout)
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()
