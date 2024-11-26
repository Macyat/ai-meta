import subprocess
import sys
import argparse


parser = argparse.ArgumentParser(description="Algorithm for gaolitong")
parser.add_argument(
    "-select",
    type=str,
    help="whether to select the samples that both have gaolitong data and daojin data",
)
parser.add_argument(
    "-compared_label",
    type=str,
    help="another element to be compared with",
)


args = parser.parse_args()
select_same_period = args.select in ["True", "Yes", "Y", "1", "true", "y", "yes"]
compared_label = args.compared_label

if select_same_period:
    folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data\\same_as_daojin"
    end = "364"
    location = "Futian_gaolitong"
else:
    folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data"
    end = "751"
    location = "Futian_gaolitong_select"


model_types = [
    "SDG",
    "tweedie",
    "gamma",
    "passive",
    "poisson",
    "quantile",
    "huber",
    "elst",
    "bayes_ridge",
    "ridge",
    "lgbm",
    "lasso",
    "ada_ridge",
    "ada_lasso",
    "pls",
    "gpr_pca",
    "gpr",
    "pls_cars",
    "ransar",
    "theilsen",
]
labels = ["KMNO", "TN", "TP", "AN", "COD", "TUR"]


for t in model_types:
    for l in labels:
        start = "0"
        if l == "TN":  # for TN, the first 10 gaolitong labels are nonsense
            start = "10"
        command = "".join(
            [
                "python Train.py -label ",
                l,
                " -start ",
                start,
                " -end ",
                end,
                " -first_wave 11 -model_type ",
                t,
                " -cars_iterations 1 -location ",
                location,
                " -compared_label ",
                compared_label,
                " -folder ",
                folder,
                " -filename ",
                "merge_data_gaolitong.csv",
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
