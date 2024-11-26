import subprocess
import sys

import argparse
from fileinput import filename

parser = argparse.ArgumentParser(description="Algorithm for lab data")
parser.add_argument(
    "-location",
    type=str,
    help="gaolitong data or daojin data",
)
parser.add_argument(
    "-compared_label",
    type=str,
    help="another element to be compared with",
)

args = parser.parse_args()
location = args.location
compared_label = args.compared_label

if location == "daojin":
    location = "Lab_daojin"
    folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\daojin"
    start = "383"
    filename = "merge_data_daojin.csv"
else:
    location = "Lab_gaolitong"
    folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data"
    start = "751"
    filename = "merge_data_gaolitong.csv"

model_types = [
    "SDG",  # this is bad at the moment
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
        command = "".join(
            [
                "python Train.py -label ",
                l,
                " -start ",
                start,
                " -end -1 -first_wave 11 -model_type ",
                t,
                " -cars_iterations 1 -location ",
                location,
                " -compared_label ",
                compared_label,
                " -folder ",
                folder,
                " -filename ",
                filename,
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
