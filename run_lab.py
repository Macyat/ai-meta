import os
import subprocess
import sys

import argparse
from fileinput import filename

parser = argparse.ArgumentParser(description="Algorithm for lab data")
parser.add_argument(
    "--location",
    "-l",
    type=str,
    help="gaolitong data or daojin data",
)
parser.add_argument(
    "--compared_label",
    "-c",
    type=str,
    help="another element to be compared with",
)

args = parser.parse_args()
location = args.location
compared_label = args.compared_label

parent_folder = "E:\\Program Files\\MATLAB\Matlab\\futian\\futian\\futian1\\raw_data"

if location == "daojin_guanlan":
    location = "Lab_daojin_guanlan"
    start = "364"
    end = "919"
elif location == "daojin_dankeng":
    location = "Lab_daojin_dankeng"
    start = "919"
    end = "-1"
elif location == "gaolitong_guanlan":
    location = "Lab_gaolitong_guanlan"
    start = "751"
    end = "1218"
elif location == "gaolitong_dankeng":
    location = "Lab_gaolitong_dankeng"
    start = "1218"
    end = "-1"


model_types = [
    # "multi",
    # "SDG",  # this is bad at the moment
    # "tweedie",
    # "gamma",
    # "passive",
    # "poisson",
    # "quantile",
    # "huber",
    # "elst",
    # "bayes_ridge",
    # "ridge",
    # "lgbm",
    # "lasso",
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
# labels = ["AN"]


for t in model_types:
    for l in labels:
        command = "".join(
            [
                "python Train.py -l ",
                l,
                " -s ",
                start,
                " -e ",
                end,
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
