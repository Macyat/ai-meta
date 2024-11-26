import subprocess
import sys
from fileinput import filename

model_types = [
    # "SDG",
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
folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\daojin\\same_as_gaolitong"
filename = "merge_data_daojin.csv"
# labels = ["AN"]


for t in model_types:
    for l in labels:
        start = "0"
        end = "364"
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
                " -cars_iterations 1 -location FUTIAN_daojin",
                " -compared_label TN",
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
