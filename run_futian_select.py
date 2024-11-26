import subprocess
import sys

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
    "ransar",
    "theilsen",
    "pls_cars",
]
labels = ["KMNO", "TN", "TP", "AN", "COD", "TUR"]
# labels = ["AN"]
folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data\\same_as_daojin"
filename = "merge_data_gaolitong.csv"

for t in model_types:
    for l in labels:
        start = "0"
        if l == "TN":
            start = "10"
        command = "".join(
            [
                "python Train.py -label ",
                l,
                " -start ",
                start,
                " -end 364 -first_wave 11 -model_type ",
                t,
                " -cars_iterations 1 -location FUTIAN",
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
