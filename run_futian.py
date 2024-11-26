import subprocess
import sys

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
# labels = ["AN"]


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
                " -end 751 -first_wave 11 -model_type ",
                t,
                " -cars_iterations 1 -location FUTIAN_all",
                " -compared_label TN",
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
