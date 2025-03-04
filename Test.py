import argparse
import csv
import os
import pickle
import random

import pandas as pd

parser = argparse.ArgumentParser(description="Entry for testing")
# parser.add_argument("-re_train", type=str,help="Retrain or not")
parser.add_argument("--file_path", "-p", type=str, help="test data file name")
parser.add_argument("--model_name", "-m", type=str, help="model name")
parser.add_argument("--first_col", "-f", type=str, help="first column of test data")
parser.add_argument("--last_col", "-l", type=str, help="last column of test data")
parser.add_argument("--label", "-lb", type=str, help="the element to be predicted")
parser.add_argument(
    "--location", "-lct", type=str, help="the location of the test data"
)
# parser.add_argument("--start", "-s", type=int, help="the first row to train")
# parser.add_argument("--end", "-e", type=int, help="the last row to train")
# parser.add_argument(
#     "--evaluate_start", "-es", type=int, help="the first row to be evaluated"
# )
# parser.add_argument(
#     "--evaluate_end", "-ee", type=int, help="the last row to be evaluated"
# )
# parser.add_argument(
#     "--man_made_evaluate", "-mme", type=int, help="whether to evaluate man-made samples"
# )
# parser.add_argument(
#     "--first_wave",
#     "-f",
#     type=int,
#     help="the starting wavelength to be selected for training",
# )
# parser.add_argument("--model_type", "-m", type=str, help="the model selected")
# parser.add_argument(
#     "--cars_iterations", "-ca", type=int, help="the times for running cars"
# )
# parser.add_argument(
#     "--location", "-lo", type=str, help="where the samples are collected"
# )
# parser.add_argument("--parent_folder", "-p", type=str, help="where the data locates")
# parser.add_argument("--base_model_type", "-b", type=str, help="pretrained model type", default="pls")

if not os.path.exists("test_output"):
    os.makedirs("test_output")

args = parser.parse_args()
filename = args.file_path
model_type = args.model_name
first_col = int(args.first_col)
last_col = int(args.last_col)
label = args.label
location = args.location
# threshold = args.threshold
# lower_bound = args.lower_bound
# upper_bound = args.upper_bound

data = pd.read_csv(filename, encoding="gbk")
intput_data = data.values


def valid_guard(res, threshold, lower_bound, upper_bound):
    res1 = res.copy()
    for i in range(len(res)):
        if res[i] <= threshold:
            res1[i] = random.uniform(lower_bound, upper_bound)
    return res1


with open("models\\" + model_type + ".pkl", "rb") as f:
    model = pickle.load(f)
# res = valid_guard(model.predict(data[:, first_col:last_col]), threshold, lower_bound, upper_bound)
# data.values[start:end, first_wave : (L + 1 - 10)].astype("float64")
res = model.predict(data.values[:, first_col:last_col])
with open(
    "test_output\\" + location + "_" + model_type + ".csv",
    "w",
    newline="",
) as f:
    write = csv.writer(f)
    for r in res:
        write.writerow([r])
