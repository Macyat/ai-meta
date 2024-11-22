import csv
import pandas as pd
import pickle
import numpy as np
import random
import warnings
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
import models
import utils
import os
import argparse
import preprocessing
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Algorithm for gaolitong")
# parser.add_argument("-re_train", type=str,help="Retrain or not")
parser.add_argument("-label", type=str, help="element to predict")
parser.add_argument("-start", type=int, help="the first row to train")
parser.add_argument("-end", type=int, help="the last row to train")
parser.add_argument("-first_wave", type=int, help="the starting wavelength to be selected for training")
parser.add_argument("-model_type", type=str, help="the model selected")
parser.add_argument("-cars_iterations", type=int, help="the times for running cars")
parser.add_argument("-location", type=str, help="where the samples are collected")

###example bash code###
# python Train.py -label CODMn -start 0 -end 363 -first_wave 11 -model_type pls -cars_iterations 1


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

# folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data\\same_as_daojin"
folder = "data\\"
filename = os.path.join(folder,"merge_data_gaolitong.csv")
data = pd.read_csv(filename,encoding = 'gbk')

# folder = "E:\\Matlab\\futian\\futian\\futian1\\raw_data\\data"
configs = pd.read_csv(os.path.join(folder,"configs_gaolitong.csv"))


### read the arguments from command line
args = parser.parse_args()
label = args.label
start = args.start
end = args.end
if end == -1:
    end = len(data)
first_wave = args.first_wave
model_type = args.model_type
location = args.location
cars_iterations = args.cars_iterations
# re_train = args.re_train in ["True", "Yes", "Y", "1", "true", "y", "yes"]


### only evaluate results where the turbidity is lower the TUR_bound
TUR_bound = 20
config = preprocessing.get_configs(label, data, start, end, first_wave)


evaluate_idx = utils.select_idx(config['target'], config['TUR'], TUR_bound, config['lower_bound'], config['upper_bound'])
X1 = config['X1']

print(f"Train data shape:{X1.shape}, Target shape:{end - start}, days shape:{config['days'].shape}, "
      f"unique days shape:{np.unique(config['days']).shape}")



### default values for CARS
min_minRMSECV = 100
lis_best = list(range(611))
rindex_best = 30

# for i in range(cars_iterations):
#     lis, minRMSECV,rindex = CARS.CARS_Cloud(X1_snv, config['target'], config['days'], 800, N=20)
#
#     if minRMSECV < min_minRMSECV:
#         min_minRMSECV = minRMSECV
#         lis_best = lis
#         rindex_best = rindex
#
# print('wavelengths selected', lis_best)
# print('')
# print('number of latent vectors', rindex_best + 1)
# print('min RMSE by cars', min_minRMSECV)
# print('')


###model training###
if model_type == 'pls':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.pls_meta(X1, config['target'], config['cut_bound'], config['days'], False, location, label, model_type))
if model_type == 'pls_cars':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.pls_meta(X1, config['target'], config['cut_bound'], config['days'], True, location, label, model_type))
elif model_type == 'ridge':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.ridge_meta(X1, config['target'], config['cut_bound'], config['days'], configs, label, False))
elif model_type == 'ada_ridge':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.ridge_meta(X1, config['target'], config['cut_bound'], config['days'], configs, label, True))
elif model_type == 'lasso':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.lasso_meta(X1, config['target'], config['cut_bound'], config['days'], configs, label, False))
elif model_type == 'ada_lasso':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.lasso_meta(X1, config['target'], config['cut_bound'], config['days'], configs, label, True))
elif model_type == 'gpr':
    kernel_ = 1.0 * RationalQuadratic(length_scale = 1, alpha = 1) + WhiteKernel(1e-1) + ConstantKernel(constant_value = np.mean(config['target']))
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model, sigma = (
        models.gpr_meta(X1, config['target'], config['cut_bound'], config['days'], kernel_))
elif model_type == 'gpr_pca':
    kernel_ = 1.0 * RationalQuadratic(length_scale = 1, alpha = 1) + WhiteKernel(1e-1) + ConstantKernel(constant_value = np.mean(config['target']))
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model, sigma = (
    models.gpr_meta(PCA(n_components=10).fit_transform(X1), config['target'], config['cut_bound'], config['days'], kernel_))
elif model_type == 'lgbm':
    res_train, best_score_fit, best_score_predict, best_fit_model, best_predict_model = (
        models.lgbm_meta(X1, config['target'], config['cut_bound'], config['days'], configs, label))



### processing the outcomes
res_train_cap = models.res_lower_cap(config['lower_bound'], config['ranges'], res_train) ### remove invalid values

utils.write_res(res_train_cap, location, label, model_type)

mape, r2_score, rmse = utils.plot(res_train_cap, config['target'], config['TUR'], evaluate_idx, label, model_type, location)
good_predict_percentage, bad_idx = utils.evaluate(res_train_cap, config['target'], config['ranges'], evaluate_idx,
                                                  config['abs_error_bound'], config['mape_bound'])
print('rate of reaching the standard', good_predict_percentage)



### update model performances
if os.path.exists('metrics\\' + location + "_" + label  + ".csv"):
    with open('metrics\\' + location + "_" + label  + ".csv", "r",newline='') as f:
        reader = csv.DictReader(f)
        metrics_data = [row for row in reader]
    flag = False
    for row in metrics_data:
        if row['model_type'] == model_type:
            flag = True
            if mape < float(row['mape']):
                row['mape'] = mape
                row['r2_score'] = r2_score
                row['rmse'] = rmse
                row['rate of reaching the standard'] = good_predict_percentage
    if not flag:
        metrics_data.append({'model_type': model_type, 'mape': mape, 'r2_score': r2_score, 'rmse': rmse,
                             'rate of reaching the standard': good_predict_percentage})

else:
    metrics_data = []
    metrics_data.append({'model_type': model_type, 'mape': mape, 'r2_score': r2_score, 'rmse': rmse,
                         'rate of reaching the standard': good_predict_percentage})


with open('metrics\\' + location + "_" + label  + ".csv", "w",newline='') as f:
    fieldnames = ['model_type', 'mape', 'r2_score', 'rmse', 'rate of reaching the standard']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    metrics_data.sort(key=lambda x: float(x['mape']))
    writer.writerows(metrics_data)



### save models
with open('models\\' + location + "_" + label + "_" + model_type + "_best_fit" + ".pkl", "wb") as f:
    pickle.dump(best_fit_model, f)
with open('models\\' + location + "_" + label + "_" + model_type + "_best_predict" + ".pkl", "wb") as f:
    pickle.dump(best_predict_model, f)










