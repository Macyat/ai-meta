import subprocess
import sys

model_types = ['lgbm', 'ridge', 'lasso', 'ada_ridge', 'ada_lasso', 'pls', 'gpr_pca', 'gpr','pls_cars']
labels = ['KMNO', 'TN', 'TP', 'AN','COD']



for t in model_types:
    for l in labels:
        command = "".join(["python Train.py -label ", l, " -start 0 -end 363 -first_wave 11 -model_type ", t,  " -cars_iterations 1 -location FUTIAN"])
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(process.stdout)
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
                sys.stdout.flush()