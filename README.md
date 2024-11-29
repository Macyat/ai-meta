# ai-meta
this is a repo for predicting water quality with UV-VIS spectrum data.

## you need to change all the places that mention the location of an input data(the .csv) to your own path
"move the files in the "data" folder to your local path, and change the "parent_folder" in "run_lab.py" and "run_futian.py"
## how to set up the requirement libs
make install

## examples
type the following codes in your command line

### for the futian river dataset (scanned by Shimadzu):
python run_futian.py --location daojin --select 0 --compared_label COD

### for the futian river dataset (scanned by Gaolitong):
python run_futian.py --location gaolitong --select 0 --compared_label COD

### for the lab dataset (scanned by Gaolitong):
python run_lab.py --location gaolitong --compared_label COD

### for the lab dataset (scanned by Shimadzu):
python run_lab.py --location daojin --compared_label COD

## how to train for a single element and on a single model
python Train.py --label KMNO --start 751 --end -1 --first_wave 11 --model_type ada_lasso --cars_iterations 1 --location Lab_gaolitong --compared_label TP --parent_folder E:\Matlab\futian\futian\futian1\raw_data\

## to update all(choose any label to be compared with)
python daily_update.py --compared_label COD

