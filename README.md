# ai-meta
this is a repo for predicting water quality with UV-VIS spectrum data.

## how to set up
make install

## how to train models and make predictions for all elements and with all models
type the following codes in your command line

### for the futian river dataset:
python run_futian.py

### for the lab dataset:
python run_lab.py

## how to train for a single element and with a single model
python Train.py -label KMNO -start 364 -end -1 -first_wave 11 -model_type ridge -cars_iterations 1 -location LAB
