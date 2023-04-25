##%%capture cap --no-stderr

#file_tot_performace_tagged="/global/cfs/cdirs/m2621/dbin/Cori_archive_2019_un_taz_parsered_tagged_orig.csv"

#file_tot_performace_tagged="/project/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training.csv"
##file_tot_performace_tagged="/global/cscratch1/sd/dbin/2019-01-total-tagged.csv"

#file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training_nonLogTranFeature.csv"
#plot_result_of="/global/homes/d/dbin/IODiagnoser/tabnet_learning_curve.pdf"
#model_save_file="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-"
#model_save_file_dir="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-dir-"

is_kflod_test_flag=False
kflod_n_splits=3
kflod_n_splits_result_file="/global/homes/d/dbin/IODiagnoser/tabnet_3fold_result"


import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
#model_save_file="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-sparse-"
#model_save_file_dir="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-dir-sparse-"

import time


file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training-v1.csv"

time_str=time.strftime("%Y%m%d-%H%M%S")
plot_result_file_name="/global/homes/d/dbin/IODiagnoser/io-ai-model-tabnet-learning-curve-"+time_str+".pdf"
model_save_file_name="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-"+time_str+".joblib"
model_save_file_dir="/global/cfs/cdirs/m2621/dbin/io-ai-model-tabnet-"+time_str+"-dir"

print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)
print("model_save_file_dir=", model_save_file_dir)


from numpy import loadtxt
from matplotlib import pyplot
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from numpy import absolute
from numpy import mean
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras import backend
from sklearn.model_selection import train_test_split
from tensorflow import keras
import joblib
import time
from multiscorer import MultiScorer
from numpy import average
import keras

import scipy.sparse


# load the train dataset
dataset = loadtxt(file_tot_performace_tagged, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
X = dataset[:,0:n_dims-1]
Y = dataset[:,n_dims-1].reshape(-1, 1)
print(type(X))
print("max(Y) =", max(Y), ", min(Y) =", min(Y))
print("x.shape=", X.shape, ", Y.shape = ", Y.shape)    
input_dim_size = n_dims -1
print("input_dim_size = ", input_dim_size)

#X=scipy.sparse.csr_matrix(X)


## Set random seed
seed_value=48
import os
import random
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


model = TabNetRegressor(verbose=1, seed=seed_value)

if is_kflod_test_flag:
    #kfold = KFold(n_splits=2)
    #results = cross_val_score(model, X, Y, cv=kfold, scoring='neg_root_mean_squared_error', n_jobs=-1)
    #print("Standardized: %.20f (%.20f) R2" % (results.mean(), results.std()))
    kfold = KFold(n_splits=kflod_n_splits)
    scorer = MultiScorer({                                               # Create a MultiScorer instance
      'rmse': (mean_squared_error, {'squared': False}),
      'r2': (r2_score, {})               # Param 'average' will be passed to precision_score as kwarg 
    })
    cross_val_score(model, X, Y, cv=kfold, scoring=scorer)
    results = scorer.get_results()                                       # Get a dict of lists containing the scores for each metric
    for metric in results.keys():                                        # Iterate and use the results
      print("%s: %.3f" % (metric, average(results[metric])))
    with open('kflod_n_splits_result_file', 'w') as f:
        f.write("%s: %.3f" % (metric, average(results[metric])))
else:
    #https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
    # fit the model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    print("X_train.shape=", X_train.shape)

    evalset = [(X_train, y_train), (X_test,y_test)]
    # fit the model
    results=model.fit(X_train, y_train, eval_set=evalset,  eval_metric=['rmse'], patience=10)
    pyplot.plot(model.history['val_0_rmse'])
    pyplot.plot(model.history['val_1_rmse'])
    #pyplot.legend()
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    pyplot.savefig(plot_result_file_name)  
    #pyplot.show()
    
    
    # evaluate performance
    yhat = model.predict(X_test)
    rmse_score = mean_squared_error(y_test, yhat, squared=False)
    print('rmse: %.3f' % rmse_score)
    
    joblib.dump(model, model_save_file_name) 
    print(model_save_file_name)
    model.save_model(model_save_file_dir)
    print(model_save_file_dir)



