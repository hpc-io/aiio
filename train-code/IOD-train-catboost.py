##https://catboost.ai/en/docs/concepts/python-usages-examples

import time
file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training-v1.csv"


time_str=time.strftime("%Y%m%d-%H%M%S")

plot_result_file_name="/global/homes/d/dbin/IODiagnoser/io-ai-model-catboost-sparse-learning-curve-"+time_str+".pdf"
model_save_file_name="/global/cfs/cdirs/m2621/dbin/io-ai-model-catboost-sparse-"+time_str+".joblib"


print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)


from numpy import loadtxt
from matplotlib import pyplot
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from numpy import absolute
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from multiscorer import MultiScorer
from numpy import average
import joblib
import time
import scipy.sparse

## Set random seed
seed_value=48
import os
import random
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

from catboost import CatBoostRegressor



# load the train dataset
dataset = loadtxt(file_tot_performace_tagged, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
X = dataset[:,0:n_dims-1]

print("Before sparse.csr_matrix = ", type(X))
X=scipy.sparse.csr_matrix(X)
print("After  sparse.csr_matrix = ", type(X))

Y = dataset[:,n_dims-1]
print("max(Y) =", max(Y), ", min(Y) =", min(Y))
    
input_dim_size = n_dims -1
print("input_dim_size = ", input_dim_size)


#n_estimators=100000
model = CatBoostRegressor(n_estimators=900000, random_state=seed_value, use_best_model=True, eval_metric='RMSE', thread_count=-1)
 


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
print("X_train.type=", type(X_train))
print("X_train.shape=", X_train.shape)

# define the datasets to evaluate each iteration
evalset = [(X_train, y_train), (X_test,y_test)]
# fit the model
# ,  plot_file=plot_result_file_name
model.fit(X_train, y_train, verbose=1,  eval_set=evalset, early_stopping_rounds=10)

# evaluate performance
yhat = model.predict(X_test)
rmse_score = mean_squared_error(y_test, yhat, squared=False)
print('rmse: %.3f' % rmse_score)


#lgb.plot_metric(model, xlabel='Iteration', ylabel='Loss', dataset_names=['valid_1'])
#pyplot.savefig(plot_result_file_name)  
#pyplot.show()
#results = model.evals_result_
results = model.get_evals_result()
pyplot.plot(results['validation_1']['RMSE'], label='train')
pyplot.xlabel('Iteration')
pyplot.ylabel('Loss')
pyplot.savefig(plot_result_file_name)  
pyplot.show()

joblib.dump(model, model_save_file_name) 
print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)
#print(model_return)




