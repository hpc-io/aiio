# first neural network with keras tutorial



##
## https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
##

### https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
###https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
###https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
###https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

##https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/

##file_tot_performace_tagged="/project/projectdirs/m2621/dbin/Cori_archive_2019_un_taz_parsered_tagged.csv"

##file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training.csv"
##plot_result_of="/global/homes/d/dbin/IODiagnoser/XGBRegressor_learning_curve.pdf"
##model_save_file="/global/cfs/cdirs/m2621/dbin/io-ai-model-xgboost-sparse-"

import time
file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training-v1.csv"


time_str=time.strftime("%Y%m%d-%H%M%S")

plot_result_file_name="/global/homes/d/dbin/IODiagnoser/io-ai-model-xgb-sparse-learning-curve-"+time_str+".pdf"
model_save_file_name="/global/cfs/cdirs/m2621/dbin/io-ai-model-xgb-sparse-"+time_str+".joblib"

print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)
    
    
    
is_kflod_test_flag=False
kflod_n_splits=3
##file_tot_performace_tagged="/global/cscratch1/sd/dbin/2019-01-total-tagged.csv"

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
from xgboost import XGBRegressor
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


##XGBRegressor.set_config(verbosity=2)
model = XGBRegressor(verbosity=2, n_estimators=10000, random_state=seed_value, early_stopping_rounds=10, eval_metric='rmse')
##xbg_reg=model.fit(X, Y)
##score = model.score(X, Y)  


if is_kflod_test_flag:
    kfold = KFold(n_splits=kflod_n_splits)
    scorer = MultiScorer({                                               # Create a MultiScorer instance
      'rmse': (mean_squared_error, {'squared': False}),
      'r2': (r2_score, {})               # Param 'average' will be passed to precision_score as kwarg 
    })
    #results = cross_val_score(model, X, Y, cv=kfold, scoring='neg_root_mean_squared_error')
    cross_val_score(model, X, Y, cv=kfold, scoring=scorer)
    #print("cross_val_score_results =", results)
    #print("rmse: %.20f (%.20f) " % (results.mean(), results.std()))
    results = scorer.get_results()                                       # Get a dict of lists containing the scores for each metric
    for metric in results.keys():                                        # Iterate and use the results
      print("%s: %.3f" % (metric, average(results[metric])))
else:
    #https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    print("X_train.type=", type(X_train))
    print("X_train.shape=", X_train.shape)

    # define the datasets to evaluate each iteration
    evalset = [(X_train, y_train), (X_test,y_test)]
    # fit the model
    model_return=model.fit(X_train, y_train,  eval_set=evalset)
    # evaluate performance
    yhat = model.predict(X_test)
    rmse_score = mean_squared_error(y_test, yhat, squared=False)
    print('rmse: %.3f' % rmse_score)
    # retrieve performance metrics
    results = model.evals_result()
    # plot learning curves
    pyplot.plot(results['validation_0']['rmse'], label='train')
    #pyplot.plot(results['validation_1']['rmse'], label='test')
    # show the legend
    #pyplot.legend()
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Loss')
    # show the plot
    pyplot.savefig(plot_result_file_name)  
    pyplot.show()
    #joblib.dump(model, model_save_file+time.strftime("%Y%m%d-%H%M%S")+".joblib") 
    #print(model_save_file+time.strftime("%Y%m%d-%H%M%S")+".joblib")
    #joblib.dump(model, model_save_file+time.strftime("%Y%m%d-%H%M%S")+".joblib") 
    #model_save_file_name=model_save_file+time.strftime("%Y%m%d-%H%M%S")+".joblib"
    joblib.dump(model, model_save_file_name) 
    print("plot_result_file_name =", plot_result_file_name)
    print("model_save_file_name=", model_save_file_name)
    print(model_return)


#print(score)

## Get feature importatnce https://stackabuse.com/bytes/get-feature-importance-from-xgbregressor-with-xgboost/
##xbg_reg.get_booster().get_score(importance_type='gain')

#import pandas as pd
#f_importance = xbg_reg.get_booster().get_score(importance_type='gain')
#importance_df = pd.DataFrame.from_dict(data=f_importance,  orient='index')
#importance_df.plot.bar()


