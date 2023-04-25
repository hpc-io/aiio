##
## https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
##
### https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
###https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
###https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
###https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
##https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/

file_tot_performace_tagged="/global/cfs/projectdirs/m2621/dbin/Cori_archive_19_20_21_22_un_taz_parsered_tagged_for_training-v1.csv"

plot_result_of="/global/homes/d/dbin/IODiagnoser/MLP_learning_curve.pdf"

model_save_file="/global/cfs/cdirs/m2621/dbin/io-ai-model-mlp-sparse-"


is_kflod_test_flag=False
kflod_n_splits=2

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
from keras import backend
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from keras.layers import BatchNormalization
import joblib
from multiscorer import MultiScorer


import scipy.sparse
import time


# load the train dataset
dataset = loadtxt(file_tot_performace_tagged, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
X = dataset[:,0:n_dims-1]
Y = dataset[:,n_dims-1]
print("max(Y) =", max(Y), ", min(Y) =", min(Y))

X=scipy.sparse.csr_matrix(X)


input_dim_size = n_dims -1
print("input_dim_size = ", input_dim_size)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1)) 
    
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(input_dim_size*2, input_shape=(input_dim_size,), kernel_initializer='normal', activation='relu'))
    for nus in range(input_dim_size*2-1, 0, -20):
        model.add(Dense(nus, kernel_initializer='normal', activation='relu')) 
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    #opt = keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer='adam')
    #print(model.summary())
    return model



if is_kflod_test_flag:
    model=KerasRegressor(model=baseline_model, epochs=100, batch_size=128, verbose=1)
    kfold = KFold(n_splits=kflod_n_splits)
    scorer = MultiScorer({                                               # Create a MultiScorer instance
      'rmse': (mean_squared_error, {'squared': False}),
      'r2': (r2_score, {})               # Param 'average' will be passed to precision_score as kwarg 
    })
    cross_val_score(model, X, Y, cv=kfold, scoring=scorer)
    results = scorer.get_results()                                       # Get a dict of lists containing the scores for each metric
    for metric in results.keys():                                        # Iterate and use the results
      print("%s: %.3f" % (metric, average(results[metric])))
else:
    model=baseline_model()
    #https://machinelearningmastery.com/tune-xgboost-performance-with-learning-curves/
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    print("X_train.type=", type(X_train))
    print("X_train.shape=", X_train.shape)
    # fit the model
    results=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, batch_size=256, verbose=1)
    pyplot.plot(results.history['loss'])
    #pyplot.legend()
    pyplot.xlabel('Iteration')
    pyplot.ylabel('loss')
    pyplot.savefig(plot_result_of)  
    pyplot.show()
    
    yhat = model.predict(X_test)
    rmse_score = mean_squared_error(y_test, yhat, squared=False)
    print('rmse: %.3f' % rmse_score)
    #joblib.dump(model, model_save_file) 
    model_save_file_name=model_save_file+time.strftime("%Y%m%d-%H%M%S")+".joblib"
    joblib.dump(model, model_save_file_name) 
    print(model_save_file_name)
   
