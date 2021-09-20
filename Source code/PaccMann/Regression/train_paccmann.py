import tensorflow as tf
from tensorflow.compat.v1.data import Dataset
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.32
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from tensorflow import keras
from tensorflow.keras.backend import clear_session


import numpy as np
from pathlib import Path
import pickle as pkl
import os.path
import sys

from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError

from model import build_model

#-------------------------------------------------------------------------
#                              METRICS
#-------------------------------------------------------------------------
def get_results(y_true, y_pred):
    
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MSE = mean_squared_error(y_true, y_pred, squared=True)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    return (RMSE, MSE, MAE, R2)


# Function to train PaccMann model
def train_paccmann(data, params, lr, n_epochs, path_to_ckpt, savepklto):
    tf.keras.backend.clear_session()
    
    # Load data generator built from the five folds created previously
    train_dataset, val_dataset, val_y = data
        
    # Build and compile model
    model = build_model(params)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, epsilon=1e-08), loss='mean_squared_error', metrics=[RootMeanSquaredError(), MeanSquaredError(), MeanAbsoluteError()])        
    
    print(model.summary())
    
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.00005, patience=1000, restore_best_weights=True, verbose=1)
   
    path_to_checkpoints = path_to_ckpt
    
    u = params.get('u') # number of hidden units == embedding size
    da = params.get('da') # attention depth
    act = params.get('activation') # activation fn for dense layers
    dc = params.get('dc') # dropout coefficient
    l2_lambda = params.get('l') # lambda for l2 regularization
    pos_enc = params.get('positional_encoding') # positional encoding
    dense_sizes_lst = params.get('dense_sizes') # No. of hidden units per dense layer in the model 
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'{path_to_checkpoints}/paccmann-{u}-{da}-{act}-{dc}-{l2_lambda}-{pos_enc}-{dense_sizes_lst[0]}.ckpt',
        save_best_only=True, save_weights_only= True)
    
    # Train
    model.fit(train_dataset,epochs=n_epochs,validation_data=val_dataset, callbacks=[earlystop_callback, checkpoint_callback],
                        verbose=0)
  
    outcome = model.evaluate(val_dataset)
    L = outcome[0]
    RMSE = outcome[1]
    MSE = outcome[2]
    MAE = outcome[3]
        
    # R2
    y_pred = model.predict(val_dataset)
    R2 = r2_score(val_y, y_pred)
    
    # Optional: save results to a pickle object associated to the checkpoint
    # to_pickle = (L, RMSE, MSE, MAE, R2)
    # pkl.dump(to_pickle, open(Path(savepklto) / f'./paccmann-{u}-{da}-{act}-{dc}-{l2_lambda}-{pos_enc}-{dense_sizes_lst[0]}.p', 'wb')