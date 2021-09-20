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

from model import build_sa_bilstm_model


#-------------------------------------------------------------------------
#                              METRICS
#-------------------------------------------------------------------------
def get_results(y_true, y_pred):
    
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MSE = mean_squared_error(y_true, y_pred, squared=True)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    return (RMSE, MSE, MAE, R2)

# Using notation stated in the paper by Zheng et al
# n: number of tokens in each SMILES (after padding). It is THE SAME for all SMILES
# d: dimensionality of each token vector computed using mol2vec. Using the pretrained model, d=300
# u: number of hidden units per LSTM nn. In a Bidirectional LSTM, the final number of hidden units would be 2u (uu)
# da: number of parameters of weight vectors W1 and w2, as stated in the paper
# r: number of heads in the multi-head attention, also called number of attention rows
def train_sa_bilstm(data, cw, n, u, d, da, r, lr, path_to_ckpt, cv, n_epochs, savepklto):
    clear_session()
    
    # Load data generator built from the five folds created previously
    train_dataset, val_dataset, val_y = data
        
    # Build and compile model
    model = build_sa_bilstm_model(n=n, d=d, u=u, da=da, r=r)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipvalue=cv), loss='mean_squared_error', metrics=[RootMeanSquaredError(), MeanSquaredError(), MeanAbsoluteError()])
    
    # Optional: since these models can take a long time to train,
    # especially with big datasets, you might want to split the training sessions.
    # If there is a checkpoint you need to load, do it here ;)
    # ruta_ckpt =  f'{path_to_ckpt}/model-sa-bilstm-{n}-{u}-{da}-{r}-{lr}.ckpt.index'
    
    # if os.path.isfile(ruta_ckpt):
    #     model.load_weights(ruta_ckpt[:-6])
    #     print('Loading pretrained model...')
    
    # Summary
    print(model.summary())
    
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.00005, patience=1000, restore_best_weights=True, verbose=1)
   
    path_to_checkpoints = path_to_ckpt
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'{path_to_checkpoints}/model-sa-bilstm-{n}-{u}-{da}-{r}-{lr}.ckpt',
        save_best_only=True, save_weights_only= True)
    
    # Train
    model.fit(train_dataset,epochs=n_epochs,validation_data=val_dataset,callbacks=[earlystop_callback, checkpoint_callback],verbose=0)
  
    outcome = model.evaluate(val_dataset)
    L = outcome[0]
    RMSE = outcome[1]
    MSE = outcome[2]
    MAE = outcome[3]

    # R2
    y_pred = model.predict(val_dataset)
    R2 = r2_score(val_y, y_pred)
    
    # Optional: save results to a pickle object
    # to_pickle = (L, RMSE, MSE, MAE, R2)
    #pkl.dump(to_pickle, open(Path(savepklto) / f'./sa_bilstm-{n}-{u}-{da}-{r}-{lr}.p', 'wb'))
