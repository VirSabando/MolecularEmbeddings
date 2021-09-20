import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np
import os
import sys
from sys import argv
import random

from data import load_descriptors, load_fingerprints, load_maccs, load_folds, DataLoader
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError

#-------------------------------------------------------------------------
#                              RANDOM SEEDS
#-------------------------------------------------------------------------
def set_seed(s):
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= s

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    #clear session before beginning
    clear_session()

#-------------------------------------------------------------------------
#                              METRICS
#-------------------------------------------------------------------------
def get_results(y_true, y_pred):
    
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MSE = mean_squared_error(y_true, y_pred, squared=True)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    return (RMSE, MSE, MAE, R2)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0)) 

#-------------------------------------------------------------------------
#                              DATA LOADING
#-------------------------------------------------------------------------
dataset = argv[2] # <ABSOLUTE_PATH_TO_CSV_FILE.csv>
dataset_name = argv[1] # ESOL or FreeSolv or Lipophilicity
mol_rep = argv[3] # D for descriptors, F for ECFP, M for MACCS keys, E for embeddings

# Retrieve folds of data
if mol_rep == 'D':
    list_of_folds = load_descriptors(dataset)
elif mol_rep == 'F':
    list_of_folds = load_fingerprints(dataset)
elif mol_rep == 'M':
    list_of_folds = load_maccs(dataset)
elif mol_rep == 'E':
    list_of_folds = load_folds(dataset)
else:
    print('ERROR: invalid mol_rep')

#-------------------------------------------------------------------------
#                              FFNN CONSTRUCTION
#-------------------------------------------------------------------------
         
act = 'relu'
mbs = 200

params_dict = {
    'ESOL_F':{'ln_lambda': 0.1, 'loss':'mean_squared_error'},
    'ESOL_M':{'ln_lambda': 0.1, 'loss':'mean_squared_error'},
    'ESOL_D':{'ln_lambda': 0.001, 'loss':'mean_squared_error'},
    'ESOL_E':{'ln_lambda': 0.01, 'loss':'mean_squared_error'},
    'FreeSolv_F':{'ln_lambda': 0.1, 'loss':root_mean_squared_error},
    'FreeSolv_M':{'ln_lambda': 0.01, 'loss':root_mean_squared_error},
    'FreeSolv_D':{'ln_lambda': 0.1, 'loss':root_mean_squared_error},
    'FreeSolv_E':{'ln_lambda': 0.01, 'loss':root_mean_squared_error},
    'Lipophilicity_F':{'ln_lambda': 0.01, 'loss':'mean_squared_error'},
    'Lipophilicity_M':{'ln_lambda': 0.01, 'loss':root_mean_squared_error}, 
    'Lipophilicity_D':{'ln_lambda': 0.001, 'loss':'mean_squared_error'},
    'Lipophilicity_E':{'ln_lambda': 0.01, 'loss':'mean_squared_error'}, 
}

params = params_dict.get(dataset_name+'_'+mol_rep)

ln_lambda = params.get('ln_lambda')
es_p = 500
    
    
# hidden units
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 10
n_outputs = 1

#adam optimizador
l_rate = 0.001

#dropout
prob_h1 = 0.25
prob_h2 = 0.1
prob_h3 = 0.05

n_epochs = np.iinfo(np.int32).max

#batch normalization
momentum_batch_norm = 0.95

def build_model(input_shape, mbs, ln, act):
    clear_session() 
    
    mini_batch_size = mbs

    #regularizer l2
    Ln_reg  = l2(ln)

    # definicion del modelo
    x = Input(shape=input_shape)

    a1 = Dense(n_hidden1, activation=act,kernel_initializer='glorot_uniform', kernel_regularizer = Ln_reg)(x)
    b1 = BatchNormalization(momentum=momentum_batch_norm)(a1)
    c1 = Dropout(prob_h1)(b1)

    a2 = Dense(n_hidden2, activation=act,kernel_initializer='glorot_uniform', kernel_regularizer = Ln_reg)(c1)
    b2 = BatchNormalization(momentum=momentum_batch_norm)(a2)
    c2 = Dropout(prob_h2)(b2)

    a3 = Dense(n_hidden3, activation=act,kernel_initializer='glorot_uniform', kernel_regularizer = Ln_reg)(c2)
    b3 = BatchNormalization(momentum=momentum_batch_norm)(a3)
    c3 = Dropout(prob_h3)(b3)

    salida = Dense(n_outputs)(c3)
    model = Model(inputs=x, outputs=salida)
    
    return model

#-------------------------------------------------------------------------
#                              FFNN TRAINING
#-------------------------------------------------------------------------

#optimizer
adam_opt = Adam(l_rate)

#early stopping
min_delta_val = 0.0005

#callbacks
#early stopping
es = EarlyStopping(monitor='val_loss', min_delta=min_delta_val, patience=es_p, verbose=1, mode='min', restore_best_weights=True)

seed_lst = [42,17,2131,222,3,342,44,6,7567,980,99]

for seed in seed_lst:
    for idx, fold in enumerate(list_of_folds):
        set_seed(seed)
        t, v = fold
        train_x, train_y = t
        val_x, val_y = v
                   
        input_shape = train_x.shape[1:]
        train_dataset = DataLoader(train_x, train_y, mbs)
        val_dataset = DataLoader(val_x, val_y, mbs)
    
        # build the model
        model = build_model(input_shape, mbs, ln_lambda, act)
                
        # compile the model
        model.compile(optimizer=adam_opt, loss=params.get('loss'), metrics=[RootMeanSquaredError(), MeanSquaredError(), MeanAbsoluteError()])

        # train the model
        training_data = model.fit(train_dataset,epochs=n_epochs,validation_data=val_dataset,callbacks=[es],verbose=0)
                

        # final evaluation
        outcome = model.evaluate(val_dataset)
        L = outcome[0]
        RMSE = outcome[1]
        MSE = outcome[2]
        MAE = outcome[3]
        
        # R2 
        y_pred = model.predict(val_dataset)
        R2 = r2_score(val_y, y_pred)

        print('seed '+str(seed)+' - results on fold '+ str(idx)+' - '+dataset_name + ', '+ str(RMSE)+','+ str(MSE)+','+str(MAE)+','+str(R2))
        sys.stdout.flush()
                
        clear_session()
        
    print('--------')
