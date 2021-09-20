import tensorflow as tf
from tensorflow.keras.backend import clear_session
from data import load_data, DataFeeder

from model import build_model
from train_paccmann import train_paccmann

import os
import random
import numpy as np
from pathlib import Path
from sys import argv

#-------------------------------------------------------------
def set_seed(seed_value):
    # Seed value
    # Apparently you may use different seed values at each stage

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
#-------------------------------------------------------------

# Path of inputs and outputs
dataset_name = argv[2] # '<NAME>.csv'
ckpt_path = # <PATH TO CHECKPOINTS>
path_to_data = # <PATH TO DATASET>

# Coefficient for gradient clipping
n_epochs = 10000

padding_dict = {
    'ESOL':97, 
    'FreeSolv':53,
    'Lipophilicity':205
}

padding = padding_dict.get(dataset_name)

t = 263 # vocabulary size - hardcoded
ds_lst = [[150, 50, 10], [100, 20, 5], [512, 256, 64, 16], [100, 50, 20, 5]]


A = {'n': padding,'t': t,'u': 100,'da': 20,'activation': 'relu','dc': 0.25,
     'l': 0.005,'positional_encoding': False,'dense_sizes': ds_lst[0],'n_layers':3}
B = {'n': padding,'t': t,'u': 50,'da': 20,'activation': 'relu','dc': 0.25,
     'l': 0.005,'positional_encoding': False,'dense_sizes': ds_lst[0],'n_layers':3}

params_dict = {
    'ESOL':A,
    'FreeSolv':A,
    'Lipophilicity':B
}

params = params_dict.get(dataset_name)

# Learning rate 
lr = 0.0001 

#-------------------------------------------------------------

# Make stratified folds for 5-fold CV
list_of_folds = load_data(path_to_data, pad_to=padding)

for idx, fold in enumerate(list_of_folds):

    seed_value = int(argv[1])
    set_seed(seed_value)
    path_to_ckpt = ckpt_path + '/cfv/checkpoints/folds/'+str(seed_value)+'/'+str(idx)
    savepklto = ckpt_path +'/cfv/results/folds/'+str(seed_value)+'/'+str(idx)
        
    if not Path(savepklto).exists():
        Path(savepklto).mkdir(parents=True, exist_ok=False)
    if not Path(path_to_ckpt).exists():    
        Path(path_to_ckpt).mkdir(parents=True, exist_ok=False)
        
    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
    
    batch_size = 256
    train_dataset = DataFeeder(train_x, train_y, batch_size)
    val_dataset = DataFeeder(val_x, val_y, batch_size)

    data = (train_dataset, val_dataset, val_y)
    
# ------------------------------------------------------------------------------------
    # Train
    with tf.device('gpu:0'):
        train_paccmann(data, params, lr, n_epochs, path_to_ckpt, savepklto)
        tf.keras.backend.clear_session()