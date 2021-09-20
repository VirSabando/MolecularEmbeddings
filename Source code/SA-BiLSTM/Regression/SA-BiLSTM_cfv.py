from tensorflow import keras
from tensorflow.keras.backend import clear_session
import tensorflow as tf

from model import build_sa_bilstm_model
from train_sa_bilstm import train_sa_bilstm

import numpy as np
import os
import random
from pathlib import Path
from sys import argv
from data import load_data, Mol2vecLoader

# Sets a random seed globally (numpy, tf and keras). Necessary to obtain fully reproducible results.
# ------------------------------------------------------------------------------------
# Seed value
# Apparently you may use different seed values at each stage
seed_value= int(argv[1])

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# ------------------------------------------------------------------------------------

# Path of inputs and outputs
dataset_name = argv[2] # '<NAME>.csv'
ckpt_path = # <PATH TO CHECKPOINTS>
path_to_data = # <PATH TO DATASET>
path_to_model = # <PATH TO PICKLE OBJECT OF THE PRETRAINED MOL2VEC MODEL (model_300dim.pkl)>

# Coefficient for gradient clipping
cv = 0.3
n_epochs = 10000

# Adapt to your own data
padding_dict = {
    'ESOL':110, 
    'FreeSolv':48,
    'Lipophilicity':230
}

padding = padding_dict.get(dataset_name)

# Number of hidden units for BiLSTM (u)
lstm_hidden = int(argv[3]) # default is 128

# Number of parameters of weight vectors W1 and W2 (for self-attention)
da = int(argv[4]) # default is 10

# Number of attention heads
r = int(argv[5]) # default is 5

# Minibatch size
batch_size = 256

# Learning rate 
lr = 0.0001 

# ------------------------------------------------------------------------------------
# Make stratified folds for 5-fold CV
list_of_folds = load_data(path_to_data, path_to_model, pad_to=padding)
    
# Dimensions of mol2vec vectors (given by pretrained model)
d = 300

for idx, fold in enumerate(list_of_folds):
    
    path_to_ckpt = ckpt_path + '/cfv/checkpoints/folds/'+str(seed_value)+'/'+str(idx)
    savepklto = ckpt_path +'/cfv/results/folds/'+str(seed_value)+'/'+str(idx)
    
    if not Path(savepklto).exists():
        Path(savepklto).mkdir(parents=True, exist_ok=False)
    if not Path(path_to_ckpt).exists():    
        Path(path_to_ckpt).mkdir(parents=True, exist_ok=False)

    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
    
    # Load generators
    train_dataset = Mol2vecLoader(train_x, train_y, padding, batch_size)
    val_dataset = Mol2vecLoader(val_x, val_y, padding, batch_size)

    data = (train_dataset, val_dataset, val_y)

# ------------------------------------------------------------------------------------
    # Train
    with tf.device('gpu:0'):
        train_sa_bilstm(data, padding, lstm_hidden, d, da, r, lr, path_to_ckpt, cv, n_epochs, savepklto)
        keras.backend.clear_session()
