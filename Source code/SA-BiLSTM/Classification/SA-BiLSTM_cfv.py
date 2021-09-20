import numpy as np
import keras
from tensorflow.keras.backend import clear_session
from model import build_sa_bilstm_model
from train_sa_bilstm import train_sa_bilstm
import os
import random
import tensorflow as tf
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
general_path = # <PATH TO CSV FILE CONTAINING THE EMBEDDINGS AND LABELS>
dataset_name = argv[2] # '<NAME>.csv'
path_to_data = general_path + dataset_name 
path_to_model = # <PATH TO PICKLE OBJECT OF THE PRETRAINED MOL2VEC MODEL>


# Coefficient for gradient clipping
cv = 0.3
n_epochs = 1000


# Adapt to your own data
padding_dict = {
    'pcba.csv':488, 
    'hiv.csv':444,
    'srare.csv':244,
    'srmmp.csv':244,
    'sratad5.csv':264
}

weights_dict = {
    'pcba.csv':{0:1, 1:3.81},
    'hiv.csv':{0:1, 1:27.5},
    'srare.csv':{0:1, 1:5.33},
    'srmmp.csv':{0:1, 1:5.42},
    'sratad5.csv':{0:1, 1:27.1}
}

padding = padding_dict.get(dataset_name)
weight = weights_dict.get(dataset_name)

# Number of hidden units for BiLSTM (u)
lstm_hidden = int(argv[3]) # default is 128

# Number of parameters of weight vectors W1 and W2 (for self-attention)
da = int(argv[4])

# Number of attention heads
r = int(argv[5])

# Minibatch size
batch_size = int(argv[6])

# Learning rate 
lr = 0.001 

# ------------------------------------------------------------------------------------
# Make stratified folds for 5-fold CV
list_of_folds = load_data(path_to_data, path_to_model, pad_to=padding)
    
# dimensiones del vector de mol2vec (hardcodear)
d = 300

for idx, fold in enumerate(list_of_folds):
    
    path_to_ckpt = path_to_data[:-4]+'/final/checkpoints/'+str(seed_value)+'/cf'+str(idx)
    savepklto = path_to_data[:-4]+'/final/results/'+str(seed_value)+'/cf'+str(idx)

    Path(savepklto).mkdir(exist_ok=True)
    Path(path_to_ckpt).mkdir(exist_ok=True)

    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
    
    # Load generators
    train_dataset = Mol2vecLoader(train_x, train_y, padding, batch_size)
    val_dataset = Mol2vecLoader(val_x, val_y, padding, batch_size)

    data = (train_dataset, val_dataset)

# ------------------------------------------------------------------------------------
    # Train
    with tf.device('gpu:0'):
        train_sa_bilstm(data, weight, padding, lstm_hidden, d, da, r, lr, path_to_ckpt, cv, n_epochs, savepklto)
        keras.backend.clear_session()
