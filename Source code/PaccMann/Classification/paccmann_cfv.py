import tensorflow as tf
from tensorflow.keras.backend import clear_session
from paccmann_data import load_data, DataFeeder
#import keras 

from model import build_model
from train_paccmann import train_paccmann

import os
import random
import numpy as np
from pathlib import Path
from sys import argv

# Sets a random seed globally (numpy, tf and keras). Necessary to obtain fully reproducible results.
# ------------------------------------------------------------------------------------
# Seed value
# Apparently you may use different seed values at each stage
seed_value= int(argv[2])

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
dataset_name = argv[1] 
path_to_data = general_path + dataset_name 

# No. of maximum epochs as stated in the reference paper
n_epochs = 500000

# Adapt to your own data
padding_dict = {
    'pcba':436,
    'hiv':484,
    'srare':240,
    'srmmp':240,
    'sratad5':240
}

weights_dict = {
    'pcba':{0:1, 1:3.81},
    'hiv':{0:1, 1:27.5},
    'srare':{0:1, 1:5.33},
    'srmmp':{0:1, 1:5.42},
    'sratad5':{0:1, 1:27.1}
}

padding = padding_dict.get(dataset_name[:-15])
weight = weights_dict.get(dataset_name[:-15])

t = 263 # vocabulary size - computed after tokenizing the data
u = int(argv[3])
da = int(argv[4])
act = 'relu'
dc = float(argv[5])
l = float(argv[6])
posenc = False
dense_sizes = [int(e) for e in argv[7].split(',')]
n_layers = 3
batch_size = int(argv[8])

params = {'n': padding,'t': t,'u': u,'da': da,'activation': act,'dc': dc,
     'l': l,'positional_encoding': posenc,'dense_sizes': dense_sizes,
     'class_weight': weight,'n_layers':n_layers}

# Learning rate 
lr = 0.001 

# ------------------------------------------------------------------------------------

# Make stratified folds for 5-fold CV
list_of_folds = load_data(path_to_data, pad_to=padding)


for idx, fold in enumerate(list_of_folds):
    
    # Modify accordingly
    path_to_ckpt = path_to_data[:-15]+'/paccmann/final/checkpoints/'+str(seed_value)+'/cf'+str(idx)
    savepklto = path_to_data[:-15]+'/paccmann/final/results/'+str(seed_value)+'/cf'+str(idx)

    Path(savepklto).mkdir(exist_ok=True)
    Path(path_to_ckpt).mkdir(exist_ok=True)

    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
    
    train_dataset = DataFeeder(train_x, train_y, batch_size)
    val_dataset = DataFeeder(val_x, val_y, batch_size)

    data = (train_dataset, val_dataset)

# ------------------------------------------------------------------------------------
    # Train
    with tf.device('gpu:0'):
        train_paccmann(data, params, lr, n_epochs, path_to_ckpt, savepklto)
        tf.keras.backend.clear_session()