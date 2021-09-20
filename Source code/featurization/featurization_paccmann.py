import tensorflow as tf
from tensorflow import keras

from model import build_model

import numpy as np
from tensorflow.keras.backend import clear_session
import os
import random
import tensorflow.keras.backend as K
from sys import argv

from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
# ------------------------------------------------------------------------------------
dataset = argv[1]

# The values in these dictionaries correspond to the ones found in the model selection process
params_dict = {
    'srare':{'seed':2131,'n': 240,'t': 263,'u': 50,'da': 20,'activation': 'relu','dc': 0.25,
             'l': 0.005,'positional_encoding': False,'dense_sizes': [150, 50, 10],
             'class_weight': {0:1, 1:5.33},'n_layers':3},
    'srmmp':{'seed':3,'n': 240,'t': 263,'u': 100,'da': 100,'activation': 'relu','dc': 0.25,
             'l': 0.005,'positional_encoding': False,'dense_sizes': [100, 20, 5],
             'class_weight': {0:1, 1:5.42},'n_layers':3},
    'sratad5':{'seed':17,'n': 240,'t': 263,'u': 100,'da': 100,'activation': 'relu','dc': 0.25,
             'l': 0.005,'positional_encoding': False,'dense_sizes': [150, 50, 10],
             'class_weight': {0:1, 1:27.1},'n_layers':3},
    'hiv':{'seed':3,'n': 484,'t': 263,'u': 50,'da': 100,'activation': 'relu','dc': 0.25,
             'l': 0.005,'positional_encoding': False,'dense_sizes': [150, 50, 10],
             'class_weight': {0:1, 1:27.5},'n_layers':3},
    'pcba':{'seed':6,'n': 436,'t': 263,'u': 16,'da': 100,'activation': 'relu','dc': 0.15,
             'l': 0.005,'positional_encoding': False,'dense_sizes': [100, 20, 5],
             'class_weight': {0:1, 1:3.81},'n_layers':3}
}

params = params_dict.get(dataset)

# Sets a random seed globally (numpy, tf and keras). Necessary to obtain fully reproducible results.
# ------------------------------------------------------------------------------------
# Seed value
# Apparently you may use different seed values at each stage
seed_value= params.get('seed')

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)
# ------------------------------------------------------------------------------------
clear_session()

class MySMILES(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename).read().splitlines()[1:]:
            meh = line.split(',')
            meh = np.array([int(float(i)) for i in meh])
            x = meh[:-1].reshape((1,meh[:-1].shape[0]))
            y = np.array(meh[-1]).reshape(1,1)
            yield (x,y)


# Example: paccmann-u-da-act-dc-l-posenc-dense_lst[0].ckpt
# Build model
model = build_model(params)
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-08), loss='binary_crossentropy', metrics=[FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), 'AUC'])
model.summary()

# And load weights
name = 'paccmann-'+str(params.get('u'))+'-'+str(params.get('da'))+'-'+str(params.get('activation'))+'-'+str(params.get('dc'))+'-'+str(params.get('l'))+'-'+str(params.get('positional_encoding'))+'-'+str(params.get('dense_sizes')[0])+'.ckpt'
model.load_weights('/home/vsabando/data_unmolemb/'+dataset+'/paccmann/final/checkpoints/'+str(seed_value)+'/'+name)

# Data generator
nuevas = MySMILES(<PATH>) # PATH TO TOKENIZED SMILES

get_flatten_layer_output = K.function([model.layers[0].input],
                                      [model.layers[4].output])

name_output = #<PATH TO OUTPUT (DATAFRAME OF EMBEDDINGS)
fd_output = open(name_output, 'w')

for (x,y) in iter(nuevas):
    layer_output = get_flatten_layer_output(x)[0]
    layer_output = np.append(layer_output, y, axis=1)[0]
    s = ''
    for i in layer_output:
        s = s+ str(i) + ','
    s = s+'\n'
    fd_output.write(s)

fd_output.close()
