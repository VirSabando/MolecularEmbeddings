import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model

from sklearn.metrics import roc_auc_score
import numpy as np
import os
from sys import argv
import random

from data import load_folds, DataLoader
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives

# Sets a random seed globally (numpy, tf and keras). Necessary to obtain fully reproducible results.
def set_seed(seed):
    #-----------------------------------------------------------------
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= seed

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    #-----------------------------------------------------------------

#-------------------------------------------------------------------------
#                              METRICS
#-------------------------------------------------------------------------
# Fn to get the results under the eight reported metrics
def get_results(TP, TN, FP, FN, AC, L):
    
    num_of_instances = TP + TN +FP + FN
    print('num of instances = '+ str(num_of_instances))
    
    accuracy = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    f1_score = 0
    h_score = 0
    
    # Metricas de performance
    if (num_of_instances > 0):
        accuracy = (TP + TN) / num_of_instances
    if ((TP+FN) > 0):
        sensitivity = TP / (TP + FN)
    if ((TN+FP) > 0):
        specificity = TN / (TN + FP)
    if ((TP+FP) > 0):
        precision = TP / (TP + FP)
    if ((2*TP + FP + FN) > 0):
        f1_score = 2*TP / (2*TP + FP + FN)
    balanced_accuracy = (sensitivity + specificity)/2
    if ((sensitivity + specificity) > 0):
        h_score = 2*(sensitivity * specificity)/(sensitivity + specificity)
    
    tuple_results = (TP,TN,FP,FN)
    tuple_metrics = (sensitivity, specificity, precision, accuracy, balanced_accuracy, f1_score, h_score, AC)
    loss = L

    return (tuple_results, tuple_metrics, loss)


#-------------------------------------------------------------------------
#                              DATA LOADING
#-------------------------------------------------------------------------
dataset = argv[1] # name
version = argv[2] # type of embedding
size = argv[3] # 100 or 300 or 384

path_to_data = # <PATH TO CSV FILE CONTAINING THE EMBEDDINGS AND LABELS>

# Make stratified folds for 5-fold CV
list_of_folds = load_folds(path_to_data, ismol2vec)

# activation function
act = argv[4]

#minibatch size
mbs = int(argv[5])

# lambda value for l2 regularization
ln = float(argv[6])

# patience for early stopping
es_p = int(argv[7])

# whether to use or not weighed cost function: 'weighed' if YES, 'non-weighed' if NO
w = argv[8]

# list of seeds used in the experiments = [17,2131,222,3,342,44,6,7567,980,99]
seed = int(argv[9])

# Adapt to your own data
weights = {
    'srare':{0:1, 1:5.32},
    'sratad5':{0:1, 1:27.17},
    'srmmp':{0:1, 1:5.42},
    'hiv':{0:1, 1:27.5},
    'pcba':{0:1, 1:3.18}
}
ismol2vec=(version=='mol2vec')

if(w.startswith('w')):
    weight = weights.get(dataset)
else:
    weight = None
    
#-------------------------------------------------------------------------
#                              FFNN CONSTRUCTION
#-------------------------------------------------------------------------
                   
# hidden units
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 10
n_outputs = 1

#adam optimizador
l_rate = 0.00001

#dropout
prob_h1 = 0.25
prob_h2 = 0.1
prob_h3 = 0.05

n_epochs = np.iinfo(np.int32).max

#batch normalization
momentum_batch_norm = 0.95

# Fn to build FFNN model based on the previous params
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

    salida = Dense(n_outputs,activation='sigmoid')(c3)
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
es = EarlyStopping(monitor='val_loss', min_delta=min_delta_val, patience=es_p, verbose=0, mode='min')

for idx, fold in enumerate(list_of_folds):

    # Set the same seed for each fold!
    set_seed(seed)
    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
                   
    input_shape = train_x.shape[1:]

    # Load data generator built from the five folds created previously
    train_dataset = DataLoader(train_x, train_y, mbs)
    val_dataset = DataLoader(val_x, val_y, mbs)
    
    # Build
    model = build_model(input_shape, mbs, ln, act)
                
    # Compile
    model.compile(optimizer=adam_opt, loss='binary_crossentropy', 
                  metrics=[FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), tf.keras.metrics.AUC()])

    # Train
    training_data = model.fit_generator(train_dataset,
                                        epochs=n_epochs,
                                        validation_data=val_dataset,
                                        callbacks=[es],
                                        verbose=0,
                                        use_multiprocessing=False,
                                        class_weight = weight)
                

    # Final evaluation on validation fold
    outcome = model.evaluate(val_dataset)
    L = outcome[0]
    FN = outcome[1]
    FP = outcome[2]
    TN = outcome[3]
    TP = outcome[4]
    AC = outcome[5]
    
    t1,t2,l = get_results(TP, TN, FP, FN, AC, L)
    print("seed " + str(seed) + " - fold " +str(idx)+"results on val - "+ dataset)
    print(t1)
    print(t2)
    print(l)
    print('--------')
                
    # Clear session
    clear_session()
