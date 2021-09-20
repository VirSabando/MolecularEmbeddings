import tensorflow as tf
from tensorflow.compat.v1.data import Dataset
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.32
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from tensorflow import keras
import numpy as np
from pathlib import Path
import pickle as pkl

from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from model import build_model

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

# Function to train PaccMann model
def train_paccmann(data, params, lr, n_epochs, path_to_ckpt, savepklto):
    tf.keras.backend.clear_session()
    
    # Load data generator built from the five folds created previously
    train_dataset, val_dataset = data
        
    # Build and compile model
    model = build_model(params)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, epsilon=1e-08), loss='binary_crossentropy', metrics=[FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), 'AUC'])
        
    print(model.summary())
    
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.0005, patience=1000, restore_best_weights=True, verbose=2)
   
    path_to_checkpoints = path_to_ckpt
    
    u = params.get('u') # number of hidden units == embedding size
    da = params.get('da') # attention depth
    act = params.get('activation') # activation fn for dense layers
    dc = params.get('dc') # dropout coefficient
    l2_lambda = params.get('l') # lambda for l2 regularization
    pos_enc = params.get('positional_encoding') # positional encoding
    dense_sizes_lst = params.get('dense_sizes') # No. of hidden units per dense layer in the classifier 
                                                                    # Assume always 3 layers
    class_weight = params.get('class_weight')
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'{path_to_checkpoints}/paccmann-{u}-{da}-{act}-{dc}-{l2_lambda}-{pos_enc}-{dense_sizes_lst[0]}.ckpt',
        save_best_only=True, save_weights_only= True)
    
    # Train
    model.fit_generator(train_dataset,
                        epochs=n_epochs,
                        validation_data=val_dataset,
                        callbacks=[earlystop_callback, checkpoint_callback],
                        verbose=0,
                        class_weight=class_weight
                        )
  
    outcome = model.evaluate(val_dataset)
    L = outcome[0]
    FN = outcome[1]
    FP = outcome[2]
    TN = outcome[3]
    TP = outcome[4]
    AC = outcome[5]
    
    # Optional: save results to a pickle object associated to the checkpoint
    # to_pickle = get_results(TP, TN, FP, FN, AC, L)
    # pkl.dump(to_pickle, open(
    #     Path(savepklto) / f'./paccmann-{u}-{da}-{act}-{dc}-{l2_lambda}-{pos_enc}-{dense_sizes_lst[0]}.p', 'wb'))
