import tensorflow as tf
from tensorflow.compat.v1.data import Dataset
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.32
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from tensorflow import keras
import numpy as np
from pathlib import Path
import pickle as pkl
import os.path

from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives

from model import build_sa_bilstm_model

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

# Using notation stated in the paper by Zheng et al
# n: number of tokens in each SMILES (after padding). It is THE SAME for all SMILES
# d: dimensionality of each token vector computed using mol2vec. Using the pretrained model, d=300
# u: number of hidden units per LSTM nn. In a Bidirectional LSTM, the final number of hidden units would be 2u (uu)
# da: number of parameters of weight vectors W1 and w2, as stated in the paper
# r: number of heads in the multi-head attention, also called number of attention rows
def train_sa_bilstm(data, cw, n, u, d, da, r, lr, path_to_ckpt, cv, n_epochs, savepklto):
    keras.backend.clear_session()
    
    # Load data generator built from the five folds created previously
    train_dataset, val_dataset = data
        
    # Build and copmpile model
    model = build_sa_bilstm_model(n=n, d=d, u=u, da=da, r=r)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipvalue=cv), loss='binary_crossentropy', metrics=[FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives(), 'AUC'])
    
    # Optional: since these models can take a long time to train,
    # especially with big datasets, you might want to split the training sessions.
    # If there is a checkpoint you need to load, do it here ;)
    # ruta_ckpt =  f'{path_to_ckpt}/model-weighed-sa-bilstm-{n}-{u}-{da}-{r}-{lr}.ckpt.index'
    
    # if os.path.isfile(ruta_ckpt):
    #     model.load_weights(ruta_ckpt[:-6])
    #     print('Loading pretrained model...')
    
    # Summary
    print(model.summary())
    
    earlystop_callback = keras.callbacks.EarlyStopping(min_delta=0.0005, patience=150, restore_best_weights=True)
   
    path_to_checkpoints = path_to_ckpt
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'{path_to_checkpoints}/model-weighed-sa-bilstm-{n}-{u}-{da}-{r}-{lr}.ckpt',
        save_best_only=True, save_weights_only= True)
    
    # Train
    model.fit_generator(train_dataset,
                        epochs=n_epochs,
                        validation_data=val_dataset,
                        callbacks=[earlystop_callback, checkpoint_callback],
                        verbose=0,
                        class_weight=cw
                        )
  
    outcome = model.evaluate(val_dataset)
    L = outcome[0]
    FN = outcome[1]
    FP = outcome[2]
    TN = outcome[3]
    TP = outcome[4]
    AC = outcome[5]
    
    # Optional: save results to a pickle object
    # to_pickle = get_results(TP, TN, FP, FN, AC, L)
    # pkl.dump(to_pickle, open(
    #     Path(savepklto) / f'./weighed_sa_bilstm-{n}-{u}-{da}-{r}-{lr}.p', 'wb'))
