import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import random

from data import load_folds
from sklearn.metrics import roc_auc_score
from sys import argv
import sys

from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

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

dataset = argv[1] # name of dataset
version = argv[2] # name of embedding
size = argv[3] # size of embedding

path_to_data = # <PATH TO CSV FILE CONTAINING THE EMBEDDINGS AND LABELS>

# Make stratified folds for 5-fold CV
list_of_folds = load_folds(path_to_data, ismol2vec)

# Adapt to your own data
weights = {
    'srare':5.32,
    'sratad5':27.17,
    'srmmp':5.42,
    'hiv':27.5,
    'pcba':3.18
}
ismol2vec=(version=='mol2vec')


weight = weights.get(dataset)

# C= coefficient for regularization - SVM
c_value = float(argv[4])

# R= max depth for Random forest
r_value = int(argv[5])

# List of ten random seeds for initializing the stochastic classifiers
seed_lst = [17,2131,222,3,342,44,6,7567,980,99]

# Fn to get the results under the eight reported metrics
def get_results(prediction, truth):
    
    # Computos auxiliares para metricas de performance
    true_positive = truth & prediction
    false_positive = ~truth & prediction
    false_negative = truth & ~prediction
    true_negative = ~truth & ~prediction
        
    TP = np.count_nonzero(true_positive)
    TN = np.count_nonzero(true_negative)
    FP = np.count_nonzero(false_positive)
    FN = np.count_nonzero(false_negative)
    num_of_instances = len(truth)
    
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
    tuple_metrics = (sensitivity, specificity, precision, accuracy, balanced_accuracy, f1_score, h_score)

    return (tuple_results, tuple_metrics)

#-----------------------------------------------------------------
# NAIVE BAYES CFV
#-----------------------------------------------------------------
for idx, fold in enumerate(list_of_folds):
    set_seed(seed)
    t, v = fold
    train_x, train_y = t
    val_x, val_y = v

    cnb = ComplementNB()
    cnb.fit(train_x, train_y.ravel())

    # val
    cnb_y_pred = cnb.predict(val_x)

    auc = roc_auc_score(val_y.ravel(), cnb_y_pred)

    prediction = cnb_y_pred ==1
    truth = val_y.ravel() ==1
    t1,t2 = get_results(prediction, truth)
    print("seed " + str(seed) + " - fold " + str(idx) + ' - ' +  dataset + " Naive Bayes val")
    print(t1)
    print(t2)
    print(auc)
    print('--------')
    sys.stdout.flush()

#-----------------------------------------------------------------
# SVM CFV
#-----------------------------------------------------------------
for seed in seed_lst:
    for idx, fold in enumerate(list_of_folds):
        set_seed(seed)
        t, v = fold
        train_x, train_y = t
        val_x, val_y = v
        n_estimators = 10
     
        # condition: if dataset begins with 'sr' (because we tested on 5 datasets, 
        # 3 of them were small in # of compounds and their names start with 'sr')
        if(dataset.startswith('sr')):
            svc = SVC(C=c_value, random_state=seed, class_weight={0:1, 1:weight}, kernel='rbf')
        else:
            svc = BaggingClassifier(SVC(C=c_value, random_state=seed, class_weight={0:1, 1:weight}, kernel='rbf'), 
                                    max_samples=1.0 / n_estimators, n_estimators=n_estimators)
        
        svc.fit(train_x, train_y.ravel())
    
        # val
        svc_y_pred = svc.predict(val_x)

        auc = roc_auc_score(val_y.ravel(), svc_y_pred)

        prediction = svc_y_pred ==1
        truth = val_y.ravel() ==1
        t1,t2 = get_results(prediction, truth)
        print("seed " + str(seed) + " - fold " + str(idx) + ' - ' +  dataset + " svm val")
        print(t1)
        print(t2)
        print(auc)
        print('--------')
        sys.stdout.flush()

#-----------------------------------------------------------------
# RANDOM FOREST CFV
#-----------------------------------------------------------------
for seed in seed_lst:
    for idx, fold in enumerate(list_of_folds):
        set_seed(seed)
        t, v = fold
        train_x, train_y = t
        val_x, val_y = v
    
        rfc = RandomForestClassifier(max_depth=r_value, random_state=seed, class_weight={0:1, 1:weight}, 
                                     n_jobs=-1, n_estimators = 30)
        rfc.fit(train_x, train_y.ravel())
    
        # val
        rfc_y_pred = rfc.predict(val_x)

        auc = roc_auc_score(val_y.ravel(), rfc_y_pred)

        prediction = rfc_y_pred ==1
        truth = val_y.ravel() ==1
        t1,t2 = get_results(prediction, truth)
        print("seed " + str(seed) + " - fold " + str(idx) + ' - ' +  dataset + " rf val")
        print(t1)
        print(t2)
        print(auc)
        print('--------')
        sys.stdout.flush()

