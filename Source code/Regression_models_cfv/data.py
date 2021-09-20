import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math

from tensorflow import keras

SEED = 202042

class DataLoader(keras.utils.Sequence):

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.X, self.y = self.myshuffle(
            self.X, self.y)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def myshuffle(self, arr1, arr2):
        assert arr1.shape[0] == arr2.shape[0]
        idx = np.random.permutation(arr1.shape[0])
        return arr1[idx], arr2[idx]

    def on_epoch_end(self):
        self.X, self.y = self.myshuffle(self.X, self.y)

# Returns 5 folds of Molecular Descriptors
def load_descriptors(data_path):
    df = pd.read_csv(data_path)
    
    # Get column names in the csv file (csv must always be formatted the same way: data,target)
    features = df.iloc[:,:-2].values
    labels = df.iloc[:,-1].values
    
    scaler = StandardScaler()
    scaler.fit(features)
    features= scaler.transform(features)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(features)

    list_of_folds = []
    for train, val in kf.split(features, labels):
        t = (features[train], labels[train])
        v = (features[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds

# Returns 5 folds of ECFP
def load_fingerprints(data_path):
    df = pd.read_csv(data_path)
    
    # Get column names in the csv file (csv must always be formatted the same way: data,target)
    features = df.iloc[:,2].values
    labels = df.iloc[:,1].values
    
    # Necessary to convert fps to arrays of ints:
    features = np.asarray([np.asarray([int(e) for e in list(features[idx])]) for idx,_ in enumerate(features)])

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(features)

    list_of_folds = []
    for train, val in kf.split(features, labels):
        t = (features[train], labels[train])
        v = (features[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds

# Returns 5 folds of MACCS keys
def load_maccs(data_path):
    df = pd.read_csv(data_path)
    
    # Get column names in the csv file (csv must always be formatted the same way: data,target)
    features = df.iloc[:,3].values
    labels = df.iloc[:,1].values
    
    # Necessary to convert fps to arrays of ints:
    features = np.asarray([np.asarray([int(e) for e in list(features[idx])]) for idx,_ in enumerate(features)])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(features)

    list_of_folds = []
    for train, val in kf.split(features, labels):
        t = (features[train], labels[train])
        v = (features[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds

# Returns 5 folds of learned representations
def load_folds(data_path):
    df = pd.read_csv(data_path)
    
    # Get column names in the csv file (csv must always be formatted the same way: data,target)
    features = df.iloc[:,:-1].values
    labels = df.iloc[:,-1].values
    
    # Necessary to convert features to arrays of floats:
    features = np.asarray([np.asarray([float(e) for e in list(features[idx])]) for idx,_ in enumerate(features)])

    scaler = StandardScaler()
    scaler.fit(features)
    features= scaler.transform(features)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(features)

    list_of_folds = []
    for train, val in kf.split(features, labels):
        t = (features[train], labels[train])
        v = (features[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds