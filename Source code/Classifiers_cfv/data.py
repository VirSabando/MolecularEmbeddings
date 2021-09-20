import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from tensorflow import keras
import numpy as np
import math

# Random seed - guarantees that the folds are always the same on each run of the script
SEED = 202042

# Generator used to feed the data to the FFNN classifier
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
        self.X, self.y = self.myshuffle(
            self.X, self.y)

# Separates the data into 5 stratified folds
def load_folds(data_path, ismol2vec):
    df = pd.read_csv(data_path)
    
    if ismol2vec:
        # Get column names in the csv file (csv must always be formatted the same way: data,target)
        features = df.iloc[:,1:-1].values
        labels = df.iloc[:,-1].values
    else:
        # Get column names in the csv file (csv must always be formatted the same way: data,target)
        features = df.iloc[:,:-1].values
        labels = df.iloc[:,-1].values
    
     # Necessary to convert fps to arrays of ints:
    features = np.asarray([np.asarray([float(e) for e in list(features[idx])]) for idx,_ in enumerate(features)])
    
    # scale
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(features)

    list_of_folds = []
    for train, val in kf.split(features, labels):
        t = (features[train], labels[train])
        v = (features[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds

