import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import math

from tensorflow import keras
import numpy as np

# Random seed - guarantees that the folds are always the same on each run of the script
SEED = 202042

# Generator used to feed the data to the FFNN classifier
class DataFeeder(keras.utils.Sequence):
    def __init__(self, smiles_vectors, targets, batch_size):
        self.smiles_vectors = smiles_vectors
        self.targets = targets
        self.batch_size = batch_size

        self.smiles_vectors, self.targets = self.myshuffle(
            self.smiles_vectors, self.targets)

    def __getitem__(self, idx):
        batch_x = self.smiles_vectors[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.smiles_vectors) / self.batch_size)
    
    def myshuffle(self, arr1, arr2):
        assert arr1.shape[0] == arr2.shape[0]
        idx = np.random.permutation(arr1.shape[0])
        return arr1[idx], arr2[idx]

    def on_epoch_end(self):
        self.smiles_vectors, self.targets = self.myshuffle(
            self.smiles_vectors, self.targets)

# Separates the data into 5 stratified folds        
def load_data(data_path, pad_to=240):
    df = pd.read_csv(data_path)
    
    # Get column names in the csv file (csv must always be formatted the same way: data,target)
    tokens= df[df.columns[0:-1]].values
    labels = df[df.columns[-1]].values
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    kf.get_n_splits(tokens)

    list_of_folds = []
    for train, val in kf.split(tokens, labels):
        t = (tokens[train], labels[train])
        v = (tokens[val], labels[val])
        list_of_folds.append((t,v))
    
    return list_of_folds