import pandas as pd
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from utils import mol2vec_features
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math

from tensorflow import keras

from mol2vec.features import mol2alt_sentence
import numpy as np

# Random seed - guarantees that the folds are always the same on each run of the script
SEED = 202042

# Generator used to feed the data to the FFNN classifier
class Mol2vecLoader(keras.utils.Sequence):
    model = word2vec.Word2Vec.load('/home/vsabando/zheng/mol2vec_models/model_300dim.pkl')

    def __init__(self, smiles_vectors, targets, pad_to, batch_size):
        self.smiles_vectors = smiles_vectors
        self.targets = targets
        self.pad_to = pad_to
        self.batch_size = batch_size

        self.smiles_vectors, self.targets = self.myshuffle(
            self.smiles_vectors, self.targets)

    def __getitem__(self, idx):
        batch_x = self.smiles_vectors[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        features, batch_y = mol2vec_features(self.model, batch_x, batch_y, self.pad_to)
        # Standardize data
        scaler = StandardScaler()
        batch_x = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
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