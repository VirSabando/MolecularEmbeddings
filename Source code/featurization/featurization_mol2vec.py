from resources.features import featurize
from sys import argv

dataset = argv[1]

path_to_model = # <PATH TO Mol2Vec TRAINED MODEL> (.model)
path_to_input = # <PATH TO SMILES DATASET TO BE FEATURIZED USING Mol2Vec> (.smi)
path_to_output = # <PATH TO OUTPUT> (DATAFRAME OF EMBEDDINGS)

featurize(path_to_input, path_to_output, path_to_model, r=2, uncommon='UNK')
