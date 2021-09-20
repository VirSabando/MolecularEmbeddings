from rdkit import Chem
from mol2vec.features import mol2alt_sentence
import numpy as np

# Fn needed to compute Mol2Vec features that will be fed as input data to the SA-BiLSTM model
def mol2vec_features(model, smiles, target, pad_to):
    mollst = [Chem.MolFromSmiles(x) for x in smiles]
    sentences = [mol2alt_sentence(x, 1) for x in mollst]
    
    # compute and return features and labels
    features = np.zeros([len(mollst), pad_to, model.vector_size])
    labels = np.reshape(target, (-1, 1))
    for idx, sentence in enumerate(sentences):
        count = 0
        for word in sentence:
            if count == pad_to:
                break
            try:
                features[idx, count] = model.wv[word]
                count += 1
            except KeyError as e:
                pass
    assert features.shape[0] == labels.shape[0]
    return features, labels

