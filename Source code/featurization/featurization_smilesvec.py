import gensim
import os,sys
from nltk import ngrams
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

class MySMILES(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename).read().splitlines():
            eightgrams = [''.join(e) for e in ngrams(line, 8)]
            yield eightgrams
            
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        model.save(output_path)
        self.epoch += 1

dataset = sys.argv[1]
size = 300 # 100 or 300

test_file = # <PATH TO SMILES DATASET TO BE FEATURIZED USING SMILESVec> (.smi)
fname2 = get_tmpfile(<PATH>) # <PATH TO SMILESVec TRAINED MODEL> (.model)
modelo = Word2Vec.load(fname2)
nuevas = MySMILES(test_file)

vocab = modelo.wv.vocab

salida = # <PATH TO OUTPUT> (DATAFRAME OF EMBEDDINGS)

L = []
for elem in iter(nuevas):
    emb = np.zeros(size)
    tokens = 0
    for token in elem:
        if token in vocab:
            tokens = tokens + 1
            emb = emb + np.array(modelo.wv[token])
    if(tokens!=0):
        emb = emb / tokens
    L.append(emb)
    
names = ['data_'+str(e) for e in range(len(L))]
df = pd.DataFrame.from_dict(dict(zip(names, L)))            
df.T.to_csv(salida, index=False)
            