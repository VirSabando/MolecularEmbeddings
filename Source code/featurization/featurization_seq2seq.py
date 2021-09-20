
# build seq2seq model
from train_py import build_hparams

# train seq2seq model
from decode_py import sample_decode, fp_decode


modelo = # <PATH TO Seq2Seq TRAINED MODEL (.model)
data = # <PATH TO SMILES DATASET TO BE FEATURIZED USING SMILESVec> (.smi)
vocab = # <PATH TO VOCAB FILE
output = # PATH TO OUTPUT (DATAFRAME OF EMBEDDINGS)

build_hparams(modelo)
fp_decode(modelo, data, vocab, output)

