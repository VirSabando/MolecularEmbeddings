import pandas as pd
import pickle as pkl
import re
from sys import argv

token_to_idx = pkl.load(open('token_to_idx.p','rb'))
dataset = argv[1] # <PATH TO DATASET>

df = pd.read_csv(dataset)
smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(x))for x in df['SMILES']]
target = df['Value'].values

padding_dict = {
    'ESOL':97,
    'FreeSolv':53,
    'Lipophilicity':205
}

pad = padding_dict.get(dataset)
pad_token = ['<PAD>']

# Tokenize the SMILES sequences according to the paper
smiles = [' '.join(list(e)) for e in smiles]
smiles = [re.sub(r'\[.*?\]', lambda x: ''.join(x.group(0).split()), e) for e in smiles]

# Split sequence
smiles = [e.split(' ') for e in smiles]

# Add padding
padded = [l + pad_token*(pad-len(l)) for l in smiles]

# Retrieve token indexes from dict
tokenizados = [[token_to_idx.get(e) for e in l] for l in padded]

# Shape into dataframe
salida = pd.DataFrame(tokenizados)
salida['TARGET']=target
df_name = dataset[:-4]+'_tokenized.csv' # Adapt accordingly
salida.to_csv(df_name, index=False)
