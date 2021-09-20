import pandas as pd
import pickle as pkl
import re
from sys import argv

token_to_idx = pkl.load(open('token_to_idx.p','rb'))
dataset = argv[1] 

df = pd.read_csv(dataset)
smiles = df['SMILES'].values
target = df.iloc[:,1].values

padding_dict = {
    'pcba.csv':436, 
    'hiv.csv':484,
    'srare.csv':240,
    'srmmp.csv':240,
    'sratad5.csv':240
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
