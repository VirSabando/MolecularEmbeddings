from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MACCSkeys, MolFromSmiles
from sys import argv
import pandas as pd

dataset = argv[1] # <NAME_OF_DATASET>.csv
output_dataset = dataset[:-4] + '_fps_keys.csv'
mi_dataset = pd.read_csv(dataset)
smiles = mi_dataset['SMILES'].values

fps = [GetMorganFingerprintAsBitVect(MolFromSmiles(e),2,nBits=1024).ToBitString() for e in smiles]
maccs = [MACCSkeys.GenMACCSKeys(MolFromSmiles(e)).ToBitString() for e in smiles]

mi_dataset['Fingerprints'] = fps
mi_dataset['MACCS_Keys'] = maccs


mi_dataset.to_csv(output_dataset,index=False)
