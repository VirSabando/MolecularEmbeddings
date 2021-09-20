from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd

# Create Mordred descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

df = pd.read_csv(<PATH>) # <PATH TO DATASET>
df.dropna(inplace=True)
smiles = df['SMILES'].values

# calculate multiple molecule
mols = []
for smi in smiles:
    print(smi)
    mols.append(Chem.MolFromSmiles(smi))
aux = calc.pandas(mols)

aux = aux.apply(pd.to_numeric, errors='coerce')
aux.drop(aux.columns[aux.isna().mean() > 0.05], inplace=True, axis=1)
aux = aux.fillna(aux.mean())
aux.to_csv(<PATH> ,index=False) # <PATH TO OUTPUT> (DATAFRAME OF MOLECULAR DESCRIPTORS)
