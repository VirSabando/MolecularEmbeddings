import pandas as pd
import numpy as np
import os
import random
import sys
from sys import argv

from data import load_descriptors, load_fingerprints, load_maccs, load_folds

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#-----------------------------------------------------------------
def set_seed(s):
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= s

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
#-----------------------------------------------------------------

dataset = argv[2] # <ABSOLUTE_PATH_TO_CSV_FILE.csv>
dataset_name = argv[1] # ESOL or FreeSolv or Lipophilicity
mol_rep = argv[3] # D for descriptors, F for ECFP, M for MACCS keys, E for embeddings

# Retrieve folds of data
if mol_rep == 'D':
    list_of_folds = load_descriptors(dataset)
elif mol_rep == 'F':
    list_of_folds = load_fingerprints(dataset)
elif mol_rep == 'M':
    list_of_folds = load_maccs(dataset)
elif mol_rep == 'E':
    list_of_folds = load_folds(dataset)
else:
    print('ERROR: invalid mol_rep')

seed_lst = [17,2131,222,3,342,44,6,7567,980,99]

#-----------------------------------------------------------------
# Regression metrics
def get_results(y_true, y_pred):
    
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    MSE = mean_squared_error(y_true, y_pred, squared=True)
    MAE = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)

    return (RMSE, MSE, MAE, R2)

#-----------------------------------------------------------------
# Ridge Regression
#-----------------------------------------------------------------

for idx, fold in enumerate(list_of_folds):
    seed = 42
    set_seed(seed)
    t, v = fold
    train_x, train_y = t
    val_x, val_y = v
    
    r_reg = Ridge(alpha=1, random_state=seed, max_iter= 10000, tol=0.001)
    r_reg.fit(train_x, train_y)
        
    # val
    y_pred = r_reg.predict(val_x)

    rmse, mse, mae, r2 = get_results(val_y, y_pred)
    print("fold " + str(idx) + " - " + dataset_name + " - "+ mol_rep + " ridge val, " + str(rmse) +' , '+ str(mse) +' , '+ str(mae) +' , '+ str(r2))

print('--------')
sys.stdout.flush()


#-----------------------------------------------------------------
# Gradient Boosting Regression
#-----------------------------------------------------------------

for seed in seed_lst:
    for idx, fold in enumerate(list_of_folds):
        set_seed(seed)
        t, v = fold
        train_x, train_y = t
        val_x, val_y = v
    
        # Best params found through grid search
        params = {'n_estimators': 500,
                  'max_depth': 3,
                  'min_samples_split': 2,
                  'learning_rate': 0.1,
                  'loss': 'ls',
                  'random_state': seed}

        gb_reg = GradientBoostingRegressor(**params)
        gb_reg.fit(train_x, train_y)
        
        # val
        y_pred = gb_reg.predict(val_x)

        rmse, mse, mae, r2 = get_results(val_y, y_pred)
        print("seed "+str(seed)+" -  fold " + str(idx) + " - " + dataset_name + " - "+ mol_rep + " GBR val, " + str(rmse) +' , '+ str(mse) +' , '+ str(mae) +' , '+ str(r2))

    print('--------')
    sys.stdout.flush()
