from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import random

def load_and_dirty_data():
    info = load_diabetes()
    X = pd.DataFrame(info["data"], columns=info["feature_names"])
    Y = pd.Series(info["target"])
    X_columns = X.columns
    random_col = np.random.choice(X_columns, size=7, replace=False)
    
    np.random.seed(40)

    for col in random_col:
        list_of_random_row = []
        random_indices = random.randint(5, 25)
    
        for i in range(0 , random_indices):
    
            random_row = random.randint(0, len(X)-1)
    
            if random_row not in list_of_random_row:
                type_of_dirty = ["missing", "outlier", "unnormal-data"]
                type = random.choice(type_of_dirty)
    
                if(type == "missing"):
                    X.loc[random_row, col] = np.nan
                
                if(type == "outlier"):
                    max_col = X[col].max()
                    min_col = X[col].min()
                    X.loc[random_row, col] = (max_col + (max_col-min_col) * 2)
                
                if(type == "unnormal-data"):
                    unnoraml_data = [12321, -1000]
                    X.loc[random_row, col] = random.choice(unnoraml_data)
                
                list_of_random_row.append(random_row)
    
    return X, Y