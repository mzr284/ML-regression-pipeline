from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer

def wensorize(X:pd.DataFrame, columns:list):
    X = X.copy()
    
    for col in columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3- Q1
        
        outliers_over_limitation = X[col] > Q3 + 1.5 * IQR
        outliers_under_limitation = X[col] < Q1 - 1.5 * IQR
        X.loc[outliers_over_limitation, col] = Q3 + 1.5 * IQR
        X.loc[outliers_under_limitation, col] = Q1 - 1.5 * IQR
    
    return X

class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, columns, lower=1.5, upper=1.5):
        self.columns = columns
        self.upper = upper
        self.lower = lower
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
    
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3- Q1
        
            outliers_over_limitation = X[col] > Q3 + self.upper * IQR
            outliers_under_limitation = X[col] < Q1 - self.lower * IQR
            X.loc[outliers_over_limitation, col] = Q3 + self.upper * IQR
            X.loc[outliers_under_limitation, col] = Q1 - self.lower * IQR
        
        return X
    
def add_features(X:pd.DataFrame):
    X = X.copy()
    X["ratio_s1"] = X["s1"] / X["bmi"]
    X["ratio_age/s5"] = X["age"] / X["s5"]
    
    return X

def preprocessing_pipeline(columns):
    winsorizer = Winsorizer(columns, lower=2.5, upper=1.5)
    return Pipeline([
        ("winsorize", winsorizer),
        ("imputer", SimpleImputer(strategy="median")),
        ("power-transformer", PowerTransformer()),
        ("standard-scaling", StandardScaler())
    ])

def get_features_after_processing(X:pd.DataFrame):
    return pd.DataFrame(preprocessing_pipeline(X).fit_transform(X.copy()),
                        columns=X.columns)