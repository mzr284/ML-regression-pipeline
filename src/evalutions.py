from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd

def get_r2_CV_mean(X:pd.DataFrame, Y:pd.Series, estimatior):
    cross_r2_score = cross_val_score(estimator=estimatior,
                                     X=X, y=Y, cv=5,
                                     scoring="r2")
    return cross_r2_score.mean()

def get_r2_score(X:pd.DataFrame, Y:pd.Series, estimatior):
    return r2_score(y_true=Y,
                    y_pred=estimatior.predict(X))

def get_MAE_score(X:pd.DataFrame, Y:pd.Series, estimatior):
    return mean_absolute_error(y_true=Y,
                               y_pred=estimatior.predict(X))