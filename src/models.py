from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet


####  after search on hyperparameters and choose the best them for models

def decision_tree_model():    
    return DecisionTreeRegressor(min_samples_split=5,
                                        min_samples_leaf=5,
                                        max_features=0.7,
                                        max_depth=3,
                                        ccp_alpha=0.01)

def random_forest_model():
    return RandomForestRegressor(n_estimators=750,
                                    min_samples_split=12,
                                    min_samples_leaf=8,
                                    max_features=0.5,
                                    max_depth=19,
                                    ccp_alpha=0.03571428571428572,
                                    bootstrap=True)

def linear_ridge_model():
    return Ridge(fit_intercept=True,
                    alpha=46.41588833612782)

def linear_lasso_model():
    return Lasso(fit_intercept=True,
                    alpha=2.848035868435802)

def linear_elasticnet_model():
    return ElasticNet(max_iter=5000,
                         l1_ratio=1.0,
                         alpha=2.329951810515372)