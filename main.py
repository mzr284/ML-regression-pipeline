import pandas as pd
import numpy as np
from src.data import load_and_dirty_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.models import random_forest_model
from src.preprocessing import preprocessing_pipeline
from src.evalutions import get_r2_CV_mean, get_r2_score, get_MAE_score
import matplotlib.pyplot as plt

def main():
    X, Y = load_and_dirty_data()

    X_train, X_test , Y_train, Y_test = train_test_split(X, Y,
                                                        random_state=42,
                                                        test_size=0.2)
    
    random_forest_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline(X_train.columns)),
        ("rfc_model", random_forest_model())
    ])
    random_forest_pipeline.fit(X_train, Y_train)
    
    print("R2 score in cross val scores >>>  ", get_r2_CV_mean(X_train, Y_train, random_forest_pipeline))
    print("R2 score on test >>>  ", get_r2_score(X_test, Y_test, random_forest_pipeline))
    print("MAE score on test >>>  ", get_MAE_score(X_test, Y_test, random_forest_pipeline))


if __name__ == "__main__":
    main()