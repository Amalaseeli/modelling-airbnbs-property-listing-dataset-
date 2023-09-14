import os
import json
import joblib
import pandas as pd
import tabular_data
import modelling
import numpy as np
import hyperparameter
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def make_predictions(model, X_train, X_test, X_validation):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_validation_pred = model.predict(X_validation)
    return y_train_pred, y_test_pred, y_validation_pred

def get_metrics_for_classification_model(y_train, y_train_pred, y_test, y_test_pred, y_validation, y_validation_pred):
    performance_metric = {}
    set_names = ['train', 'test', 'validation']
    sets = [(y_train, y_train_pred), (y_test, y_test_pred), (y_validation, y_validation_pred)]

    for i in range(len(sets)):
        y, y_hat = sets[i]
        
        accuracy    = accuracy_score(y, y_hat)
        precision   = precision_score(y, y_hat, average="macro")
        recall      = recall_score(y, y_hat, average="macro")
        f1          = f1_score(y, y_hat, average="macro")

        performance_metric[f"accuracy_{set_names[i]}"]  = accuracy
        performance_metric[f"precision_{set_names[i]}"] = precision
        performance_metric[f"recall_{set_names[i]}"]    = recall
        performance_metric[f"f1_{set_names[i]}"]        = f1

    return performance_metric

if __name__=='__main__':
    file='clean_tabular_data.csv'
    df=pd.read_csv(file)
    X,y=tabular_data.load_airbnb(df)
    X_train,y_train,X_test,y_test,X_validation,y_validation=modelling.split_data(X,y)

    #Select model
    model=LogisticRegression(max_iter=200)

    #fitting model
    model.fit(X_train,y_train)

    # estimate predicted labels
    y_train_pred, y_test_pred, y_validation_pred = make_predictions(model, X_train, X_test, X_validation)
    performance_metric=get_metrics_for_classification_model(y_train,y_train_pred,y_test,y_test_pred,y_validation,y_validation_pred)
    

