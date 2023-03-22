from tabular_data import clean_tabular_data
import pandas as pd
import numpy as np
# from sklearn import datasets, model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def split_data(X, y):
    '''Splits Test, Train and Validation data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, y_train, X_test, y_test, X_validation, y_validation

if __name__ == '__main__':

    # load the data
    tabular_data=clean_tabular_data()
    df = pd.read_csv('clean_data.csv')
    X,y=tabular_data.load_airbnb(df)
    X=X.to_numpy()
    y=y.to_numpy()
    print(type(X))
    print(X.shape)
    print(y.shape)

   
    #split data in train, test and validation sets
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X,y)
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))

    # # select model
    model = SGDRegressor()
    model.fit(X_train, y_train)


    y_train_pred = model.predict(X_train)
    # y_validation_pred = model.predict(X_validation)
    # y_test_pred = model.predict(X_test)
    

