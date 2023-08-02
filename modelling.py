import tabular_data
import pandas as pd
import numpy as np
# from sklearn import datasets, model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def split_data(X, y):
    '''Splits Test, Train and Validation data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, y_train, X_test, y_test, X_validation, y_validation



if __name__ == '__main__':

    # load the data
    
    df = pd.read_csv('clean_data.csv')
    X,y=tabular_data.load_airbnb(df)
    X=X.to_numpy()
    y=y.to_numpy()
    print(type(X))
    print(X.shape)
    print(y.shape)
    X=scale(X)
   
    #split data in train, test and validation sets
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X,y)
    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))

    #select model
    model = SGDRegressor()
    model.fit(X_train, y_train)

    #Estmate predcted labels
    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    #model performance:RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print((f'Train_RMSE:{train_rmse:.2f} \n Validation_RMSE:{ validation_rmse:.2f} \n Test_RMSE:{test_rmse:.2f}'))

    #R^2
    train_score = r2_score(y_train, y_train_pred)
    validation_score = r2_score(y_validation, y_validation_pred)
    test_score = r2_score(y_test, y_test_pred)
    print((f'Train_Score:{train_rmse:.2f} \n Validation_:{ validation_rmse:.2f} \n Test_RMSE:{test_rmse:.2f}'))

