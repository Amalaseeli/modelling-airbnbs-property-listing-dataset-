from neural_network import AirbnbNightlyPriceRegressionDataset,find_best_nn
from modelling import split_data
import torch
from modelling_regression import evaluate_all_models,find_best_model,model_list, parameter_grid_list
from modelling_classification import evaluate_all_cls_models,find_best_cls_model,model_list, parameter_grid_list
from torch.utils.data import DataLoader,random_split
import pandas as pd
from modelling_classification import tune_classification_model_hyperparameters
from hyperparameter import tune_regression_model_hyperparameters
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 
def load_spilit_data(df):
    label=df["bedrooms"]
    df['Category']=label_encoder.fit_transform(df["Category"])
# print(df['Category'].unique())
    df = df.drop(columns=["Unnamed: 19"])
    df = df.drop(columns=["Unnamed: 0"])
    feature=df.drop(columns=["ID","Title","Description","Amenities", "Location", "url","bedrooms"])
    return feature,label

df = pd.read_csv('clean_data.csv')
feature,label=load_spilit_data(df)
              
X_train, y_train, X_test, y_test, X_validation, y_validation=split_data(feature,label)

if __name__=='__main__':
    model_type="regression"

    dataset=AirbnbNightlyPriceRegressionDataset(feature,label)
    train_size=int(0.7*len(dataset))
    val_size=int(0.1*len(dataset))
    test_size=len(dataset)-(train_size+val_size)
    dataset_train,dataset_val,dataset_test=random_split(dataset=dataset,lengths=[train_size,val_size,test_size])
#Define Dataloader
    batch_size=32
    dataloader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test=DataLoader(dataset_test,batch_size=batch_size,shuffle=False)
    dataloader_val=DataLoader(dataset_val,batch_size=batch_size,shuffle=False)
   
    if model_type == "regression":
        for i in range(len(model_list)):

            model = model_list[i]               
            parameter_grid = parameter_grid_list[i]
        folder = "models/regression/neuralnetworks"
        evaluate_all_models(model_list, parameter_grid_list,X_train, y_train, X_test, y_test, X_validation, y_validation,folder)

        best_reg_model, parameters, performance_metric = find_best_model(model_list, folder)

    if model_type=="classification":
        best_model, best_params, performance_metric = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid)
        folder = "models/classification/neuralnetworks"
        best_cls_model, parameters, performance_metric = find_best_cls_model(model_list, folder)  

    find_best_nn(X_train,y_train,X_validation, y_validation,X_test,y_test,dataloader_train)
    print(f"The best regression model is {best_reg_model}")
    print(f"Performance metric for the best model is {performance_metric}")
   

