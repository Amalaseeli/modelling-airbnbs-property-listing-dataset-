from typing import Any
import numpy as np
import os
import joblib
from pathlib import Path
import torch
import time
import json
import yaml
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tabular_data
import pandas as pd
import modelling
import itertools
import torch.nn.functional as F
import sys


#Dataloader class 
class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self,feature,label):
        super().__init__()
        self.X=torch.tensor(feature.values, dtype=torch.float32)
        self.y=torch.tensor(label.values, dtype=torch.float32)

    def __getitem__(self, index):
        feature=self.X[index]  
        label=self.y[index]
        return (feature,label)
    
    def __len__(self):
        return len(self.X)
    
#model class
class FeedForward(torch.nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        width=config['hidden_layer_width']
        depth=config['depth']
        layers=[]
        layers.append(torch.nn.Linear(11,width))
        for hidden_layer in range(depth-1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        self.layers=torch.nn.Sequential(*layers)


    def forward(self, features) -> Any:
        return self.layers(features)
      
def train(model,config,dataloader,epochs:int):
     '''The functions trains the model and adds loss to TensorBoard'''
     optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
     writer=SummaryWriter()
    #  batch_idx=0

     for epoch in range(epochs):
          for batch_idx,batch in enumerate(dataloader):
               features,labels=batch # tensors: features:[16,11], labels:[16]
               features=features.type(torch.float32)
               labels=torch.unsqueeze(labels,1) # labels tensor size now is [16, 1]

               #forward pass
               prediction=model(features)
               loss=F.mse_loss(prediction,labels)
               
               loss.backward()
               
               #Gradient optimisation
               optimizer.step()
               optimizer.zero_grad()
               writer.add_scalar("loss",loss.item(),batch_idx)
            #    batch_idx+=1


def get_rmse_r2_score(model,feature,label):
    feature=torch.tensor(feature.values).type(torch.float32)   
    label=torch.tensor(label.values).type(torch.float32)
    label=torch.unsqueeze(label,1)
    prediction=model(feature)
    rmse_loss=torch.sqrt(F.mse_loss(prediction,label.float()))
    r2_score = 1 - rmse_loss / torch.var(label.float())
    return rmse_loss, r2_score

def get_performance_of_matric(model,epochs,training_duration,X_train,y_train,X_val,y_val,X_test,y_test):
    metrics_dict={'training_duration': training_duration}
    number_of_predictions = epochs * len(X_train)
    inference_latency = training_duration / number_of_predictions

    train_RMSE,train_r2_score=get_rmse_r2_score(model,X_train,y_train)
    val_RMSE,val_r2_score=get_rmse_r2_score(model,X_val,y_val)
    test_RMSE,test_r2_score=get_rmse_r2_score(model,X_test,y_test)

    print(f"Train RMSE: {train_RMSE.item():.2f} | Train R2: {train_r2_score.item():.2f}")
    print(f"Validation RMSE: {val_RMSE.item():.2f} | Validation R2: {val_r2_score.item():.2f}")
    print(f"Test RMSE: {test_RMSE.item():.2f} | Test R2: {test_r2_score.item():.2f}")

    metrics_dict['RMSE_loss_train']=train_RMSE.item()
    metrics_dict['RMSE_loss_val']=val_RMSE.item()
    metrics_dict['RMSE_loss_test']=test_RMSE.item()
    metrics_dict['R_squared_train']=train_r2_score.item()
    metrics_dict['R_squared_val']=val_r2_score.item()
    metrics_dict['R_squared_test']=test_r2_score.item()

    
    metrics_dict['inference_latency']=inference_latency

    return metrics_dict

def get_nn_config():
    with open('nn_config.yaml','r') as file:
        data=yaml.safe_load(file)
        #print(data)
        return data
    
def save_model(model_type:str, model_name:str, model,hyperparam_dict,metrics_dict):
    '''detects whether the model is a PyTorch module'''
    if not isinstance(model,torch.nn.Module):
        print("The Model is not a Pytorch module")
    else:
        model_folder=Path(f'models/{model_type}/{model_name}')
        current_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        model_folder = Path(f"{model_folder}/{current_date_time}")
        model_folder.mkdir(parents=True, exist_ok=True)
        #os.makedirs(folder,exist_ok=True)
        #torch.save(model.state_dict(), f"{folder}/model.pt")
        model_path=model_folder.joinpath('model.pt')
        joblib.dump(model,model_path)
        hyperparameter_path=model_folder.joinpath('hyperparameters.json')
        with open(hyperparameter_path, 'w') as fp:
            json.dump(hyperparam_dict, fp)
            metrics_path=model_folder.joinpath('metrics.json')
        with open( metrics_path, 'w') as fp:
            json.dump( metrics_dict, fp)
        #with open(f"{folder}/hyperparameters.json", 'w') as fp:
         #   json.dump(hyperparam_dict, fp)
        # Saves performance metrics
        #with open(f"{folder}/metrics.json", 'w') as fp:
         #   json.dump(metrics_dict, fp)
            
def generate_nn_configs():
    ''' Finds all possible combinations of hyperparameters
    and returns a list of dictionaries '''
    param_space = {
    'optimizer': ['Adam', 'AdamW'],
    'learning_rate': [0.01, 0.001,0.0001],
    'hidden_layer_width': [32, 64],
    'depth': [4, 8]
    }
    # Finds all combindations of hyperparameters    
    keys, values = zip(*param_space.items())
    param_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_dict_list

if __name__=='__main__':
    file='clean_tabular_data.csv'
    df=pd.read_csv(file)
    X,y=tabular_data.load_airbnb(df)

    X_train,y_train,X_test,y_test,X_val,y_val=modelling.split_data(X,y)
    
    #Define Datasets
    dataset_train=AirbnbNightlyPriceRegressionDataset(X_train,y_train)
    dataset_test =AirbnbNightlyPriceRegressionDataset(X_test,y_test)
    dataset_val = AirbnbNightlyPriceRegressionDataset(X_val, y_val)

    #Define Dataloader
    batch_size=64
    dataloader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test=DataLoader(dataset_test,batch_size=batch_size,shuffle=False)
    dataloader_val=DataLoader(dataset_val,batch_size=batch_size,shuffle=False)
   
    config=get_nn_config()
    model=FeedForward(config)
    epochs= 100
    model_folder="models/neural_networks/regression"

    parameter_dictionary=generate_nn_configs()
    best_val_loss=np.inf
    for idx,parameter_dict in enumerate(parameter_dictionary):
        model=FeedForward(parameter_dict)
        start_time=time.time()
        train(model,parameter_dict,dataloader_train,epochs)
        end_time=time.time()
        duration=end_time-start_time

        metric_dict=get_performance_of_matric(model,epochs,duration,X_train,y_train,X_val,y_val,X_test,y_test)
        save_model('neural_networks', 'regression', model, parameter_dict,metric_dict)

        validation_RMSE=metric_dict['RMSE_loss_val']
        if validation_RMSE < best_val_loss :
            best_val_loss = validation_RMSE

            best_model_folder = "models/neural_networks/regression/best_model"

            os.makedirs(best_model_folder, exist_ok = True)

            if os.path.exists(f"{best_model_folder}/model.pt") == False:
                print("Best model written")
            else:
                print("The model is overwritten")

            torch.save(model.state_dict(), f"{best_model_folder}/model.pt")
            #save parameters
            with open(f"{best_model_folder}/hyperparameter.json",'w')as fp:
                json.dump(parameter_dict,fp)
            
            #Sae performance metrics
            with open(f"{best_model_folder}/metrics.json",'w') as fp:
                json.dump(metric_dict,fp)
        print("--" * 10)

