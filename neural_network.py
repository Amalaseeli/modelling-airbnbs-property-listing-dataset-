from typing import Any
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tabular_data
import pandas as pd
import modelling

import torch.nn.functional as F


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
        layers.append(torch.nn.Sigmoid())
        self.layers=torch.nn.Sequential(*layers)


    def forward(self, features) -> Any:
        return self.layers(features)
      
def train(model,config,dataloader,epochs:int):
     '''The functions trains the model and adds loss to TensorBoard'''
     optimiser=torch.optim.SGD(model.parameters(),lr=config['learning_rate'])
     writer=SummaryWriter()
     batch_idx=0

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
               optimiser.step()
               optimiser.zero_grad()
               writer.add_scalar("loss",loss.item(),batch_idx)
               batch_idx+=1


def get_rmse_r2_score(model,feature,label):
    feature=torch.tensor(feature.values).type(torch.float32)   
    label=torch.tensor(label.value).type(torch.float32)
    label=torch.unsqueeze(label,1)
    prediction=model(feature)
    rmse_loss=torch.sqrt(F.mse_loss(prediction,label.float()))
    r2_score = 1 - rmse_loss / torch.var(label.float())
    return rmse_loss, r2_score

def get_nn_config():
    with open('nn_config.yaml','r') as file:
        data=yaml.safe_load(file)
        print(data)
        return data
       
     



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
    batch_size=16
    dataloader_train=DataLoader(dataset_train,batch_size,shuffle=True)
    dataloader_test=DataLoader(dataset_test,batch_size,shuffle=False)
    dataloader_val=DataLoader(dataset_val,batch_size,shuffle=True)
    
  
    # for batch in dataloader_train:
    #         print(batch)
    #         features,labels=batch
    #         print(features.shape)
    #         print(labels.shape)
    #         break
    config=get_nn_config()
    model=FeedForward(config)

    print(model)
    epochs=200
    train(model,dataloader_train,epochs)
    get_nn_config()