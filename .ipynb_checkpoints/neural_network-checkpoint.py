from typing import Any
import torch
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
    def __init__(self) -> None:
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Linear(11,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1)
        )

    def forward(self, features) -> Any:
        return self.layers(features)
      
def train(model,dataloader,epochs:int):
     '''The functions trains the model and adds loss to TensorBoard'''
     optimiser=torch.optim.SGD(model.parameters(),lr=0.001)
     writer=SummaryWriter()

     for epoch in range(epochs):
          for batch_idx,batch in enumerate(dataloader):
               features,labels=batch 
               
            
               #forward pass
               prediction=model(features)
               loss=F.mse_loss(prediction,labels)
               loss.backward()
               
               #Gradient optimisation
               optimiser.step()
               optimiser.zero_grad()
               writer.add_scalar("loss",loss.item(),batch_idx)
        
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
    model=FeedForward()
    epochs=200
    train(model,dataloader_train,epochs)