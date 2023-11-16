import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tabular_data
import pandas as pd
import modelling

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
    