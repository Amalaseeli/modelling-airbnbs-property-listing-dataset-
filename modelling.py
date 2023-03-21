from tabular_data import clean_tabular_data 
import pandas as pd
# from sklearn import datasets, model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
tabular_data=clean_tabular_data()

df=pd.read_csv('./tabular_data/listing.csv')
kwargs={'label':'Price_Night', 'features':['beds', 'bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','Unnamed: 19']}

X,y=tabular_data.load_airbnb(df,**kwargs)
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))

reg.fit(X_train, y_train)

# y_pred = reg.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean squared error:", mse)




