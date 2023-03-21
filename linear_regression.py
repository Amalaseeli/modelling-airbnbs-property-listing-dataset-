from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt
X, y = datasets.fetch_california_housing(return_X_y=True)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y, test_size=0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
)
print(X_train.shape, y_train.shape)

class LinearRegression:
    def __init__(self,n_features:int): #initialise parameter
        np.random.seed(10)
        self.W=np.random.randn(n_features,1) #randomly initialise weight
        self.b=np.random.randn(1) #randomly initailise bias

    def __call__(self, x):# how do we calculate the output from an input in our model?
         ypred = np.dot(X, self.W) + self.b
         return ypred #return prediction

    def update_params(self,W,b):
        self.W=W ## set this instance's weights to the new weight value passed to the function
        self.b=b ## do the same for the bias

    def plot_predictions(self,y_pred, y_true):
        samples=len(y_pred)
        plt.figure()
        plt.scatter(np.arange(samples),y_pred, c='r', label='predictions')
        plt.scatter(np.arange(samples),y_true, c='b', label='true labels', marker='x')
        plt.legend()
        plt.xlabel('Sample numbers')
        plt.ylabel('Values')
        plt.show()

    def mean_squared_error(self, y_pred, y_true):
        error=y_pred - y_true
        squred_error = error ** 2
        return np.mean(squred_error)

    def minimize_loss(self,X_train, y_train):
        X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        optimal_w = np.matmul(
            np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
            np.matmul(X_with_bias.T, y_train),
        )
        return optimal_w[1:], optimal_w[0]


    
model = LinearRegression(n_features=8)
y_pred = model(X_test) # make predictions with data
print("Predictions:\n", y_pred[:10]) # print the first 10 predictions
model.plot_predictions(y_pred[:10], y_test[:10])

weights, bias = model.minimize_loss(X_train, y_train)
print(weights, bias)
model.update_params(weights, bias)
y_pred = model(X_train)
cost = model.mean_squared_error(y_pred, y_train)
print(cost)


from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.fetch_california_housing(return_X_y=True)

model = linear_model.LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred[:5], "\n", y[:5])

from sklearn import metrics

metrics.mean_squared_error(y, y_pred)