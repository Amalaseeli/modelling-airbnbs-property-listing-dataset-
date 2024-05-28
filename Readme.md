# Modelling Airbnb's property listing dataset.
    This is a datascience project which predict the price of Airbnb's property listing using deep learning techniques.
* Libraries used in this project.
1. pandas : To store and process dataframes created from datasets.
2. numpy : Perform some mathematical operations.
3. sklearn : Create machine learning models from dataset
4. pytorch : Define and train a neural network model, and view loss curve using summarywriter.
## Step1:
### Data Preparation

- initially using pandas to load the data
- Data cleaning
    * remove rows that have missing values.
    * As pandas don't recognize the values as a list, combine them in the string format.
    * There were some missing values, which were dropped.
    * The code "tabular_data.py" performs this task. The cleaned data stored in the "clean_tabular_data.csv"

## Step2
### Create Regression and Classification model
- Technologies/skills
* models 

    * Linear Regression
    * Logistic Regression
    * Decision trees
    * Random Forest
    * Gradient Boosting
* Skills

    * Gradient Descent
    * Validation and Testing model
    * Hyperparameter 

Regression
- Create baseline model using SGDRegressor.The code is available in 'Modelling.py'. But the model does not perform well as per the expectation beacause of the poor metrices of traing data.

- Tune hyperparameter using SGDRegressor.
A GridSearchCV was performed for the parameter metrics.However the model doesnot perform well.

Classification

- The four models were trained for this task:

    - LogisticRegression
    - DecisionTreeClassifier
    - RandomForestClassifier
    - GradientBoostingClassifier

The hyperparameters are tuned and models are saved in direcotry 'models/classification'

## Step3
### Creating Neural Network
- Technologies/skills

    * pytorch 
    * Datasets and Dataloaders
    * Activation function
    * optimizers
    * Backpropagation
    * Tensorboard

## Regression 

A simple neural network was implemented in PyTorch with the aim of improving on the machine learning regression model predictions for nightly property price.
- Following steps   are taken for developing, traing, testing the neural network.
1. Create Dataloader class :

    A dataloader class is defined for converting the features and labels from pandas dataframe to torch tensors.
2. Creating model class : 

    A model class Feedforward is created which defines the layers of the network. The model is defined to have Linear layers and also ReLU as activation layers. The hidden layer width and depth of the neural network are provided through a yaml file.
3. Defined parameter space
    There can be different combinations of hyperparameters and therefore all configurations are generated using a function 'generate_nn_config'. The network is trained for each of these configurations in loop.
    The model tune with following parameter 


    'optimizer': ['Adam', 'AdamW'],

    'learning_rate': [0.01, 0.001,0.0001],

    'hidden_layer_width': [32, 64],

    'depth': [4, 8]
    


4. Model training: 

    The network is trained for each of the coinfigurations mentioned above. Training is done for a number of epochs after dividing the data into batches, and the losses are measured at each step. These losses are written to tensorboard for visual inspection of network training. The following information is stored in directories named after date-time of training :

    * the best model
    * configuration: hyperparameters
    * performance metrics

5. Find best model : 

    the code evaulates the model by lowest RMSE loss for validation set and copies the best model, hyperparameters and performance metrics in a directory named 'models/neural_networks/regression/best_model'.

## Step 4
### Reuse the framework for another use-case with airbnb data

* Technologies/skills

    * Lable encoding: 

        Categorical variables in Python can be transformed into numerical labels using the label encoding technique. It gives each category in a variable a distinct numerical value, enabling machine learning algorithms to interpret and analyze the data effectively.
    * Refactoring

Earlier the model uses only the numerical columns as a input.But here label is the integer number of bedrooms where in feature the category column included. 








