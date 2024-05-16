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






