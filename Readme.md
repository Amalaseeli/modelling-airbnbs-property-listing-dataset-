# Modelling Airbnb's property listing dataset.
## Building a Modelling Framework
step1:
### Data Preparation
- initially using pandas to load the data
- Data cleaning
    * remove rows that have missing values.
    * As pandas don't recognize the values as a list, combine them in the string format.

### Split data as a features, label
    - features: pandas data frame of tabular data
    - label: Used for the prediction of the model
