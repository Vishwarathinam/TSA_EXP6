<H1 ALIGN =CENTER> Ex.No: 6 --  HOLT WINTER'S METHOD...</H1>

### Date: 

### AIM :

To create and implement Holt Winter's Method Model using python.

### ALGORITHM :

#### Step 1 : 

You import the necessary libraries.

#### Step 2 : 

You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration.

#### Step 3 : 

You group the data by date and resample it to a monthly frequency (beginning of the month.

#### Step 4 : 

You plot the time series data.

#### Step 5 : 

You import the necessary 'statsmodels' libraries for time series analysis.

#### Step 6 : 

You decompose the time series data into its additive components and plot them.

#### Step 7 : 

You calculate the root mean squared error (RMSE) to evaluate the model's performance.

#### Step 8 : 

You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions.

#### Step 9 : 

You plot the original sales data and the predictions.

### PROGRAM :

```python

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df=pd.read_csv('dailysales.csv',parse_dates=['date'])
df.info()
df.head()
df.isnull().sum()

df=df.groupby('date').sum()
df.head(10)
df=df.resample(rule='MS').sum()
df.head(10)
df.plot()

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(df,model='additive').plot();
train=df[:19] #till Jul19
test=df[19:] # from aug19
train.tail()
test

from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwmodel=ExponentialSmoothing(train.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

test_pred=hwmodel.forecast(5)
test_pred
train['sales'].plot(legend=True, label='Train', figsize=(10,6))
test['sales'].plot(legend=True, label='Test')
test_pred.plot(legend=True, label='predicted_test')

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test,test_pred))
df.sales.mean(), np.sqrt(df.sales.var())

final_model=ExponentialSmoothing(df.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

pred=final_model.forecast(10)
pred
df['sales'].plot(legend=True, label='sales', figsize=(10,6))
pred.plot(legend=True, label='prediction')

```

### OUTPUT :

#### SALES PLOT : 
![t1](https://github.com/Vishwarathinam/TSA_EXP6/assets/95266350/398b1623-be9a-4c4c-8887-874de9d519bb)


#### SEASONAL DECOMPOSING (ADDITIVE) :

![t2](https://github.com/Vishwarathinam/TSA_EXP6/assets/95266350/54bef8b2-5e55-466e-9e37-0723ca6b3113)

#### TEST_PREDICTION :

![t3](https://github.com/Vishwarathinam/TSA_EXP6/assets/95266350/60f99c9e-5910-45b7-a593-0ec36042479a)

#### FINAL_PREDICTION :

![t4](https://github.com/Vishwarathinam/TSA_EXP6/assets/95266350/fbe9f8d6-e2a9-487a-b3bb-44da9b5e8ddb)

### RESULT :

Thus, the program run successfully based on the Holt Winter's Method model.

