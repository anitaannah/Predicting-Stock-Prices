#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df=pd.read_csv(r'C:\Users\HP\Downloads\BAJFINANCE.csv')
df.head()


# In[6]:


df.set_index('Date',inplace=True)


# Plotting the target variable VWAP over time

# In[7]:


df['VWAP'].plot()


# We can observe some kind of seasonality here.

# NOTE:
# Feature Engineering Almost every time series problem will have some external features or some internal feature engineering to help the model.
# 
# Let's add some basic features like lag values of available numeric features that are widely used for time series problems. Since we need to predict the price of the stock for a day, we cannot use the feature values of the same day since they will be unavailable at actual inference time. We need to use statistics like mean, standard deviation of their lagged values.
# 
# We will use three sets of lagged values, one previous day, one looking back 7 days and another looking back 30 days as a proxy for last week and last month metrics.

# Data Pre-Processing

# In[8]:


df.isna().sum()


# In[9]:


df.dropna(inplace=True)


# In[10]:


df.isna().sum()


# In[11]:


data=df.copy()


# In[12]:


data.dtypes


# In[13]:


data.columns


# In[14]:


lag_features=['High','Low','Volume','Turnover','Trades']
window1=3
window2=7


# In[15]:


for feature in lag_features:
    data[feature+'rolling_mean_3']=data[feature].rolling(window=window1).mean()
    data[feature+'rolling_mean_7']=data[feature].rolling(window=window2).mean()


# In[16]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[17]:


data.head()


# In[18]:


data.columns


# In[19]:


data.shape


# In[20]:


data.isna().sum()


# In[21]:


data.dropna(inplace=True)


# In[22]:


data.columns


# In[23]:


data.isna().sum()


# In[24]:



#independent features

ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[25]:


training_data=data[0:1800]
test_data=data[1800:]


# In[26]:


training_data


# In[27]:


get_ipython().system('pip install pmdarima')


# In[28]:


from pmdarima import auto_arima


# In[29]:


import warnings
warnings.filterwarnings('ignore')


# In[30]:


model=auto_arima(y=training_data['VWAP'],exogenous=training_data[ind_features],trace=True)


# In[31]:


model.fit(training_data['VWAP'],training_data[ind_features])


# In[32]:


forecast=model.predict(n_periods=len(test_data), exogenous=test_data[ind_features])


# In[33]:


test_data['Forecast_ARIMA']=forecast


# In[34]:


test_data[['VWAP','Forecast_ARIMA']].plot(figsize=(14,7))


# The Auto ARIMA model seems to do a fairly good job in predicting the stock price

# CHECKING ACCURACY OF OUR MODEL

# In[35]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[36]:


np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA']))


# In[37]:


mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA'])


# In[ ]:




