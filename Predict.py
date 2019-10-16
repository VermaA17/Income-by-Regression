#Importing all the libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#Loading training and test data
df=pd.read_csv('TrainingData.csv')
df1=pd.read_csv('TestData.csv')

#Imputer step to fill in all missing values:
df.iat[0,2] = 'other'
df.iat[4,7] ='Master'


df=df.replace('0',np.nan)
df=df.fillna(method='ffill')

df1=df1.replace('0',np.nan)
df1=df1.fillna(method='ffill')

#Encoding step to transform categorical data to numerical data using Target Encoder :
mean_encode1 =df.groupby('Gender')['Income in EUR'].mean()
df.loc[:,'Gender_mean_enc']=df['Gender'].map(mean_encode1)

mean_encode2 =df.groupby('Country')['Income in EUR'].mean()
df.loc[:,'Country_mean_enc']=df['Country'].map(mean_encode2)

mean_encode3 =df.groupby('University Degree')['Income in EUR'].mean()
df.loc[:,'University Degree_mean_enc']=df['University Degree'].map(mean_encode3)

mean_encode4 =df.groupby('Profession')['Income in EUR'].mean()
df.loc[:,'Profession_mean_enc']=df['Profession'].map(mean_encode4)


#Target encoding for test data

df1.loc[:,'Gender_mean_enc']=df1['Gender'].map(mean_encode1)

df1.loc[:,'Country_mean_enc']=df1['Country'].map(mean_encode2)

df1.loc[:,'University Degree_mean_enc']=df1['University Degree'].map(mean_encode3)


df1.loc[:,'Profession_mean_enc']=df1['Profession'].map(mean_encode4)

#After encoding actual test data has some null values ,again they are replaced by most frequent values:
df1=df1.fillna(method='ffill')


#Training predictors and target:

x = df.iloc[:, [1,3,5,12,13,14,15]].values
y = df.iloc[:, 11].values

#Features from Test data:
x1=df1.iloc[:, [1,3,5,12,13,14,15]].values

#Splitting the test and training data:
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#Random forest regressor to predict the income
regressor =RandomForestRegressor(n_estimators=150,random_state=0,max_features = "auto",bootstrap = True,min_samples_split=2,verbose=1,min_samples_leaf=1,warm_start=True)
regressor.fit (x_train,y_train)
y_pred=regressor.predict(x_test)

#To check the accuracy of model

from math import sqrt
rms = sqrt(mean_squared_error(y_test,y_pred))
print(rms)

#same regressor to predcit the income from test data:
y_pred1=regressor.predict(x1)


#generating and saving the predcited income in a .csv file:
res = pd.DataFrame(y_pred1)
res.to_csv("prediction_results.csv")
