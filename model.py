# Import the required libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# from compress_pickle import dump
import pickle

# Import the excel dataset
file = input('Enter the excel file for prediction: ')
df = pd.read_excel(file)

df2 = df.corr()
df1 = df2.round(decimals=2)

df1.drop(['YS (Mpa)', 'UTS (Mpa)'], axis =1)
final_YS = []
valYS = input('Enter the minimum Correlation value for YS: ')
for i in range (1, 34):
    if abs(df1.iloc[i, 34].astype(float)) >= abs(float(valYS)) :
        final_YS.append(df1.columns[i])
print(final_YS)

final_UTS = []
valUTS = input('Enter the minimum Correlation value for UTS: ')
for i in range (1, 34):
    if abs(df1.iloc[i, 35].astype(float)) >= abs(float(valUTS)):
        final_UTS.append(df1.columns[i])
print(final_UTS)
# Rename the target coulumn names
df.rename(columns = {'UTS (Mpa)':'UTS_mpa', 'YS (Mpa)': 'YS_mpa'}, inplace = True)

#Yield Strength
X = df[final_YS]
Y = df[['YS_mpa']]

# Train_Test_Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#standard scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Defining the model for Random Forest Regressor
model = RandomForestRegressor(n_jobs = -1)

# Estimating the best possible n_estimator by visualizing it
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, Y_train['YS_mpa'])
    scores.append(model.score(X_test, Y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
print("The best score possible for prediction of YS: ", np.amax(scores))
print("The best value of estimators for YS: ", np.amax(estimators))

# Building the model with best n_estimator
regressor = RandomForestRegressor(n_estimators = np.amax(estimators), random_state =0)

# Fit the Model
regressor.fit(X_train, Y_train['YS_mpa'])

# Predict the test value
y_pred = regressor.predict(X_test)

# Calculating the Score and Error values
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('Acuuracy:', metrics.r2_score(Y_test, y_pred)*100)

pickle.dump(regressor, open('model_rfYS.pkl', 'wb'))
# obj = dict(key1=[None, 1, 2, "3"] * 10000, key2="Test key")
# fname = "model_rfYS.pkl"
# dump(obj, fname)

# Ultimate Tensile Strength
# Defining the input and target data
x = df[final_UTS]
y = df[['UTS_mpa']]

# Train_Test_Split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Random Forest
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Defining the model for Random Forest Regression
model = RandomForestRegressor(n_jobs = -1)

# Estimating the best possible n_estimator by visualizing it
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(x_train, y_train['UTS_mpa'])
    scores.append(model.score(x_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
print("The best score possible for prediction of UTS: ", np.amax(scores))
print("The best value of estimators for UTS: ", np.amax(estimators))

# Building the model with best n_estimator
regressor = RandomForestRegressor(n_estimators = np.amax(estimators), random_state =0)

# Fit the model
regressor.fit(x_train, y_train['UTS_mpa'])

# Predict the test value
y_pred = regressor.predict(x_test)

# Calculating the Score and Error values
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Accuracy:', metrics.r2_score(y_test, y_pred)*100)

pickle.dump(regressor, open('model_rfUTS.pkl', 'wb'))
# obj = dict(key1=[None, 1, 2, "3"] * 10000, key2="Test key")
# fname1 = "model_rfUTS.pkl"
# dump(obj, fname1)

