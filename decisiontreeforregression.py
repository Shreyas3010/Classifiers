import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#import data
data= pd.read_excel (r'Data_Training_Model2.xlsx')

#slipt train and test data
data_train=data.iloc[0:617,:]
data_test=data.iloc[617:707,:]

#remove data and availability cols
data_train= data_train.drop(['DATE','Availability.(%)'],axis=1)
m=data_train.columns[data_train.isna().any()].tolist()
data_train=data_train.dropna(axis=1)

#remove those cols that is removed from train test
data_test= data_test.drop(m,axis=1)

data_test=data_test.dropna(axis=0)
date_test= data_test.loc[:, data_test.columns == 'DATE']
data_test= data_test.drop(['DATE','Availability.(%)'],axis=1)

X_train= data_train.loc[:, data_train.columns != 'Total.Production.(mt)']
y_train= data_train.loc[:, data_train.columns == 'Total.Production.(mt)']
X_test= data_test.loc[:, data_test.columns != 'Total.Production.(mt)']
y_test= data_test.loc[:, data_test.columns == 'Total.Production.(mt)']


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
