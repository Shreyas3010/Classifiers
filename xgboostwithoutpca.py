import xgboost
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import operator
import math

def sortSecond(val): 
    return val[1] 

data_train= pd.read_excel (r'train_comp.xlsx')
data_train= data_train.drop(['Date'],axis=1)
data_train=data_train[data_train['Outlier'] == 'No']
data_train= data_train.drop(['Outlier'],axis=1)

X_train= data_train.loc[:, data_train.columns != 'Total.Production.(mt)']
y_train = data_train.loc[:, data_train.columns == 'Total.Production.(mt)']


data_test= pd.read_excel (r'test_comp.xlsx')

data_test=data_test[data_test['Outlier'] == 'No']
data_test= data_test.drop(['Outlier'],axis=1)
x1=data_test['Date']
data_test= data_test.drop(['Date'],axis=1)

X_test= data_test.loc[:, data_test.columns != 'Total.Production.(mt)']
y_test = data_test.loc[:, data_test.columns == 'Total.Production.(mt)']




model = xgboost.XGBRegressor()
model.fit(X_train , y_train)

y_pred=model.predict(X_test)

y1=y_test['Total.Production.(mt)']
y_test_arr=np.array(y1)
num_test=len(y_pred)
diff1=np.arange(num_test,dtype=np.float)
sum1=0
for i in range(num_test):
    diff1[i]=(abs(y_pred[i]-y_test_arr[i])/y_test_arr[i])
    sum1=sum1+diff1[i]

min1=min(diff1)
max1=max(diff1)
print("min",min1)
print("max",max1)
avg1=sum1/num_test
print("avg",sum1/num_test)

sum2=0
diff2=np.arange(num_test,dtype=np.float)
for i in range(num_test):
    diff2[i]=math.pow(abs((y_pred[i]-y_test_arr[i])/y_test_arr[i])-avg1,2)
    sum2=sum2+diff2[i]

std1=sum2/num_test
std1=math.sqrt(std1)
maxstr=str(max1)
minstr=str(min1)
avgstr=str(avg1)
stdstr=str(std1)

f_imp=[]
features_imp=model.feature_importances_
train_cols=X_train.columns.values
for i in range(len(train_cols)):
    f_imp.append((train_cols[i],features_imp[i]))

f_imp.sort(key=sortSecond,reverse=True)

plt.figure(figsize=(20,10))
plt.suptitle('Min : '+minstr+'Max : '+maxstr+' Avg : '+avgstr+' St. devi. :'+stdstr, fontsize=14, fontweight='bold')
plt.plot(x1,diff1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="")
plt.title('Performance (XGB lib)')
plt.xticks(rotation=90)
plt.xlabel('Production')
plt.ylabel('(Predicted-Actual)/Actual ')
plt.savefig('xgb/train_test_lib_xgb.png')
#plt.savefig('resultsrfnewdata.png')
plt.show()
