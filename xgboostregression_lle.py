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
from sklearn.manifold import Isomap
import math
from sklearn.manifold import LocallyLinearEmbedding

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


x_train_val= X_train.values
# Separating out the target
y_train_val= y_train.values
# Standardizing the features
x_train_val= StandardScaler().fit_transform(x_train_val)

#number_of_components=40
#85%
#number_of_components=75
#95%
#number_of_components=95
#96%

number_of_components=33
#iso best

n_n=10
#5 is bydefalut

embedding = Isomap(n_components=number_of_components,n_neighbors=n_n)
#embedding = LocallyLinearEmbedding(n_components=number_of_components)
#pca = PCA(n_components=number_of_components)
principalComponents_train= embedding.fit_transform(x_train_val)
#print(embedding.explained_variance_.sum())

col=[]
for i in range(1,number_of_components+1):
  col_name='principal coponent '+ str(i)
  col.append(col_name)

principalDf_train= pd.DataFrame(data = principalComponents_train, columns = col)
#print("explained_variance_ratio_",embedding.explained_variance_ratio_)
#print("pca.components_",abs( embedding.components_ ))
#print("abs( pca.components_ )[0]",abs( embedding.components_ )[0])

#ind_to_col={}
#count=0
#for j in abs( embedding.components_ )[0]:
#  ind_str=X_train.columns[count]
#  ind_to_col[ind_str]=j
#  count+=1
  
#print("ind_to_col",ind_to_col)

#sorted_x_train_96 = sorted(ind_to_col.items(), key=operator.itemgetter(1),reverse=True)
#print("sorted_x_train_96",sorted_x_train_96)
finalDf_train= pd.concat([principalDf_train, y_train[['Total.Production.(mt)']]], axis = 1)




x_test_val= X_test.values
# Separating out the target
y_test_val= y_test.values
# Standardizing the features
x_test_val= StandardScaler().fit_transform(x_test_val)


#embedding = LocallyLinearEmbedding(n_components=number_of_components)
embedding = Isomap(n_components=number_of_components,n_neighbors=n_n)
#pca = PCA(n_components=number_of_components)
principalComponents_test= embedding.fit_transform(x_test_val)
#print(embedding.explained_variance_.sum())


principalDf_test= pd.DataFrame(data = principalComponents_test, columns = col)
#print("explained_variance_ratio_",embedding.explained_variance_ratio_)
#print("pca.components_",abs( embedding.components_ ))
#print("abs( pca.components_ )[0]",abs( embedding.components_ )[0])

#ind_to_col={}
#count=0
#for j in abs( embedding.components_ )[0]:
#  ind_str=X_test.columns[count]
#  ind_to_col[ind_str]=j
#  count+=1
#  
#print("ind_to_col",ind_to_col)
#
#sorted_x_test= sorted(ind_to_col.items(), key=operator.itemgetter(1),reverse=True)
#print("sorted_x_test",sorted_x_test)

finalDf_test= pd.concat([principalDf_test, y_test[['Total.Production.(mt)']]], axis = 1)



model = xgboost.XGBRegressor()
model.fit(principalDf_train , y_train)

y_pred=model.predict(principalDf_test)

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

plt.figure(figsize=(20,10))
plt.suptitle('Min : '+minstr+'Max : '+maxstr+' Avg : '+avgstr+' St. devi. :'+stdstr, fontsize=14, fontweight='bold')
plt.plot(x1,diff1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="")
plt.title('Performance (XGB) 33 Isomap neighbours 10')
plt.xticks(rotation=90)
plt.xlabel('Production')
plt.ylabel('(Predicted-Actual)/Actual (%)')
#plt.savefig('Dimensionality Reduction/train_test_Isomap_xgb_33cmp_n_n_10.png')
#plt.savefig('resultsrfnewdata.png')
plt.show()
