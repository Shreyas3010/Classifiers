from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sortSecond(val): 
    return val[1]  

data= pd.read_excel (r'Data_Training_Model2.xlsx')
#print(data_train.head())
#data_train=data_train.dropna(axis=0)
data_train=data.iloc[0:617,:]
data_test=data.iloc[617:707,:]
data_train= data_train.drop(['DATE','Availability.(%)'],axis=1)
m=data_train.columns[data_train.isna().any()].tolist()
data_train=data_train.dropna(axis=1)
#print("NAN",m)
#print(data_test.head())
data_test= data_test.drop(m,axis=1)
#data_test=data_test.dropna(axis=0)
data_test=data_test.dropna(axis=0)
date_test= data_test.loc[:, data_test.columns == 'DATE']
data_test= data_test.drop(['DATE','Availability.(%)'],axis=1)
#data_test= data_test.drop(['DATE'],axis=1)
X_train= data_train.loc[:, data_train.columns != 'Total.Production.(mt)']
y_train= data_train.loc[:, data_train.columns == 'Total.Production.(mt)']
X_test= data_test.loc[:, data_test.columns != 'Total.Production.(mt)']
y_test= data_test.loc[:, data_test.columns == 'Total.Production.(mt)']

# fit model no training data

#R=[0,1,2,3,4,5,6,7,8,9]
R=[1]
row1=np.arange(len(R))
row2=np.arange(len(R)*5)
results= pd.DataFrame(data=None,index=row1,columns = ['Random State','Min','Max','Avg'])
features1=pd.DataFrame(data=None,index=row2,columns = ['Random State','Feature','Score'])
a1=0
a2=0
for rs in R:
    print("Random State",rs)  
    results['Random State'][a1]=rs
    clf=RandomForestRegressor(random_state=rs,n_estimators=100,oob_score=True)
    clf.fit(X_train,y_train.values.ravel())
    print(clf)
    y_pred=clf.predict(X_test)
    rsstr=str(rs)
    x1=date_test['DATE']
    y1=y_test['Total.Production.(mt)']
    
    y_test_arr=np.array(y1)
    num_test=len(y_pred)
    y2=y_pred
    diff1=np.arange(num_test,dtype=np.float)
    sum1=0
    for i in range(num_test):
        diff1[i]=(abs(y_pred[i]-y_test_arr[i])/y_pred[i])
        sum1=sum1+diff1[i]
        
    plt.figure(figsize=(20,10))
    plt.bar(range(num_test),diff1)
    plt.title('Performance (Random State : '+rsstr+')')
    #plt.xticks(rotation=90)
    plt.xlabel('Production')
    plt.ylabel('(Predicted-Actual)/Predicted (%)')
    plt.savefig('Graph/resultsrandomforest'+rsstr+'.png')
    plt.show()
    
    
    min1=min(diff1)
    max1=max(diff1)
    print("min",min1)
    print("max",max1)
    avg1=sum1/num_test
    print("avg",sum1/num_test)
    results['Min'][a1]=min1
    results['Max'][a1]=max1
    results['Avg'][a1]=avg1
    a1=a1+1
   
    #plot
    
    plt.figure(figsize=(20,10))
    plt.plot(x1,y1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="Actual Data")
    plt.plot(x1,y2,marker='o', markerfacecolor='forestgreen', markersize=7, color='lightgreen', linewidth=0,label="Predicted Data")
    plt.xticks(rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Production')
    plt.title('Actual Data vs Predicted Data (Random State : '+rsstr+')')
    plt.legend()
    plt.savefig('Graph/randomforest'+rsstr+'.png')
    plt.show()
    
    #5 best feature 
    f_imp=[]
    features_imp=clf.feature_importances_
    train_cols=X_train.columns.values
    for i in range(len(train_cols)):
        f_imp.append((train_cols[i],features_imp[i]))
    
    f_imp.sort(key=sortSecond,reverse=True)
    for i in range(5):
        features1['Random State'][a2]=rs
        features1['Feature'][a2]=f_imp[i][0]
        features1['Score'][a2]=f_imp[i][1]
        a2=a2+1
    
    # plot feature importance
    
    plt.figure(figsize=(20,10))
    #print(train_cols)
    #print(clf.feature_importances_)
    plt.title('Feature importances (Random State : '+rsstr+')')
    plt.bar(train_cols, clf.feature_importances_)
    plt.xticks(rotation=90)
    plt.ylabel('Feature Importance (%)')
    plt.xlabel('Features')
    plt.legend()
    plt.savefig('Graph/randomforestfeatureimportance'+rsstr+'.png')
    plt.show()

