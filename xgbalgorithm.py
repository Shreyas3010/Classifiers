#import xgboost
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

data_train= pd.read_excel (r'train_comp.xlsx')
data_train= data_train.drop(['Date'],axis=1)
data_train=data_train[data_train['Outlier'] == 'No']
data_train=data_train.reset_index()
data_train= data_train.drop(['Outlier','index'],axis=1)

X_train= data_train.loc[:, data_train.columns != 'Total.Production.(mt)']
y_train = data_train.loc[:, data_train.columns == 'Total.Production.(mt)']

data_test= pd.read_excel (r'test_comp.xlsx')

data_test=data_test[data_test['Outlier'] == 'No']
data_test=data_test.reset_index()
data_test= data_test.drop(['Outlier','index'],axis=1)
x1=data_test['Date']
data_test= data_test.drop(['Date'],axis=1)

X_test= data_test.loc[:, data_test.columns != 'Total.Production.(mt)']
y_test = data_test.loc[:, data_test.columns == 'Total.Production.(mt)']

row1=np.arange(36)
results= pd.DataFrame(data=None,index=row1,columns = [ 'iteration','alpha','min','max','avg','stdev'])

A=[5,10,15,20,25,30,40,50,100]
B=[0.5,0.1,0.05,0.01]
#A=[10]
a1=0
for noiteration in A:
    no_of_iteration=noiteration
    for alpha in B:
        results['iteration'][a1]=no_of_iteration
        results['alpha'][a1]=alpha
        no_of_clf=list(range(no_of_iteration))
        prediction_train=list(np.array(y_train['Total.Production.(mt)']*1.0) for i in range(no_of_iteration+1))
        
        #y_train_forxgb=y_train
        
        for i in range(no_of_iteration):
            no_of_clf[i] = RandomForestRegressor(n_estimators=100, max_depth=5,random_state=0,oob_score=True)
            no_of_clf[i].fit(X_train ,prediction_train[i])
            y_train_pred=no_of_clf[i].predict(X_train)
            for j in range(len(y_train)):
                prediction_train[i+1][j]=2*alpha*(prediction_train[i][j]-y_train_pred[j])
        
        #y_train_5=prediction_train[i]
        #model = xgboost.XGBRegressor()
        #model.fit(X_train , y_train)
        
        prediction_test=list(np.array(y_test['Total.Production.(mt)']) for i in range(no_of_iteration))
        
        y_pred=np.array(y_test)
        
        for i in range(no_of_iteration):
            prediction_test[i]=no_of_clf[i].predict(X_test)
            
        for k in range(no_of_iteration-1):
            for j in range(len(y_test)):
                prediction_test[k+1][j]=prediction_test[k+1][j]/(2*alpha)
                
        
            
        for j in range(len(X_test)):
            res=0
            for k in range(no_of_iteration):
                res=res+prediction_test[k][j]
            y_pred[j]=res    
            
        
        
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
        
        results['min'][a1]=round(min1,6)
        results['max'][a1]=round(max1,6)
        
        std1=sum2/num_test
        std1=math.sqrt(std1)
        results['avg'][a1]=round(avg1,6)
        results['stdev'][a1]=round(std1,6)
        a1=a1+1
        maxstr=str(max1)
        minstr=str(min1)
        avgstr=str(avg1)
        stdstr=str(std1)
        
        #f_imp=[]
        #features_imp=model.feature_importances_
        #train_cols=X_train.columns.values
        #for i in range(len(train_cols)):
        #    f_imp.append((train_cols[i],features_imp[i]))
        
        noi=str(no_of_iteration)
        alphastr=str(alpha)
        plt.figure(figsize=(20,10))
        plt.suptitle('Min : '+minstr+'Max : '+maxstr+' Avg : '+avgstr+' St. devi. :'+stdstr, fontsize=14, fontweight='bold')
        plt.plot(x1,diff1,marker='o', markerfacecolor='blue', markersize=7, color='skyblue', linewidth=0,label="")
        plt.title('Performance (XGB algo) No. of iteration : '+noi +'Alpha : '+alphastr)
        plt.xticks(rotation=90)
        plt.xlabel('Production')
        plt.ylabel('(Predicted-Actual)/Actual ')
        #plt.savefig('train_test_tryxgbalgo_xgb'+noi+'aplha'+alphastr+'.png')
        #plt.savefig('resultsrfnewdata.png')
        plt.show()
