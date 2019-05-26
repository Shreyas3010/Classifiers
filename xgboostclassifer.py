from xgboost import XGBClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections
from sklearn.metrics import confusion_matrix
def rnd1(val,pt):
    if val>pt:
        return 1
    else:
        return 0

data = pd.read_csv("diabetes.csv")
X = data.loc[:, data.columns != 'Outcome']
y = data.loc[:, data.columns == 'Outcome']
datasize=collections.Counter(y['Outcome'])
print("data size",datasize)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,train_size=0.7, random_state = 0)
testingdatasize=collections.Counter(y_test['Outcome'])
print("test data size",testingdatasize)
trainingdatasize=collections.Counter(y_train['Outcome'])
print("original training data size",trainingdatasize)
numofmaj=trainingdatasize[0]
numofmin=trainingdatasize[1]
raw1=np.arange(2)
results= pd.DataFrame(data=None,index=raw1,columns = ['Class','Datasize','Training Datasize','Precision','Recall','f1-score','Testing Datasize'])
b=a
a=a+1
results['Class'][b]=0
results['Class'][a]=1
results['Datasize'][b]=datasize[0]
results['Datasize'][a]=datasize[1]
results['Training Datasize'][b]=trainingdatasize[0]
results['Training Datasize'][a]=trainingdatasize[1]
results['Testing Datasize'][b]=testingdatasize[0]
results['Testing Datasize'][a]=testingdatasize[1]
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [rnd1(value,i) for value in y_pred]
y_test_arr=np.array(y_test['Outcome'])
#feature_imp=grid_best_clf.feature_importances_
#print("feature importances",feature_imp)
y_act_count=collections.Counter(y_test['Outcome'])
y_pred_count=collections.Counter(y_pred)
print("predicted",y_pred_count)
print("actual",y_act_count)
y_test_arr=np.array(y_test['Outcome'])
cn_mat=confusion_matrix( predictions,y_test['Outcome'])
print(cn_mat)
TP=cn_mat[1][1]
TN=cn_mat[0][0]
FN=cn_mat[0][1]
FP=cn_mat[1][0]
pre1=TP/(TP+FP)
recall1=TP/(TP+FN)
NPV1=TN/(TN+FN)
speci1=TN/(TN+FP)
results['Precision'][b]=round(NPV1,5)
results['Recall'][b]=round(speci1,5)
results['Precision'][a]=round(pre1,5)
results['Recall'][a]=round(recall1,5)
print("Precision : ",pre1,"Recall : ",recall1,"Negative Predictive Rate(TN/(TN+FN)) : ",NPV1,"Specificity : ",speci1)
F1=2*(pre1*recall1)/(pre1+recall1)
F2=2*(NPV1*speci1)/(NPV1+speci1)
results['f1-score'][b]=round(F2,5)
results['f1-score'][a]=round(F1,5)
print("F1 : ",F1,"F2 : ",F2)
a=a+1
print("a",a)
print(classification_report(y_test_arr, predictions))

