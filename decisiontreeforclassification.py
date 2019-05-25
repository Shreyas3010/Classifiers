# Run this program on your local python 
# interpreter, provided you have installed 
# the required libraries. 

# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.ensemble import RandomForestClassifier
# Function importing Dataset 
def importdata(): 
    #balance_data=
    balance_data=load_iris()
    #balance_data = pd.read_csv( 
#'https://archive.ics.uci.edu/ml/machine-learning-'+
#'databases/balance-scale/balance-scale.data', 
	#sep= ',', header = None) 
	
    #balance_data= load_boston();
    #balance_data =load_breast_cancer()
	#print ("Dataset Lenght: ", len(balance_data)) 
	# Printing the dataswet shape 
	#print ("Dataset Shape: ", balance_data.shape) 
	
	# Printing the dataset obseravtions 
	#print ("Dataset: ",balance_data.head()) 
    return balance_data 

# Function to split the dataset 
def splitdataset(balance_data): 
    
    X=balance_data['data']
    Y=balance_data['target']
    	# Seperating the target variable 
    #X = balance_data.values[:, 1:5] 
	#Y = balance_data.values[:, 0] 

    #X =  load_breast_cancer()['data']
    #Y = load_breast_cancer()['target'] 	
	
    #print(X)
	# Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100) 
	
    return X, Y, X_train, X_test, y_train, y_test 
	
# Function to perform training with giniIndex. 
def train_using_gini(data,X_train, X_test, y_train): 
		# Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
    			random_state = 100,max_depth=3, min_samples_leaf=5) 
    
    	# Performing training 
    clf_gini.fit(X_train, y_train)
    #tree.export_graphviz(clf_gini,out_file='iris_gini.dot') 
    dot_data = tree.export_graphviz(clf_gini, out_file=None,feature_names=data.feature_names,class_names=data.target_names,filled=True, rounded=True,special_characters=True)
    graph = graphviz.Source(dot_data)
    print(graph)
    return clf_gini
	
# Function to perform training with entropy. 
def train_using_entropy(data,X_train, X_test, y_train): 

	# Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(X_train, y_train)
    #tree.export_graphviz(clf_entropy,out_file='iris_entropy.dot')
    dot_data = tree.export_graphviz(clf_entropy, out_file=None,feature_names=data.feature_names,class_names=data.target_names,filled=True, rounded=True,special_characters=True)
    graph = graphviz.Source(dot_data)
    print(graph)
    return clf_entropy 

def train_randomforest(features,target):    
    clf = RandomForestClassifier(n_estimators=100,random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf.fit(features, target)
    return clf
# Function to make predictions 
def prediction(X_test, clf_object): 

	# Predicton on test with giniIndex 
	y_pred = clf_object.predict(X_test) 
	print("Predicted values:") 
	print(y_pred) 
	return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
	
	print("Confusion Matrix: ", 
		confusion_matrix(y_test, y_pred)) 
	
	print ("Accuracy : ", 
	accuracy_score(y_test,y_pred)*100) 
	
	print("Report : ", 
	classification_report(y_test, y_pred)) 

# Driver code 
def main(): 
	
	# Building Phase 
    data = importdata() 
	#print("daattaa")
	#print(data[0])
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(data,X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(data,X_train, X_test, y_train) 
    clf_randomforest = train_using_entropy(X_train,y_train)
	#print("y_predgini")
	#print(clf_gini)
	#print(clf_entropy)
	# Operational Phase 
    print("Results Using Gini Index:") 
	
	# Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
	
    print("Results Using Entropy:") 
	# Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 

    print("Results Using Random forest:") 
	# Prediction using entropy 
    #y_pred_entropy = prediction(X_test, clf_randomforest) 
    #cal_accuracy(y_test, y_pred_entropy) 
	
# Calling main function 
if __name__=="__main__": 
	main() 
