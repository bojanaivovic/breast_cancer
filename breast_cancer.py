# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:19:23 2022

@author: Bojana Ivovic 637/2018
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sns
from seaborn import countplot
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


col_names=["class",'age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
data = pd.read_csv('breast-cancer.csv',names=col_names, header=None)


#tekstualni podaci-> numericki
data['age'] = data['age'].map( {'20-29':25, '30-39':35,'40-49':45,'50-59':55, '60-69':65,'70-79':75,'80-89':85,'90-99':95} )
data['menopause'] = data['menopause'].map( {'premeno':1, 'ge40': 2, 'lt40':3} )
data['tumor-size'] = data['tumor-size'].map( {'0-4':2, '5-9':7,'10-14':12,'15-19':17, '20-24':22,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52} )
data['inv-nodes'] = data['inv-nodes'].map( {'0-2':1, '3-5':4,'6-8':7,'9-11':10, '12-14':13,'15-17':16,'18-20':19,'21-23':22,'24-26':25,'27-29':28,'30-32':31,'33-35':34,'36-38':37,'39':39} )
data['node-caps'] = data['node-caps'].map( {'no':0, 'yes': 1, '?':np.nan} )
#deg malig : 1,2,3
data['breast'] = data['breast'].map( {'left':1, 'right':2} )
data['breast-quad'] = data['breast-quad'].map( {'left_up':1, 'left_low': 2, 'right_up':3, 'right_low':4, 'central':5, '?':np.nan} )
data['irradiat'] = data['irradiat'].map( {'no': 0, 'yes': 1} )
data['class']=data['class'].map({'no-recurrence-events': 0, 'recurrence-events': 1})

#provera NaN vrednosti
#print(data.breast_quad.isnull().sum() / len(data))
#print(data.inv_nodes.isnull().sum() / len(data))



'''
jedan=data['breast-quad'].__eq__(1.0).sum()
dva=data['breast-quad'].__eq__(2.0).sum()
tri=data['breast-quad'].__eq__(3.0).sum()
cetiri=data['breast-quad'].__eq__(4.0).sum()
pet=data['breast-quad'].__eq__(5.0).sum()

print("1="+str(jedan)+", 2="+str(dva)+", 3="+str(tri)+", 4="+str(cetiri)+", 5="+str(pet))

nula=data['node-caps'].__eq__(0).sum()
jedan=data['node-caps'].__eq__(1).sum()
print("0="+str(nula)+", 1="+str(jedan))
'''

#popunjavanje missing values
data['node-caps']=data['node-caps'].fillna(0);
data['breast-quad']=data['breast-quad'].fillna(2);



#deljenje skupa podataka na trening i test skup
X = data[['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']]
Y= data[["class"]]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

Ytrain=np.ravel(Ytrain) 
Ytest = np.ravel(Ytest) 



#vizuelizacija kategorickih podataka
for i in col_names:
    ax=countplot(x=data[i], hue=data['class'])
    ax.legend(title='Klasa', loc='upper right')
    plt.tight_layout()
    plt.show()
    


def Metrika(TP,TN,FP,FN,klasa):
    print()
    print("klasa " + str(klasa) + ":")
    
    senzitivnost = TP/(TP+FN)
    specificnost = TN/(FP+TN)
    ppv = TP/(TP+FP) 
    npv = TN/(TN+FN)
    f1 = 2*(ppv*senzitivnost)/(ppv+senzitivnost)
    acc = (TP+TN)/(TP+FP+TN+FN)
    print("senzitivnost: " + str(round(senzitivnost,2)) + " specificnost: " + str(round(specificnost,2)))
    print("PPV: " + str(round(ppv,2)) + " NPV: " + str(round(npv,2)))
    print("f1 score: " + str(round(f1,2)))
    print("tacnost: " + str(round(acc,2)))
    


def UporediRez(Ytest,Ypredict,method):
    print()
    print("Dobijeni rezultati")
    print(Ypredict)
    print("Trazeni rezultati")
    print(Ytest)

    counter=0;
    
    for i in range (0,len(Ytest)):
        if(Ytest[i]!=Ypredict[i]):
            counter+=1
    
    
    print("Velicina tening skupa je: " + str(len(Ytrain)))
    print("Velicina test skupa je:" + str(len(Ytest)))
    print("Broj pogodaka: " + str(len(Ytest)-counter))
    print("Broj promasaja: " + str(counter))
    

    classes = np.unique(Ytest)
    
    fig, ax = plt.subplots()
    cm = metrics.confusion_matrix(Ytest, Ypredict, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Predicted", ylabel="True", title="confusion matrix: " + method)
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()
    
    #klasa 0
    TP=cm[0,0]
    TN=cm[1,1]
    FP=cm[0,1] 
    FN=cm[1,0]
    Metrika(TP,TN,FP,FN,0)
    #klasa 1
    TP=cm[1,1]
    TN=cm[0,0]
    FP=cm[1,0] 
    FN=cm[0,1]
    Metrika(TP,TN,FP,FN,1)


def LogistickaRegresija(Xtrain,Ytrain,Xtest,Ytest):
    print("\nLogisticka regresija: \n")

    log=LogisticRegression(class_weight={0:0.8,1:1}, C=0.6)
    log.fit(Xtrain,Ytrain)
    
    Ypredict=log.predict(Xtest)
    UporediRez(Ytest,Ypredict,"Logisticka regresija")
    
    
    
def StabloOdlucivanja(Xtrain,Ytrain,Xtest,Ytest):
    print("\nStablo odlucivanja: \n")
    
    dtc = DecisionTreeClassifier(min_samples_leaf=20, max_depth=5)
    dtc.fit(Xtrain,Ytrain)
    
    Ypredict = dtc.predict(Xtest)
    
    UporediRez(Ytest,Ypredict,"Stablo odlucivanja")
    
def RandomForest(Xtrain,Ytrain,Xtest,Ytest):
    print("\nRandom forest: \n")
    
    param_grid = {'max_depth':[10,20,30,40,50,None], 'max_features':['auto','sqrt'], 'min_samples_leaf':[1,2,3,4], 'n_estimators':[10,50,100,150]}
    
    rf = RandomForestClassifier()
    rf.fit(Xtrain,Ytrain)
    
    Ypredict = rf.predict(Xtest)
    
    UporediRez(Ytest,Ypredict,"Random forest")    
    
    grid = GridSearchCV(rf, param_grid, cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print("\ngrid rezultat:"+str(grid.best_params_)+"\n")
    
    depth=grid.best_params_.get('max_depth')
    features = grid.best_params_.get('max_features')
    leaf = grid.best_params_.get('min_samples_leaf')
    estim = grid.best_params_.get('n_estimators')
    
    grf = RandomForestClassifier(max_depth=depth, max_features=features, min_samples_leaf=leaf, n_estimators=estim)
    grf.fit(Xtrain,Ytrain)
    
    Ypredict=grf.predict(Xtest)
   
    UporediRez(Ytest,Ypredict,"Grid, Random forest")  
    
def NaivniBajes(Xtrain,Ytrain,Xtest,Ytest):
    print("\nNaivni bayes: \n")
    
    #priors=[0.5,0.5]
    gnb = GaussianNB()
    gnb.fit(Xtrain,Ytrain)
    
    Ypredict = gnb.predict(Xtest)
    
    UporediRez(Ytest,Ypredict,"Naivni Bayes")
    
    
def SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest):
    print("\nSupport vector machine: \n")
    
    param_grid={'kernel':['linear','rbf'], 'C':[0.1,0.25,0.5,0.75,1]} 
    
    svc = SVC()
    svc.fit(Xtrain,Ytrain)
    Ypredict = svc.predict(Xtest)
    
    UporediRez(Ytest,Ypredict,"Support Vector Machine")
    
    grid = GridSearchCV(svc,param_grid,cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print("\ngrid rezultat:"+str(grid.best_params_)+"\n")
    
    c=grid.best_params_.get('C')
    kernel=grid.best_params_.get('kernel')
    
    gsvc=SVC(C=c,kernel=kernel)
    gsvc.fit(Xtrain,Ytrain)
    
    Ypredict=gsvc.predict(Xtest)
   
    UporediRez(Ytest,Ypredict,"Grid, Support vector machine")
     
def KNN(Xtrain,Ytrain,Xtest,Ytest):
    print("\nK-nearest neighbors: \n")    
    
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(Xtrain,Ytrain)
    
    Ypredict = knn.predict(Xtest)
    
    UporediRez(Ytest,Ypredict,"KNearestNeighbors")


mlp = MLPClassifier(hidden_layer_sizes=(35,30,30), activation='relu', solver='adam', max_iter=5000)
mlp.fit(Xtrain,Ytrain)
Ypredict=mlp.predict(Xtest)
UporediRez(Ytest,Ypredict,"neuronska mreza")


#LogistickaRegresija(Xtrain,Ytrain,Xtest,Ytest)
#StabloOdlucivanja(Xtrain,Ytrain,Xtest,Ytest)
#RandomForest(Xtrain,Ytrain,Xtest,Ytest)
#NaivniBajes(Xtrain,Ytrain,Xtest,Ytest)
#SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest)
#KNN(Xtrain,Ytrain,Xtest,Ytest)

