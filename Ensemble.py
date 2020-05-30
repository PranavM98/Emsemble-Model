#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:24:28 2020

@author: pranavmanjunath
"""

import encoding
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer




data=pd.read_csv('final_type.csv')
#BRANCH

lb_make = LabelEncoder()
data["Branch"] = lb_make.fit_transform(data["Branch"])
#Mechanical:3
#ISE:2
#CSE:0
#ECE:1


data["Admission_Entrance_Exam"]=lb_make.fit_transform(data["Admission_Entrance_Exam"])
#print(data[['Admission_Entrance_Exam1','Admission_Entrance_Exam']].head(11))
#CET:0
#COMEDK:1
#MANAGEMENT:2

data["Sem_1"]=lb_make.fit_transform(data["Sem_1"])
data["Sem_2"]=lb_make.fit_transform(data["Sem_2"])
data["Sem_3"]=lb_make.fit_transform(data["Sem_3"])
data["Sem_4"]=lb_make.fit_transform(data["Sem_4"])
data["Sem_5"]=lb_make.fit_transform(data["Sem_5"])
data["Sem_6"]=lb_make.fit_transform(data["Sem_6"])
data["Aggregate"]=lb_make.fit_transform(data["Aggregate"])
data["10th%"]=lb_make.fit_transform(data["10th%"])
data["12th%"]=lb_make.fit_transform(data["12th%"])


#FC:0
#FCD:1
#SC:2

#data["Sem_1"]=lb_make.fit_transform(data["Sem_1"])
#print(data[['Sem_11','Sem_1']].head(11))



lb_style = LabelBinarizer()
data["Gender"] = lb_style.fit_transform(data["Gender"])

#male:1
#female:0

X=data[['Admission_Entrance_Exam','Gender','Sem_1','Sem_2','Sem_3','Sem_4','Sem_5','Sem_6','Aggregate','Branch','10th%','12th%']]  # Features
y=data[['Company Type']]  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=126) # 70% training and 30% test

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

clf1=LogisticRegression(random_state=1)
clf2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()


labels= ['Logistic Regression','Random Forest', 'Naive Bayes']
'''
for clf,label in zip([clf1,clf2,clf3],labels):
    scores=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    
    print("Accuracy: %0.2f [%s]" % (scores.mean(), label))
'''    
    
voting_clf_hard = VotingClassifier(estimators = [(labels[0], clf1),
                                                 (labels[1], clf2),
                                                 (labels[2], clf3)],
                                                voting='hard')

voting_clf_soft = VotingClassifier(estimators = [(labels[0], clf1),
                                                 (labels[1], clf2),
                                                 (labels[2], clf3)],
                                                voting='soft')


labels_new = ['Logistic Regression','Random Forest', 'Naive Bayes', 'Voting_Classifier_Hard',
              'Voting_Classifier_Soft']

for (clf,label) in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    
    score=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (score.mean(), label))
