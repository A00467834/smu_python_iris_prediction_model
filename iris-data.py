# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:19:29 2023

@author: ASUS
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

data = load_iris() #data

x = data['data']
y = data['target']

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2, random_state=44)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x,train_y)

pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y)

print(accuracy)

pred_y_train = clf.predict(train_x)
train_accuracy = accuracy_score(train_y, pred_y_train)
print(train_accuracy)

print(data['target_names'][pred_y])