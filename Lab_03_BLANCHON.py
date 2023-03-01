#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:13:11 2023

@author: chloe
"""

""" Exercice 1 """
print(" ----------- Exercice 1 ----------- ")

# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

#We load IRIS dataset 
from sklearn.datasets import load_iris
#We store it in the iris variable
iris=load_iris()    
#An array that contains the names of the four features in the Iris dataset.                             
iris.feature_names

#Prints out the names of the features
print("Names of the features", iris.feature_names)
#Prints out the first five rows of the data matrix
print("First five rows", iris.data[0:5,:])
#Prints out the first five values of the target variable
print("First five target variable", iris.target[0:5])
#print(iris.data)

#We split data into training and testing parts:
from sklearn.model_selection import train_test_split
#We assign the data matrix from the Iris dataset to the X variable.
X=iris.data[iris.target!=2,0:2]
#We assign the data matrix from the Iris dataset to the y variable.
y=iris.target[iris.target!=2]
#Training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#The training matrix will have 120 rows and 4 columns
#The test matrix will have 30 rows and 4 columns
print(" X train shape", X_train.shape)
print("X test shape", X_test.shape)

#We use a Support Vector Machine for classification:
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
print("SVM Model score", SVMmodel.score(X_test,y_test))
print("SVM Model parameters", SVMmodel.get_params())

#We choose only first two features (columns) of iris.data' and we eliminate iris.target =2 from the data


supvectors=SVMmodel.support_vectors_
print("Supvectors shape", supvectors.shape)
print(" X train shape", X_train.shape)
print("Y train shape", y_train.shape)

print("Supvectors", supvectors)
#We plot scatterplots of targets 0 and 1 and check the separability of the classes:
plt.scatter(X[y==0,0],X[y==0,1],color='green')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.scatter(X[y==2,0],X[y==2,1],color='cyan')
plt.scatter(supvectors[:,0],supvectors[:,1],color='red',marker='+',s=50)

W=SVMmodel.coef_
b=SVMmodel.intercept_
xgr=np.linspace(min(X[:,0]),max(X[:,0]),100)



print(W[:,0])
print(W[:,1])
print(b)
ygr=-W[:,0]/W[:,1]*xgr-b/W[:,1]
plt.scatter(xgr,ygr)



""" Exercice 2 """
print(" ----------- Exercice 2 ----------- ")

from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

plt.figure()
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

plt.scatter(x[:,0], x[:,1])
plt.show()


SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)

SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()






# pred=SVMmodel.predict(X_test)
# print(classification_report(y_test, pred))

# cm = confusion_matrix(y_test, pred)
# print(cm)

# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = SVMmodel, X = X_train, y = y_train, cv = 10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))































# from sklearn.svm import SVC
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm, datasets
# iris = datasets.load_iris()# Select 2 features / variables
# X = iris.data[:, :2] # we only take the first two features.
# y = iris.target
# feature_names = iris.feature_names[:2]
# classes = iris.target_names
# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy
# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
# # The classification SVC model
# model = svm.SVC(kernel="linear")
# clf = model.fit(X, y)
# fig, ax = plt.subplots()
# # title for the plots
# title = ("Decision surface of linear SVC ")
# # Set-up grid for plotting.
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)
# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
# ax.set_ylabel("{}".format(feature_names[0]))
# ax.set_xlabel("{}".format(feature_names[1]))
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# plt.show()


# lignes 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import svm
# np.random.seed(2)
# # we create 40 linearly separable points
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20
# # fit the model
# clf = svm.SVC(kernel="linear", C=1)
# clf.fit(X, Y)
# # get the separating hyperplane
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-5, 5)
# yy = a * xx - (clf.intercept_[0]) / w[1]
# margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
# yy_down = yy - np.sqrt(1 + a ** 2) * margin
# yy_up = yy + np.sqrt(1 + a ** 2) * margin
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.plot(xx, yy, "k-")
# plt.plot(xx, yy_down, "k-")
# plt.plot(xx, yy_up, "k-")
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#  facecolors="none", zorder=10, edgecolors="k")
# plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
#  edgecolors="k")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()











