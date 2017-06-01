#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""

# Code source: Jaques Grobler
# License: BSD 3 clause


import numpy as np
from sklearn import linear_model


# Load the diabetes dataset
# diabetes = datasets.load_diabetes()

# Use only one feature
# diabetes_X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 16))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = listFromLine[0:16]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('E:\Github\jan2048\data\data.txt')
datingDataMat2, datingLabels2 = file2matrix('E:\Github\jan2048\data\data2.txt')
datingDataMat3, datingLabels3 = file2matrix('E:\Github\jan2048\data\data3.txt')
datingDataMat4, datingLabels4 = file2matrix('E:\Github\jan2048\data\data4.txt')
datingDataMat5, datingLabels5 = file2matrix('E:\Github\jan2048\data\data5.txt')
diabetes_X_train = np.append(datingDataMat, datingDataMat2, axis=0)
diabetes_X_train = np.append(diabetes_X_train, datingDataMat3, axis=0)
diabetes_X_train = np.append(diabetes_X_train, datingDataMat4, axis=0)
diabetes_X_train = np.append(diabetes_X_train, datingDataMat5, axis=0)
diabetes_y_train = np.append(datingLabels, datingLabels2, axis=0)
diabetes_y_train = np.append(diabetes_y_train, datingLabels3, axis=0)
diabetes_y_train = np.append(diabetes_y_train, datingLabels4, axis=0)
diabetes_y_train = np.append(diabetes_y_train, datingLabels5, axis=0)
# diabetes_X_test = [[4, 4], [5, 5]]
# diabetes_y_test = [4, 5]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

print('Coefficients: \n', regr.coef_)
# print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))
print(regr.predict([[16, 0, 4, 0, 16, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 32, 0, 0, 0, 4, 0, 0, 2, 2, 0, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 2, 2, 2],
                    [32, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]]))
