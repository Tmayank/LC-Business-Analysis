# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:02:17 2020

@author: mayan
"""


import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt

data = pd.read_csv("Lending_Data.csv")
data.head(5)

data.dtypes

# Dropping irrelevant columns
data = data.drop(['purpose', 'home_ownership', 'verification_status'], axis=1)
data.head(5)

df.shape

duplicate_rows_df = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)

data.count()
data = data.drop_duplicates()

data = data.dropna()
data.count()

X = data[['loan_amnt','int_rate' , 'installment' , 'annual_inc' , 'revol_bal']]
y = data['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
classifierTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierTree.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred2 = classifierTree.predict(X_test)
cm = confusion_matrix(y_test, y_pred2)
print(cm)
accuracy_score(y_test, y_pred2)

data['loss'] = data['total_pymnt'] - data['loan_amnt']

X1 = data[['revol_util','int_rate' , 'installment' , 'annual_inc' , 'revol_bal']]
y1 = data['loss']

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X1_train, y1_train)

y_pred3 = regressor.predict(X1_test)
np.set_printoptions(precision=2)

from sklearn.metrics import r2_score
r2_score(y1_test, y_pred3)

from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state = 0)
regressor1.fit(X1_train, y1_train)

y_pred4 = regressor1.predict(X1_test)

from sklearn.metrics import r2_score

