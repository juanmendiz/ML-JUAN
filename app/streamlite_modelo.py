#importar las librerias necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import pickle
import os

#cargar los datos en un dataset
Train = pd.read_csv(os.path.join('../data', 'Train.csv'))
Test = pd.read_csv(os.path.join('../data', 'Test.csv'))

X_test = Test[['weight_kg','gender', 'age', 'diastolic', 'systolic', 'height_cm', 'body fat_%','gripForce', 'sit and bend forward_cm', 'sit-ups counts',
       'broad jump_cm']]
y_test = Test['class']

X_train = Train[['weight_kg','gender', 'age', 'diastolic', 'systolic', 'height_cm', 'body fat_%','gripForce', 'sit and bend forward_cm', 'sit-ups counts',
       'broad jump_cm']]

y_train = Train['class']

#separar los datos en entrenamiento y prueba


lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_m = SVC()

#entrenar modelos
lin_regr = lin_reg.fit(X_train, y_train)
log_regr = log_reg.fit(X_train, y_train)
svc_mo = svc_m.fit(X_train, y_train)
print( "todo ok")

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_mo, sv)