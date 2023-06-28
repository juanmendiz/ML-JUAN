import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score, confusion_matrix,accuracy_score,precision_score, multilabel_confusion_matrix, f1_score
import pickle
import os


Test = pd.read_csv(os.path.join('data', "Test.csv"))

X_test = Test[['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic',
       'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts',
       'broad jump_cm']]
y_test = Test['class']

ruta_modelo = os.path.abspath(os.path.join('models','trained_model.pkl'))
with open(ruta_modelo, "rb") as archivo:
   modelo =  pickle.load(archivo)

y_pred = modelo.predict(X_test)

print("La Exactitud:",accuracy_score(y_test,y_pred))
print("\t")
print("Matriz de Confusion", multilabel_confusion_matrix(y_test,y_pred))

# print("Sensibilidad :",recall_score(y_test, y_pred))
# print("Precisión :",precision_score(y_test, y_pred))
# print("F1 Score", f1_score(y_test,y_pred))