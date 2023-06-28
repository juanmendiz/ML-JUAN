import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score
import os
import pickle

Train = pd.read_csv(os.path.join('data', "Train.csv"))

X = Train[['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic',
       'systolic', 'gripForce', 'sit and bend forward_cm', 'sit-ups counts',
       'broad jump_cm']]
y = Train['class']

seed = 0
scorer = make_scorer(recall_score, average='macro')
                     
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)

gbc_cv = cross_val_score(model, X, y, cv=10, scoring= scorer,error_score='raise')

model.fit(X, y)

print('CV',gbc_cv)
print('CV media',gbc_cv.mean())
print('CV desv',gbc_cv.std())

model_guardar = os.path.abspath(os.path.join('models', "trained_model.pkl"))
with open(model_guardar, "wb") as archivo:
    pickle.dump(model,archivo)

#Modelo GBC

# model = GradientBoostingClassifier(n_estimators=100, random_state=seed)

# gbc_cv = cross_val_score(model, X, y, cv=10, scoring="precision")

# print('CV',gbc_cv)
# print('CV media',gbc_cv.mean())
# print('CV desv',gbc_cv.std())



# Modelo Hiperparametrizado
# algun parametro que defina como multiclase 
# model = GradientBoostingClassifier()

# parameters = {"n_estimators":[50,100,150],
#               "max_depth":  [3,4,5],
#               "learning_rate": [0.1,0.5,0.9],
#               "min_samples_split": [4,8,12]}

# gb_gs = GridSearchCV(model, parameters, cv=5, scoring="precision")

# gb_gs.fit(X, y)

# # hay que guardar las variables no impirmirlas 
# print("Best Score", gb_gs.best_score_)
# print("\t")
# print("Best Params", gb_gs.best_params_)
# print("\t")
# print("Best Estimators", gb_gs.best_estimator_)