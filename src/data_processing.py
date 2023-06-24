# aqui hay que meter el procesado de los datos. Vamos a ello

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importo los datos de Github en Raw

url = "https://raw.githubusercontent.com/juanmendiz/ML-JUAN/master/data/Raw/bodyPerformance.csv"
df = pd.read_csv(url)

# Cambio con replace las etiquetas M y F por 0 y 1
clase_sexo = { "M":1,"F":0}
df['gender'] = df['gender'].replace(clase_sexo)

# y la clase Target en vez de letras a numeros siendo A la mejor y E la peor
clase_int = {"A":1,"B":2,"C":3,"D":4,"E":5}
df['class'] = df['class'].replace(clase_int)

# Guardo los datos procesados "levemente" en un csv dentro de data y en una variable. df_proc
ruta_archivo = os.path.join('data', 'Pocesados.csv')
df.to_csv(ruta_archivo, index=False)
df_proc = df

# Para ver el directorio
# print("Estamsos trabajando en ello! y aqui:", os.getcwd())



X_train, X_test, y_train, y_test = train_test_split(df_proc.drop('class',axis=1),
                                                     df_proc['class'],
                                                     test_size=0.2,
                                                     random_state=42)

ruta_archivo = os.path.join('data', 'Train.csv')
X_train.to_csv(ruta_archivo, index=False)

ruta_archivo = os.path.join('data', 'Test.csv')
X_test.to_csv(ruta_archivo, index=False)

ruta_archivo = os.path.join('data', 'y_train.csv')
y_train.to_csv(ruta_archivo, index=False)

"ESCALAR"

# OPCION 1 DE MinMax
# scaler = MinMaxScaler()
# columns = X.columns
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=columns)

# OPCION 2 de StandarScaler
# Almaceno en el objeto scaler todo lo necesario para estandarizar, con los datos de train
scaler = StandardScaler()
scaler.fit(X_train) # Utilizo los datos de train para escalar train y test.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

