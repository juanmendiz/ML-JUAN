# ML-JUAN
# Evaluar físicamente nuevos usuarios de un gimnasio

Datos para calcular el grado de rendimiento con la edad y algunos datos de rendimiento del ejercicio. Saber cuanto tiempo vamos a tener que dedicar a cada usuario y en funcion de eso tener mas o menos personal contratado.

Se crea un modelo de Gradiand Boosting Classifier con un data set de unas 14000  muestras de personas con unas caracteristicas fisicas y una clasificacion segun esten o no en forma.

-Edad : 20 ~ 64

-Sexo : F,M

-Altura cm :

-Peso kg

-Grasa_corporal %

-Presion sanguinea Diastolica (min)

-Presión sanguinea sistólica (min)

-Fuerza de agarre

-Flexibilidad_cm

-Abdominales (numero)

-salto longitud_cm

-class : **A,B,C,D** ( A: la mas favorable) 


----------

En la carpeta src se encuentran los scripts de Python;  

        **data_procesing.py** donde se carga la base de datos y se modifica. De aqui se generan los csv Procesados.csv, Train.cv y Test.csv que se guardan en la carperta data.

        **model.py** rentrena el modelo GBC con los datos generados en data_procesing.py y guarda el resultao del modelo en pickel en la carpeta models.

        **evaluation.py** carga de pickel el resultado de modelo y Test.csv para hacer la evaluacion .