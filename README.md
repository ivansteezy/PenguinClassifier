# ※ Clasificador de especies de Pingüinos

## ※ Descripción (TL;DR)
Este programa utiliza un modelo de machine learning en el cual se entrena una red neuronal dado un Dataset que contiene información relacionada a 3 especies de Pingüinos (Adelie, Chinstrap y Gentoo) y en base a la información correspondiente a el largo de su pico y el largo de su aleta hará una predicción para definir la especie de un pingüinos.

### ※ Preparación de datos
Para ello se hace uso de la clase de scikit-learn ```MLPClassifier``` definiendo parámetros como el número de capas ocultas, el número de iteraciones (o epochs), la función de activación y el algoritmo de optimización.

Posteriomente importamos los datos con los cuales se trabajarán, (en este caso solo nos interesan tres características; ```species```, ```bill_length_mm``` y ```flipper_length_mm```).

Con estos datos, se procede a hacer un escalado de características ([feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)) esto con el fin de estandarizar un rango para todos los datos y así, mejorar la precisión.

Posterior al escalado se procede a generar un _subconjunto_ de datos con los cuales se entrenará la red neuronal (el 70% de los datos en este caso).

Una vez se tiene el conjunto de datos de entrenamiento, se separan los datos en:
- La matriz de entrada ```<X>```, en este caso ```bill_length_mm``` y ```flipper_length_mm```.
- Los objetivos ```<y>```, en este caso ```species```.


### ※ Entrenamiento de la red.
Ya con todos los datos necesarios y en la forma necesaria únicamente queda entrenar a la red neuronal para posteriormente hacer una predicción, esto con el método ```MLPClassifier.fit()```

### ※ Prediccion y resultados
Al momento de ejecutar ```MLPClassifier.predict()```, este retornará una colección de predicciones. 

Para evaluar la precisión de los resultados se utiliza la [matriz de confusión](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n), la cual es generada a partir de las predicciones obtenidas y el método ```sklearn.metrics.confusion_matrix```. La precisión es determinada como el número de predicciones correctas dividido por el número total de predicciones.

Por último se genera un archivo csv que contiene los [datos de entrenamiento](output-data/training-data.csv) y un csv con las [predicciones obtenidas](output-data/predicted-data.csv).

## ※ Mas información
Si deseas saber más a profundidad respecto a como se entrenó la red neuronal y el como se implementó, existe este [Reporte](Reporte.md) el cual tiene información al respecto.
