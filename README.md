# ※ Clasificador de especies de Pingüinos

## ※ Descripcion (TL;DR)
Este programa utiliza un modelo de machine learning en el cual se entrena una red neuronal dado un Dataset que contiene información relacionada a 3 especies de Pingüinos (Adelie, Chinstrap y Gentoo) y en base a la informacion correspondiente a el largo de su pico y el largo de su aleta hara una prediccion para definir la especie de un pingüinos.

### ※ Preparacion de datos
Para ello se hace uso de la clase de scikit-learn ```MLPClassifier``` definiendo parametros como el numero de capas ocultas, el numero de iteracion (o epochs), la funcion de activacion y el algoritmo de optimizacion.

Posteriomente importamos los datos con los cuales se trabajaran, (en este caso solo nos interesan tres caracteristicas; ```species```, ```bill_length_mm``` y ```flipper_length_mm```).

Con estos datos, se procede a hacer un escalado de caracteristicas ([feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)) esto con el fin de estandarizar un rango para todos los datos y asi, mejorar la precision.

Posterior al escalado se procede a generar un _subconjunto_ de datos con los cuales se entrenara la red neuronal (el 70% de los datos en este caso).

Una vez se tiene el conjunto de datos de entrenamiento, separamos los datos en:
- La matriz de entrada ```<X>```, en este caso ```bill_length_mm``` y ```flipper_length_mm```.
- Los objetivos ```<y>```, en este caso ```species```.


### ※ Entrenamiento de la red.
Ya con todos los datos necesarios y en la forma necesaria unicamnte queda entrenar a la red neuronal para posteriormente hacer una prediccion, esto con el metodo ```MLPClassifier.fit()```

### ※ Prediccion y resultados
Al momento de ejecutar ```MLPClassifier.predict()```, este retornara una coleccion de predicciones. 

Para evaluar la precision de los resultados se utiliza la [matriz de confusion](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n), la cual es generada a partir de las predicciones obtenidas y el metodo ```sklearn.metrics.confusion_matrix```. La precision es determinada como el numero de predicciones correctas dividido por el numero total de predicciones.

Por ultimo se genera un archivo csv que contiene los [datos de entrenamiento](output-data/training-data.csv) y un csv con las [predicciones obtenidas](output-data/predicted-data.csv).

## ※ Mas informacion
Si deseas saber mas a profundidad respecto a como se entreno la red neuronal y el como se implemento, existe este [Reporte](Reporte.md) el cual tiene informacion al respecto.