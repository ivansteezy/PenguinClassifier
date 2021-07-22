# ※ Clasificador de especies de Pingüinos

Este programa utiliza un modelo de _machine learning_ en el cual se entrena una _red neuronal_ dado un _Dataset_ que contiene información relacionada a 3 especies de Pingüinos (Adelie, Chinstrap y Gentoo) asi como sus caracteristicas como el largo de su pico y el largo de sus aletas.

La libreria utilizada es [scikit-learn](https://scikit-learn.org/stable/install.html), la red neurnal usada pertenece al modulo [sklearn.neural_network](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) y el _dataset_ usado es [palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

## ※ Codigo

La clase encargada de realizar la clasificacion se encuentra en el archivo [PenguinClassifier.py](https://github.com/ivansteezy/PenguinClassifier/blob/main/src/PenguinClassifier.py) la cual tiene la funcion de ser un _wrapper_ para los parametros necestarios para construir una instancia de la clase [```MLPClassifier```](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) que es la encargada de realizar la tarea requerida.

La clase ```MLPClassifier``` implementa un perceptrón multicapa, el cual se entrena utilizando el algoritmo de [Propagación hacia atrás (Backpropagation)](https://es.wikipedia.org/wiki/Propagaci%C3%B3n_hacia_atr%C3%A1s).

El constructor de ```PenguinClassifier``` es tal que:

```py
    def __init__(self, hiddenLayers, maxIterations, activationFunc, solver, trainingDataSetSize):
        self.__hiddenLayers = hiddenLayers
        self.__maxIterations = maxIterations
        self.__activationFun = activationFunc
        self.__solver = solver
        self.__trainingDataSetSize = trainingDataSetSize

        self.__FetchData()
```


Se crea una instancia tal que: 

```py
PenginClassifier((50, 50, 50), 100, 'relu', 'adam', 0.3)
```

Primero se define el numero de _capas ocultas_ con las que contara la red neuronal, en este caso 3 capas de 50 nodos cada una.

El segundo parametro es el numero de iteraciones (o [epochs](https://radiopaedia.org/articles/epoch-machine-learning)) que se desea que ejecute. Una iteracion corresponde al ciclo compuesto por la "alimentacion hacia adelante" (feed-forward) y la propagacion hacia atras(backpropagation).