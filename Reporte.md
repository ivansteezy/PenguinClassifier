# ※ Clasificador de especies de Pingüinos

Este programa utiliza un modelo de _machine learning_ en el cual se entrena una _red neuronal_ dado un _Dataset_ que contiene información relacionada a 3 especies de Pingüinos (Adelie, Chinstrap y Gentoo) asi como sus caracteristicas como el largo de su pico y el largo de sus aletas.

La libreria utilizada es [scikit-learn](https://scikit-learn.org/stable/install.html), la red neurnal usada pertenece al modulo [sklearn.neural_network](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) y el _dataset_ usado es [palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

## ※ Codigo

La clase encargada de realizar la clasificacion se encuentra en el archivo [PenguinClassifier.py](https://github.com/ivansteezy/PenguinClassifier/blob/main/src/PenguinClassifier.py) la cual tiene la funcion de ser un _wrapper_ para los parametros necestarios para construir una instancia de la clase [```MLPClassifier```](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) que es la encargada de realizar la tarea requerida.

La clase ```MLPClassifier``` implementa un perceptrón multicapa, el cual se entrena utilizando el algoritmo de [Propagación hacia atrás (Backpropagation)](https://es.wikipedia.org/wiki/Propagaci%C3%B3n_hacia_atr%C3%A1s).

Se puede crear una instacia _via_ constructor de ```PenguinClassifier``` es tal que:

```py
PenginClassifier((50, 50, 50), 100, 'relu', 'lbfgs', 0.3)
```

A grandes rasgos se cuentan con 5 parametros:

- Primero se define el numero de [_capas ocultas_]() con las que contara la red neuronal, en este caso 3 capas de 50 nodos cada una.

<center>
<img src="https://png.pngitem.com/pimgs/s/417-4176751_english-neural-feed-forward-network-for-machine-translation.png" height=150 style="display: block; margin: auto;">
<em>Capas ocultas en una red neuronal</em>
</center>
<br>

- El segundo parametro es el numero de iteraciones (o [epochs](https://radiopaedia.org/articles/epoch-machine-learning)) que se desea que ejecute. Una iteracion corresponde al ciclo compuesto por la "alimentacion hacia adelante" (feed-forward) y la propagacion hacia atras(backpropagation).

- El tercero es la funcion de activacion para las capas ocultas, hay 4 opciones:
    - 'identity', retorna ```f(x) = x```
    - 'logistic', retorna ```f(x) = 1 / (1 + exp(-x))```
    - 'tanh', retorna ```f(x) = tanh(x)```
    - 'relu', retorna ```f(x) = max(0, x)```

    para esta tarea se utilizo _relu_ (es muy facil de procesar, es la opcion por defecto tambien)

- El cuarto es el algoritmo de optimizacion, sklearn da 3 opciones:
    - 'lbfgs' es un optimizador de la familia de los metodos quasi-Newton
    - 'sgd' es un deceso gradiente estocástico
    - 'adam' es una extension del anterior

    Por defecto, sklearn utiliza la opcion de _adam_, sin embargo, en la documencion oficial hace un parentesis destacando que adam funciona bien en cojuntos de datos relativamente grandes (miles de datos de entramiento o mas, por ejemplo) y que para pequeños conjuntos de datos 'lbfgs' se desempeña mejor, por lo que fue el que se uso en esta tarea.

- Y por ultimo el porcentaje con el cual se quieren tomar los datos, en este caso el 70% (representando en un rango 0.0 a 1.0, siendo por ejemplo 0.3 el 70%, 0.2 el 80%, etc).

### ※ Preparacion de datos
Primero, necesitamos hacer la importacion de los datos con los cuales se trabajara.

```py
from palmerpenguins import load_penguins

class PenguinClassifier:
    #...
    # obtenemos y sanitizamos los datos
    def __FetchData(self):
    self.__rawData = load_penguins().dropna()[['species', 'bill_length_mm','flipper_length_mm']]
    self.__ScaleData()

    # escalamos los datos a un nuevo rango
    def __ScaleData(self):
        self.__scaler = StandardScaler()
        self.__rawData[['bill_length_mm','flipper_length_mm']] = self.__scaler.fit_transform(self.__rawData[['bill_length_mm','flipper_length_mm']])
```

Cargamos los datos con la funcion ```load_penguins()``` y eliminamos los valores no existentes con ```dropna()```, para esta tarea utilizaremos unicamente las caracteristicas ```species```, ```bill_length_mm``` y ```flipper_length_mm```, para justo despues escalar los datos a un rango estandarizado para aumentar la precision.

En este punto los datos se ven algo asi: 
<center>
<img src="img/scaled-data.png" height=200>
</center>

### ※Entrenamiento de la red neuronal

```py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class PenguinClassifier:
    #...

    # Se entrena la red neuronal tomando una porcion de los datos al azar
    # y se dividen en dos colecciones, una que contiene 'bill_length_mm' y 'flipper_length_mm'
    # y otra que contiene 'species'
    def TrainNeuralNetwork(self):
        trainingSet = train_test_split(self.__rawData, test_size=self.__trainingDataSetSize, random_state=20)
        self.__yTrainer = trainingSet[0]['species'].values
        self.__xTrainer = trainingSet[0][['bill_length_mm', 'flipper_length_mm']].values
        self.__SetExpectedResults()
        self.__classifier = MLPClassifier(hidden_layer_sizes=self.__hiddenLayers,
                                            max_iter=self.__maxIterations,
                                            activation=self.__activationFun, 
                                            solver=self.__solver, random_state=21)
        self.__classifier.fit(self.__xTrainer, self.__yTrainer)
```