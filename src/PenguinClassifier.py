from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from palmerpenguins import load_penguins

class PenginClassifier:
    # Se inicializan los parametros que se utilizaran
    def __init__(self, hiddenLayers, maxIterations, activationFunc, solver, trainingDataSetSize):
        self.__hiddenLayers = hiddenLayers
        self.__maxIterations = maxIterations
        self.__activationFun = activationFunc
        self.__solver = solver
        self.__trainingDataSetSize = trainingDataSetSize

        self.__FetchData()
        
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

    # Se hace una prediccion y despues se escala de regreso al rango original de los datos,
    # se genera la matriz de confusion para calcular la precisión
    def PredictData(self):
        self.__predictionResults = self.__classifier.predict(self.__xValues)
        self.__xValues = self.__scaler.inverse_transform(self.__xValues)
        confusionMatrix = confusion_matrix(self.__predictionResults, self.__yValues)
        self.__CalculateAccuracyPercentage(confusionMatrix)

    # Se obtienen los datos del dataset y se sanitizan los valores
    def __FetchData(self):
        self.__rawData = load_penguins().dropna()[['species', 'bill_length_mm','flipper_length_mm']]
        self.__ScaleData()
        
    # Se escalan los datos de la longitud via StandardScaler
    def __ScaleData(self):
        self.__scaler = StandardScaler()
        self.__rawData[['bill_length_mm','flipper_length_mm']] = self.__scaler.fit_transform(self.__rawData[['bill_length_mm','flipper_length_mm']])

    # Guardamos los datos con los que se comparara el resultado
    def __SetExpectedResults(self):
        self.__xValues = self.__rawData[['bill_length_mm', 'flipper_length_mm']].values
        self.__yValues = self.__rawData['species'].values

    # Se calcula el porcentaje de precision dada una matriz de confusion
    def __CalculateAccuracyPercentage(self, confusionMatrix):
        diagonalSum = confusionMatrix.trace()
        sumOfAllElements = confusionMatrix.sum()
        self.__accuracyPercentage = (diagonalSum / sumOfAllElements) * 100

    # Encapsulamiento de los datos
    def SetTrainerSize(self, trainerSize):
        self.__trainingDataSetSize = trainerSize

    def SetMaxIterations(self, maxIterations):
        self.__maxIterations = maxIterations

    def SetActivationFunction(self, activationFunc):
        self.__activationFun = activationFunc

    def GetPredictionResults(self):
        return self.__predictionResults
    
    def GetAccuracyPercentage(self):
        return self.__accuracyPercentage

    def GetTrainerData(self):
        self.__xTrainer = self.__scaler.inverse_transform(self.__xTrainer)
        return (self.__xTrainer, self.__yTrainer)

    def GetExpectedData(self):
        return (self.__yValues, self.__xValues)

    def GetRawData(self):
        return self.__rawData

    __xTrainer = None
    __yTrainer = None

    __xValues = None
    __yValues = None

    __predictionResults = None 
    __accuracyPercentage = 0.0

    __rawData = None

    __hiddenLayers = (0, 0, 0)
    __trainingDataSetSize = 0
    __maxIterations = 10
    __activationFun = ''
    __solver = ''

    __classifier = None
    __scaler = None
