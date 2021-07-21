from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from palmerpenguins import load_penguins

class PenginClassifier:
    def __init__(self, hiddenLayers, maxIterations, activationFunc, solver, trainingDataSetSize):
        self.__hiddenLayers = hiddenLayers
        self.__maxIterations = maxIterations
        self.__activationFun = activationFunc
        self.__solver = solver
        self.__trainingDataSetSize = trainingDataSetSize

        self.__FetchData()
        

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

    def PredictData(self):
        self.__predictionResults = self.__classifier.predict(self.__xValues)
        confusionMatrix = confusion_matrix(self.__predictionResults, self.__yValues)
        self.__CalculateAccuracyPercentage(confusionMatrix)

    def __FetchData(self):
        self.__rawData = load_penguins().dropna()[['species', 'bill_length_mm','flipper_length_mm']]
        self.__ScaleData()
        
    def __ScaleData(self):
        scaler = StandardScaler()
        self.__rawData[['bill_length_mm','flipper_length_mm']] = scaler.fit_transform(self.__rawData[['bill_length_mm','flipper_length_mm']])

    def __SetExpectedResults(self):
        self.__xValues = self.__rawData[['bill_length_mm', 'flipper_length_mm']].values
        self.__yValues = self.__rawData['species'].values

    def __CalculateAccuracyPercentage(self, confusionMatrix):
        diagonalSum = confusionMatrix.trace()
        sumOfAllElements = confusionMatrix.sum()
        self.__accuracyPercentage = (diagonalSum / sumOfAllElements) * 100

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
        return (self.__xTrainer, self.__yTrainer)

    def GetExpectedData(self):
        return self.__xValues.concat(self.__yValues)

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
