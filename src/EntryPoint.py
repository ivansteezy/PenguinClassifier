from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from DataFetcher import DataFetcher

def accuracy(confusion_matrix):
    diagonalSum = confusion_matrix.trace()
    sumOfAllElements = confusion_matrix.sum()
    return diagonalSum / sumOfAllElements


def main():
    d = DataFetcher()
    d.SetSampleRatio(60)

    print("Los pinguinos cargados son: antes de escalar\n")
    allData = d.GetPenguinsDataList()    
    print(allData)

    scaler = StandardScaler()
    allData[['bill_length_mm','flipper_length_mm']] = scaler.fit_transform(allData[['bill_length_mm','flipper_length_mm']])
    print("Los pinguinos cargados son: despues de escalar\n")
    print(allData)

    #Splitting the dataset into training and validation sets
    trainingSet, validationSet = train_test_split(allData, test_size=0.6, random_state=21)
    print("\n\n\n")
    print("El dataset de entrenamiento")
    print(trainingSet)

    xTrainer = trainingSet[['bill_length_mm', 'flipper_length_mm']].values
    yTrainer = trainingSet['species'].values

    #maybe change names by an id, or switch name to ytrainer
    print("Trainers X")
    print(xTrainer)

    print("Trainer Y")
    print(yTrainer)

    xVal = trainingSet[['bill_length_mm', 'flipper_length_mm']].values
    yVal = trainingSet['species'].values

    #set up algorithm and function to classify
    classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=100, activation='relu', solver='adam', random_state=1)

    # train the neural network
    classifier.fit(xTrainer, yTrainer)

    #predict y for x val
    yPred = classifier.predict(xVal)

    print("Resultados predecidos")
    print(yPred)

    print("Resultados")
    print(yVal)

    #Print results
    cm = confusion_matrix(yPred, yVal)
    print("EL Accuracy fue de: ", accuracy(cm))

if __name__ == "__main__":
    main()