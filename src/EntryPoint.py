import numpy
from PenguinClassifier import PenginClassifier
import pandas as pd

def main():
    pc = PenginClassifier((150, 100, 50), 100, 'relu', 'adam', 0.6)
    pc.TrainNeuralNetwork()

    trainerData = pc.GetTrainerData()
    print("Los datos con los que se entrenara")
    print(trainerData)

    pc.PredictData()
    predictedData = pc.GetPredictionResults()
    print("Los datos predecidos son") # put all in a tuple array
    print(predictedData)

    trainerTuples = trainerData[0].tolist()
    trainerSpecies = trainerData[1].tolist()
    print(trainerTuples)
    print(trainerSpecies)

    trainerDf = pd.DataFrame(list(zip(trainerTuples, trainerSpecies)), columns=['Data', 'Specie'])
    print("Tabla de entrenmiento")
    print(trainerDf)

    expectedTrainer = pc.GetExpectedData()[0].tolist()
    expectedSpecies = pc.GetExpectedData()[1].tolist()
    speciesResult = pc.GetPredictionResults().tolist()
    trainerRes = pd.DataFrame(list(zip(expectedTrainer, expectedSpecies, speciesResult)), columns=['Datos', 'especie', 'resultado'])
    print("Tabla de resultados")
    print(trainerRes)

    accuracy = pc.GetAccuracyPercentage()
    print("El accuracy fue de: {:.4f}".format(accuracy))

if __name__ == "__main__":
    main()