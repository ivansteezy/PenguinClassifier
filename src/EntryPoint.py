from scipy.sparse import data
from PenguinClassifier import PenginClassifier
import pandas as pd


def ExportDataAsCsv(data, path):
    data.to_csv(path_or_buf=path, index=False)

def main():
    pc = PenginClassifier((150, 100, 50), 100, 'relu', 'adam', 0.6)
    pc.TrainNeuralNetwork()

    trainerData = pc.GetTrainerData()
    pc.PredictData()

    ##################################
    ala, pico = trainerData[0].T #unscale this
    trainerSpecies = trainerData[1].tolist()

    trainerDf = pd.DataFrame(list(zip(ala, pico, trainerSpecies)), columns=['ala', 'pico', 'Specie'])
    print("Tabla de entrenmiento")
    print(trainerDf)
    ExportDataAsCsv(trainerDf, "output-data/training-data.csv")

    expectedTrainer = pc.GetExpectedData()[0].tolist()
    alaRes,PicoRes = pc.GetExpectedData()[1].T #unscale this
    speciesResult = pc.GetPredictionResults().tolist()
    trainerRes = pd.DataFrame(list(zip(alaRes, PicoRes, expectedTrainer, speciesResult)), columns=['Ala', 'Pico', 'ResEsp', 'Res'])
    print("Tabla de resultados")
    print(trainerRes)
    ExportDataAsCsv(trainerRes, "output-data/predicted-data.csv")
    ######################################
    
    accuracy = pc.GetAccuracyPercentage()
    print("El accuracy fue de: {:.4f}%".format(accuracy))

if __name__ == "__main__":
    main()