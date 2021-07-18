from palmerpenguins import load_penguins
import random

#load_penguins() give me a pandas.DataFrame with all the data

class DataFetcher:
    def __init__(self):
        self.__FetchData()

    def GetPenguinsDataList(self):
        return self.__penguinsList

    def GetSampleList(self):
        self.__PickSample()
        return self.__penguinsSampleList

    def SetSampleRatio(self, sampleRatio):
        if sampleRatio < 60 or sampleRatio > 80:
            print('The range must be between 60 and 80 percent.')
        self.__sampleRatio = sampleRatio

    def __FetchData(self):
        penguins = load_penguins().dropna().values
        self.__penguinsList = penguins.tolist()

    def __CalculateDataSampleSize(self):
        listSize = len(self.__penguinsList)
        numberOfSamples = round((self.__sampleRatio / 100) * listSize)
        return numberOfSamples

    def __PickSample(self):
        self.__penguinsSampleList = random.sample(self.__penguinsList, self.__CalculateDataSampleSize())

    __penguinsSampleList = []
    __penguinsList = []
    __sampleRatio = 60