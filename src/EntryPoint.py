from random import sample
from DataFetcher import DataFetcher

def main():
    d = DataFetcher()
    d.SetSampleRatio(60)

    print("Los pinguinos cargados son: \n")
    allData = d.GetPenguinsDataList()
    for i in range(5):
        print(allData[i])
        print("\n")

    print("El sample de los datos es: \n")
    sample = d.GetSampleList()
    for i in range(5):
        print(sample[i])
        print("\n")

if __name__ == "__main__":
    main()