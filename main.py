import csv
import os.path
import linear_regression

debug = False
print("** linear regression v1 **")
print('note: datafiles must be located in the data folder in root')


def loadData(filePath):
    csvFile = open(filePath)
    csvData = csv.reader(csvFile)
    
    dataList = []

    for row in csvData:
        dataList.append(row)
    csvFile.close()
    return dataList
    

gotfile = False
data = []
while not gotfile :    
    print("Type in the full filename of the csv datafile:")
  #  fileName = input("filename: ")
  #TODO: use input
    fileName = "day_length_weight.csv"
    if fileName.endswith(".csv") :

        fullPath = os.path.dirname(os.path.abspath(__file__))+'/data/' + fileName
        if os.path.isfile(fullPath) :
            gotfile = True
            print("loading " + fileName)
            data = loadData(fullPath)
        else :
            print("ERROR: " + fileName + " was not found in the data directory.")
    else : 
        print("ERROR: the file should be a .csv file")

linear_regression.performLinearRegression(data)



