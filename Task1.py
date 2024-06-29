# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# reading the train.csv file to get required training data
def training():
    data = pd.read_csv(r"D:\Task !\train.csv" )
    df = pd.DataFrame(data)
    squarefootage = df[['TotalBsmtSF','1stFlrSF','2ndFlrSF', 'GarageArea', 'PoolArea', 'WoodDeckSF', 'ScreenPorch', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch']]
    cols_to_sum = squarefootage.columns[:squarefootage.shape[1]]
    squarefootage['TotalArea'] = squarefootage[cols_to_sum].sum(axis=1)

    nbedroom = df['BedroomAbvGr']

    nbathroom = df[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']]
    nbathroom['TotalBathRooms'] = nbathroom[nbathroom.columns[:nbathroom.shape[1]]].sum(axis=1)

    IndepVar = pd.DataFrame()
    IndepVar['Bedroom'] = nbedroom
    IndepVar['Bathroom'] = nbathroom['TotalBathRooms']
    IndepVar['Area'] = squarefootage['TotalArea']

    DepVar = pd.DataFrame()
    DepVar = df['SalePrice']

    lr = LinearRegression()
    lr.fit(IndepVar, DepVar)
    
    return lr

# the testing function using data from test.csv
def testing(lr):
    data = pd.read_csv(r"D:\Task !\test.csv")
    df = pd.DataFrame(data)
    squarefootage = df[['TotalBsmtSF','1stFlrSF','2ndFlrSF', 'GarageArea', 'PoolArea', 'WoodDeckSF', 'ScreenPorch', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch']]
    cols_to_sum = squarefootage.columns[:squarefootage.shape[1]]
    squarefootage['TotalArea'] = squarefootage[cols_to_sum].sum(axis=1)

    nbedroom = df['BedroomAbvGr']
    
    nbathroom = df[['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']]
    nbathroom['TotalBathRooms'] = nbathroom[nbathroom.columns[:nbathroom.shape[1]]].sum(axis=1)

    IndepVar = pd.DataFrame()
    IndepVar['Bedroom'] = nbedroom
    IndepVar['Bathroom'] = nbathroom['TotalBathRooms']
    IndepVar['Area'] = squarefootage['TotalArea']
    DepVar = pd.DataFrame(index=range(1461, 2920))
    DepVar['SalesPrice'] = lr.predict(IndepVar)
    DepVar.index.name = 'Id'
    return DepVar

lr = training()
pred_Sales =  pd.DataFrame(testing(lr))
print(pred_Sales)