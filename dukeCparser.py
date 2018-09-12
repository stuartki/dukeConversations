import os
import numpy as np
import pandas as pd

directory = "/Users/stuartki/Downloads/Fall Sign Ups"
dataIndex = []
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        dtemp = pd.ExcelFile(directory + "/" + filename).parse(0, index = False)
        dtemp['prof'] = filename.split(' ')[-2].lower()
        dataIndex.append(dtemp)

data = pd.concat(dataIndex)

data.reset_index()
data.to_excel('test2.xlsx')