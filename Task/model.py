from numpy import number
import pandas as pd
import matplotlib.pyplot as plt


# Read excel file into a dictionary of pandas dataframes
data = pd.read_excel("data.xlsx", sheet_name=[0,1,2,3,4,5])
rendita_azioni = data[3] #Selecting each dataframe, corresponding to eaech sheet in the excel file
print(rendita_azioni.head(3))
print(" ")
anums = rendita_azioni.iloc[0:20,1:3]

print(anums.value_counts(normalize=True))