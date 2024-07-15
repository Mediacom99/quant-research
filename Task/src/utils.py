import pandas as pd

#Reads sheets of excel file, set datetime format and 
#date column as index. Returns collection of dataframes,
# one for each sheet of the xlsx file.
def get_data_from_excel(file_name):
    #Get Formatted Data into 
    xls = pd.ExcelFile(file_name)
    sheet_names = xls.sheet_names
    data = {}
    
    for sheet_name in sheet_names:
        data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    
            
    for df in data:
        try:
            data[df]['Date'] = pd.to_datetime(data[df]['Date']) #Format Date to datetime format
            data[df].set_index('Date', inplace=True) #Set Date column as index
        except:
            print("Date column is already in datetime format and used as index!")
        
    return data
