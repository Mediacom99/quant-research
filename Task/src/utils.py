import pandas as pd
from sklearn.preprocessing import StandardScaler

#Reads sheets of excel file, set datetime format and 
#date column as index. Returns collection of dataframes,
# one for each sheet of the xlsx file.
def get_data_from_excel(file_name):
    """Load data from excel file with multiple sheets

    Args:
        file_name (str): relative or absolute path of excel file to load

    Returns:
        {sheet_name:pd.DataFrame,...}: returns a collection of pandas dataframe, each dataframe
                            corresponds to a sheet in the excel file.
    """
    print("INFO: Loading data from excel file")
    #Get Formatted Data into 
    xls = pd.ExcelFile(file_name)
    sheet_names = xls.sheet_names
    data = {}
    for sheet_name in sheet_names:
        data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    for df in data:
            data[df]['Date'] = pd.to_datetime(data[df]['Date']) #Format Date to datetime format
            data[df].set_index('Date', inplace=True) #Set Date column as index
        
    return data

def count_nans(collection):
    """
    Count number of NaN in the collection of dataframes passed as input. Prints the number 
    for each column, for each dataframe on stdout.
    """
    
    for df in collection:
        for cols_name, series in collection[df].items():
            print(f"Num of NaNs in {cols_name}: ", series.isna().sum())
    
    return

def normalize_dataframe(dataframe):
    """Normalize dataframe values for mean zero and variance one. It uses StandardScaler
       from sklearn. It does not change the original dataframe.

    Args:
        dataframe (pandas.DataFrame): pandas dataframe to normalize

    Returns:
        pandas.DataFrame: normalized dataframe.
    """
    scaler = StandardScaler()
    df = dataframe.copy()
    for cols_name, series in df.items():
        df[cols_name] = pd.Series(scaler.fit_transform(series.array.reshape(-1,1)).flatten(), index=df.index);
    
    return df

def divide_dataframe_lastyear(dataframe):
    """Divide dataframe into two dataframes, cutting the original one at a certain date

    Args:
        dataframe (pd.DataFrame): original dataframe to cut

    Returns:
        (pd.DataFrame, pd.DataFrame): returns a tuple like (training DataFrame, testing DataFrame) 
    """
    training = dataframe[:'2018-12-31']
    testing = dataframe['2019-01-01':]
    return (training, testing)