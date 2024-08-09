import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy import abs
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

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
    values = scaler.fit_transform(df)
    df = pd.DataFrame(values, columns=df.columns, index=dataframe.index)
    return df

#TODO Generalize date
def divide_df_lastyear(df):
    """Divide dataframe into two dataframes, cutting the original one at a certain date

    Args:
        dataframe (pd.DataFrame): original dataframe to cut

    Returns:
        (pd.DataFrame, pd.DataFrame): returns a tuple like (training DataFrame, testing DataFrame) 
    """
    training = df[:'2018-12-31']
    testing = df['2019-01-01':]
    return (training, testing)

#Clean cov matrix by replacing extremely small values with zeroes
def clean_cov_matrix(df, threshold):
    # Calculate the covariance matrix
    cov_matrix = df.cov()
    
    # Create a mask for values smaller than the threshold
    mask = abs(cov_matrix) < threshold
    
    # Replace values smaller than the threshold with 0
    cov_matrix[mask] = 0
    
    return cov_matrix

def force_diagonal_cov(df) -> pd.DataFrame:
    
    cov_matrix = df.cov()
    
    identity_matrix = np.eye(cov_matrix.shape[0])
    
    return cov_matrix * identity_matrix


#Plot results of normality tests for 5 stocks in a single image
def five_fig_plot(df:pd.DataFrame):
    
    # Set up the plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes for easier indexing

    # Colors for each stock
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Plot for each stock
    for i, column in enumerate(df.columns):
        data = df[column]
        
        # Fit a normal distribution to the data
        mu, std = stats.norm.fit(data)
        print(mu,std)
        
        # Plot the histogram
        axs[i].hist(data, bins=250, density=True, alpha=0.7, color=colors[i])
        
        # Plot the PDF
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 500)
        p = stats.norm.pdf(x, mu, std)
        axs[i].plot(x, p, 'k', linewidth=2)
        
        axs[i].set_title(f'{column}')
        axs[i].set_xlabel('Returns')
        axs[i].set_ylabel('Density')

    # Plot all stocks together
    axs[5].set_title('All Stocks Indices')
    axs[5].set_xlabel('Returns')
    axs[5].set_ylabel('Density')

    # Remove the unused subplot
    fig.delaxes(axs[5])

    plt.tight_layout()
    plt.show()
    return

#log transform dataframe
def log_transform(df):
    # Make a copy of the DataFrame to avoid modifying the original
    df_log = df.copy()
    
    # Select numeric columns
    numeric_columns = df_log.select_dtypes(include=[np.number]).columns
    
    # Apply log(1 + x) transformation to numeric columns
    df_log[numeric_columns] = np.log1p(df_log[numeric_columns])
    
    return df_log
