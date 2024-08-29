import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy import abs
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import logging as log

logger = log.getLogger('utils')


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
    logger.info('loading data from excel file')
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
            logger.info("num of NaNs in %s: %s", cols_name, series.isna().sum())
    
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

# #TODO Generalize date
# def divide_df_lastyear(df):
#     """Divide dataframe into two dataframes, cutting the original one at a certain date

#     Args:
#         dataframe (pd.DataFrame): original dataframe to cut

#     Returns:
#         (pd.DataFrame, pd.DataFrame): returns a tuple like (training DataFrame, testing DataFrame) 
#     """
#     training = df.loc[:'2018-12-31']
#     testing = df.loc['2019-01-01':]
#     return (training, testing)


def timeFilterDataframeCollection(data, start_date, end_date) -> pd.DataFrame:
    """
    Filter the date-indexed (datetime format) collection of dataframes
    given a starting date and end_date

    Args:
    data: collection of dataframes to time filter
    start/end_date: start and end date in pandas format for filtering

    Returns:
    time filtered dataframe
    """
    
    offset_data = {}
    for df in data:
        offset_data[df] = data[df].loc[start_date:end_date]
    return offset_data



def cleanMatrixSmall(df, threshold) -> pd.DataFrame:
    """
    Clean matrix by replacing with zero any value
    smaller than threhshold.

    Args:
    df: dataframe to clean
    threshold: if number is small than this arg than replace with zero

    Returns:
    cleaned dataframe
    """
    # Calculate the covariance matrix
    cov_matrix = df.cov()
    
    # Create a mask for values smaller than the threshold
    mask = abs(cov_matrix) < threshold
    
    # Replace values smaller than the threshold with 0
    cov_matrix[mask] = 0
    
    return cov_matrix

def forceDiagonalCov(df) -> pd.DataFrame:
    """
    force the matrix passed as dataframe to be
    diagonal by multiplying it with the identity matrix

    Args: dataframe to force into diagonal matrix

    Returns:
    forced diagonal matrix
    """
    cov_matrix = df.cov()
    
    identity_matrix = np.eye(cov_matrix.shape[0])
    
    return cov_matrix * identity_matrix


def fiveFigurePlot(df:pd.DataFrame):
    """
    Fit each column against gaussian and plot
    histogram of each column and fitted gaussian.
    Single image using different subplots.
    """
    
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
        logger.info("mu: %s, std: %s", mu,std)
        
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


def logTransform(df):
    """
    Apply log transformation to dataframe.

    Args:
    df: dataframe to transform

    Returns:
    log(1 + df)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_log = df.copy()
    
    # Select numeric columns
    numeric_columns = df_log.select_dtypes(include=[np.number]).columns
    
    # Apply log(1 + x) transformation to numeric columns
    df_log[numeric_columns] = np.log1p(df_log[numeric_columns])
    
    return df_log

# FIXME should add resampling other than daily
def graphPortfolioStocksPerformance(portfolio_simple_returns, returns_testing_simple):
    """
    Graph cumulative returns of portfolio and each stock index, highlighting the portfolio curve.

    Args:
    df_portfolio_simple_returns: dataframe containing the portfolio cumulative returns
    returns_testing_simple: dataframe containing each stock simple cumulative returns

    Returns:
    nothing, prints graph interactively
    """

    df = pd.concat([portfolio_simple_returns, returns_testing_simple], axis=1)
    series_to_highlight = portfolio_simple_returns.name
    for column in df.columns:
        if column != series_to_highlight:
            plt.plot(df.index, df[column], label=column, alpha=0.5)
            
    # Highlight specific series
    plt.plot(df.index, df[series_to_highlight], linewidth=2, color='blue', label=f"{series_to_highlight}")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return

