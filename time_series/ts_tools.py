# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:08:48 2024

@author: domingosdeeularia
"""


'''
        SUMMARY
        ------
        
        
        
        
        - Time Series Analysis Tools -
        
        This module contains functions for time series analysis, including visualizations, statistical tests, 
        and model evaluation metrics.
        
        Imports:
        --------
        - pandas: for data manipulation and analysis.
        - matplotlib.pyplot: for plotting.
        - seaborn: for advanced plotting.
        - scipy.signal: for signal processing functions, including periodograms.
        - statsmodels: for statistical modeling, including time series analysis.
        - sklearn.metrics: for model evaluation metrics.
        
        Functions:
        ----------
        1. show_periodogram(data, detrend='linear', ax=None, fs=365, color='brown'):
            Displays the periodogram of the given time series data.
        
        2. show_seasonal(df, period, freq, ax=None, title=None, x_label=None, y_label=None):
            Creates seasonal plots for visualizing seasonal patterns in time series data.
        
        3. show_lags(data, n_lags=10, title='Lag Plots'):
            Generates lag plots to visualize the autocorrelation of a time series.
        
        4. adf_test(df):
            Performs the Augmented Dickey-Fuller test to check for stationarity in the time series.
        
        5. show_correlogram(data, lags=6, ACF=True, PACF=True):
            Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the time series.
        
        6. evaluate(models_forecast, test_set, view='results'):
            Evaluates multiple forecasting models and displays their performance metrics.
        7. split_data(df, train_proportion = 0.8):
            Splits a DataFrame into training and testing sets.
        
'''



'''
        IMPORTING MODULES
        ------------
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import linregress



'''
        Implementation
        --------------
'''
        
# 1. show_periodogram()

def show_periodogram(data, detrend='linear', ax=None, fs=365, color='brown'):
    '''
    Displays the periodogram of the given time series data.

    The periodogram helps identify the dominant frequencies in the time series, providing insights into 
    its periodic components.

    Parameters:
    ----------
    data : pandas.DataFrame or pandas.Series
        The time series data to analyze. Should be a 1D series or a DataFrame with a single column.
    
    detrend : str, optional
        The detrending method to use. Options include 'linear' (default) or 'constant'. Detrending removes trends from the data 
        before calculating the periodogram.

    ax : matplotlib.axes.Axes, optional
        The Axes object on which to plot the periodogram. If None, a new figure and axes are created.

    fs : int, optional
        The sampling frequency of the data. Default is 365, which is suitable for daily data over a year.

    color : str, optional
        The color of the plot line. Default is 'brown'.

    Returns:
    -------
    None
        The function displays the periodogram plot and does not return any value.
    
    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts_data = pd.Series(np.random.randn(365))
    >>> show_periodogram(ts_data)
    '''
    series = data.squeeze()
    
    # Number of observations
    n_obs = len(data)
     
    # Computing frequencies and spectrum
    frequencies, spectrum = periodogram(series, fs, window='boxcar', detrend=detrend,
                                        scaling='spectrum')

    # Plotting the periodogram
    if ax is None:
        _, ax = plt.subplots()

    # Frequency adjustment
    freqs = [1, 2, 3, 4, 6, 12, 26, 52, 104]
    freqs_labels = ['Annual (1)', 'Semiannual (2)', 'Triannual (3)',
                    'Quarterly (4)', 'Bimonthly (6)', 'Monthly (12)',
                    'Biweekly (26)', 'Weekly (52)', 'Semiweekly (104)']

    ax.step(frequencies, spectrum, color=color)
    ax.set_xscale('log')
    ax.set_xticks(freqs)
    ax.set_xticklabels(freqs_labels, rotation=55, fontsize=11)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(f'Periodogram ({n_obs} Observations)')
    plt.show()
    
    return


# 2. show_seasonal()

def show_seasonal(df, period, freq, ax=None, title=None, x_label=None, y_label=None):
    '''
    Creates seasonal plots for visualizing seasonal patterns in time series data.

    This function generates line plots that illustrate the variations in the time series across specified 
    seasonal periods, helping to identify trends and seasonal effects.

    Parameters:
    ----------
    df : pandas.DataFrame
        The time series DataFrame containing the data to analyze. The DataFrame should have a DateTime index.

    period : str
        The column name representing the seasonal period. This is used to group the data for the seasonal plot.

    freq : str
        The column name representing the frequency of the seasonality. This typically denotes the time interval 
        at which the seasonal effect is observed (e.g., month, week).

    ax : matplotlib.axes.Axes, optional
        The Axes object on which to plot the seasonal plot. If None, a new figure and axes are created.

    title : str, optional
        Title of the plot. If None, a default title will be generated.

    x_label : str, optional
        Label for the x-axis. If None, the frequency name will be used. If 'hide', the label will not be shown.

    y_label : str, optional
        Label for the y-axis. If None, the first column name of the DataFrame will be used.

    Returns:
    -------
    None
        The function displays the seasonal plot and does not return any value.

    Example:
    --------
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    >>> data = pd.DataFrame({'value': np.random.rand(len(date_rng))}, index=date_rng)
    >>> show_seasonal(data, period='month', freq='day')
    '''
    # Copying the dataset
    data = df.copy()
    
    # Passing the date column as the index of the DataFrame
    data.set_index(data.index.values, inplace=True)
    
    # Extracting date elements from the index
    data['year'] = data.index.year
    data['month'] = data.index.strftime('%b')
    data['week of year'] = data.index.isocalendar().week.astype(int)  
    data['day of month'] = data.index.day    
    data['day of week'] = data.index.strftime('%a')
    data['day of year'] = data.index.dayofyear

    # Plotting seasonal effects
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette('rocket', n_colors=data[period].nunique())
    ax = sns.lineplot(
        x=freq, y=data.columns[0], hue=period, data=data, ax=ax, palette=palette)

    # Setting options for the plot title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Seasonal Plot ({period}/{freq})')

    # Setting label options for x axis    
    if x_label is None:
        plt.xlabel(freq)    
    elif x_label == 'hide':
        plt.xlabel('')    
    else:
        plt.xlabel(x_label)

    # Setting label options for y axis     
    if y_label is None:
        plt.ylabel(data.columns[0])    
    else:
        plt.ylabel(y_label)

    plt.legend(fontsize=8, loc='best')
    plt.show()

    return


# 3. show_lags()

def show_lags(data, n_lags=10, title='Lag Plots'):
    '''
    Generates lag plots to visualize the autocorrelation of a time series.

    This function creates scatter plots of the time series against its lagged versions, helping to assess 
    the degree of autocorrelation at different lags. Each plot includes a regression line to indicate 
    the strength of the relationship between the original series and its lagged version.

    Parameters:
    ----------
    data : pandas.DataFrame or pandas.Series
        The time series data to analyze. Should be a 1D series or a DataFrame with a single column.

    n_lags : int, optional
        Number of lag plots to generate. Default is 10, creating lag plots from 1 to 10 lags.

    title : str, optional
        Title of the overall figure. Default is 'Lag Plots'.

    Returns:
    -------
    None
        The function displays the lag plots and does not return any value.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts_data = pd.Series(np.random.randn(100))
    >>> show_lags(ts_data)
    '''
    fig, axes = plt.subplots(2, n_lags // 2, figsize=(10, 6),
                             sharex=False, sharey=True, dpi=240)

    for i, ax in enumerate(axes.flatten()[:n_lags]):
        lag_data = pd.DataFrame({'x': data.squeeze(),
                                 'y': data.squeeze().shift(i + 1)}).dropna()

        x, y = lag_data['x'], lag_data['y']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = [(slope * xi) + intercept for xi in x]   
        ax.scatter(x, y, c='k', alpha=0.6)
        ax.plot(x, regression_line, color='m', label=f'{r_value**2:.2f}')
        ax.set_title(f'Lag {i + 1}')
        ax.legend()
        ax.grid(True) 

    plt.tight_layout()
    plt.suptitle(title, y=1.05)
    plt.show()

    return


# 4. adf_test()

def adf_test(df):
    ''''
    Checks the stationarity of a time series using the Augmented Dickey-Fuller Test.

    This function performs the Augmented Dickey-Fuller (ADF) test to assess whether a given time series 
    is stationary or contains a unit root. It prints the test results and displays key statistics for 
    evaluation.

    Parameters:
    ----------
    df : pandas.DataFrame or pandas.Series
        The time series data to analyze. Should be a 1D series or a DataFrame with a single column.

    Returns:
    -------
    None
        The function prints the results of the ADF test and displays a DataFrame containing the test statistics.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts_data = pd.Series(np.random.randn(100).cumsum())
    >>> adf_test(ts_data)
    '''
    # Performing the Augmented Dickey-Fuller Test    
    adft = adf(df, autolag='AIC')
     
    # DataFrame to store the results of the test    
    scores_df = pd.DataFrame({'Scores': [adft[0], adft[1], 
                                         adft[2], adft[3], adft[4]['1%'],
                                         adft[4]['5%'], adft[4]['10%']]},
                             index=['Test Statistic',
                                    'p-value', 'Lags Used',
                                    'Observations', 
                                    'Critical Value (1%)',
                                    'Critical Value (5%)',
                                    'Critical Value (10%)'])
     
    # Printing the result of the test    
    if adft[1] > 0.05 and abs(adft[0]) > adft[4]['5%']:
        print('\033[1mThis series is not stationary!\033[0m')
    else:
        print('\033[1mThis series is stationary!\033[0m')
         
    print('\nResults of Dickey-Fuller Test\n' + '=' * 29)
      
    # Displaying the DataFrame with the statistics parameters of the ADF test    
    display(scores_df)
     
    return


# 5. show_correlogram()

def show_correlogram(data, lags=6, ACF=True, PACF=True):
    '''
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of a time series.

    This function generates correlograms to help assess the autocorrelation and partial autocorrelation 
    of the time series, which are crucial for identifying the orders of Autoregressive (AR) and Moving Average 
    (MA) components in time series models.

    Parameters:
    ----------
    data : pandas.DataFrame or pandas.Series
        The time series data to analyze. Should be a 1D series or a DataFrame with a single column.

    lags : int, optional
        The number of lags to include in the ACF and PACF plots. Default is 6.

    ACF : bool, optional
        Whether to plot the Autocorrelation Function. Default is True.

    PACF : bool, optional
        Whether to plot the Partial Autocorrelation Function. Default is True.

    Returns:
    -------
    None
        The function displays the ACF and PACF plots and does not return any value.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts_data = pd.Series(np.random.randn(100))
    >>> show_correlogram(ts_data)
    '''
    # Transforming the DataFrame to a pandas Series 
    series = data.squeeze()
    
    # Setting the plots for correlograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if ACF:       
        # ACF Plot
        plot_acf(series, lags=lags, ax=axes[0])
        axes[0].set_title('ACF Plot')
        
    if PACF:
        # PACF Plot
        plot_pacf(series, lags=lags, ax=axes[1])  
        axes[1].set_title('PACF Plot')  

    plt.tight_layout()
    plt.show()
    
    return


# 5. evaluate()

def evaluate(models_forecast, test_set, view='results'):
    '''
    Assesses the performance of forecast models using various metrics.

    This function compares the forecasts from multiple models against a test dataset by calculating 
    common performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), 
    Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). It can display either 
    the performance metrics or a comparison of the actual vs. predicted values.

    Parameters:
    ----------
    models_forecast : list of tuples
        A list containing tuples where each tuple consists of a model name (str) and the corresponding 
        forecasted data (pandas.Series or pandas.DataFrame).

    test_set : pandas.DataFrame
        The actual test dataset containing the true values for comparison. Should have a single column.

    view : str, optional
        Specifies the output view. Options are 'metrics' to display model performance metrics or 
        'results' to display a comparison of actual vs. predicted values. Default is 'results'.

    Returns:
    -------
    None
        The function displays the performance metrics or results comparison and does not return any value.

    Example:
    --------
    >>> models_forecast = [
    >>>     ('Model A', forecast_a),
    >>>     ('Model B', forecast_b)
    >>> ]
    >>> evaluate(models_forecast, test_set)
    '''
    # Empty list of all models' metrics
    df_metrics_global = []
    test_set.rename(columns={test_set.columns[0]: 'Actual'}, inplace=True)
    
    # Empty variable to receive the forecast sets and data set to be compared
    df_comp = pd.DataFrame()
    
    for model_name, forecast_data in models_forecast:
        # Calculate the parameters
        mse = mean_squared_error(test_set, forecast_data)
        rmse = mean_squared_error(test_set, forecast_data, squared=False)
        mae = mean_absolute_error(test_set, forecast_data)
        mape = mean_absolute_percentage_error(test_set, forecast_data)
    
        # Create a DataFrame of metrics
        metrics_list = [mse, rmse, mae, mape]
        metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
        df_metrics = pd.DataFrame({model_name: metrics_list}, index=metrics_names)
    
        # Add the DataFrame to the list
        df_metrics_global.append(df_metrics)
        
        # Add the forecast to the comparison DataFrame
        df_comp[model_name] = forecast_data 
    
    # Concatenate all DataFrames of metrics and results comparison
    metrics = pd.concat(df_metrics_global, axis=1)
    df_comp_final = pd.concat([test_set, df_comp], axis=1)
    
    # Option to display the transposed performance metrics DataFrame 
    if view == 'metrics':
        display(metrics.T)
        
    # Option to display the transposed comparison DataFrame 
    elif view == 'results':
        display(df_comp_final.head(10).T)
    else:
        print("This option is not available! :(\nPlease choose whether you want to display the performance inserting "
              "\033[1m'metrics'\033[0m or \033[1m'results'\033[0m in case of forecast comparison.")
    
    return


# 7. Function to split the series

def split_data(df, train_proportion=0.8):
    '''
    Splits a DataFrame into training and testing sets.

    This function divides the provided DataFrame into two subsets: a training set and a testing set,
    based on a specified proportion. The training set contains the first portion of the data, while
    the testing set contains the remaining data.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame to be split. Should contain time series or sequential data.

    train_proportion : float, optional
        The proportion of the data to include in the training set. Must be between 0 and 1.
        Default is 0.8, meaning 80% of the data will be used for training.

    Returns:
    -------
    tuple
        A tuple containing two elements:
        - train_set (pandas.DataFrame): The training set.
        - test_set (pandas.DataFrame): The testing set.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> train, test = split_data(data)
    >>> print(train)
    >>> print(test)
    '''
    train_size = round(len(df) * train_proportion)
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    
    return train_set, test_set



'''
        HOW TO IMPORT?
        ------------
        
        All Functions: from time_series.ts_tools import *
        Some Functions: from time_series.ts_tools import show_seasonal, show_periodogram        
'''