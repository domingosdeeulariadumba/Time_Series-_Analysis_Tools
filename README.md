ABOUT
        
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


HOW TO IMPORT THESE FUNCTIONS?     

        All Functions: from time_series.ts_tools import *
        Some Functions: from time_series.ts_tools import show_seasonal, show_periodogram 


