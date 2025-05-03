import pandas as pd
import numpy as np


# Funcion para incorporar lags y delta lags.
def add_lags(df, num_lags, exclude_columns=None, delta=False):

    """
    Adds lagged columns for each column in the DataFrame (except those in exclude_columns 
    or columns whose name already contains '_lag'). Optionally, it computes delta lags,
    which are the differences between the original column and its lagged version.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame.
        num_lags (int): Number of lag periods to add.
        exclude_columns (list): List of columns to exclude from lagging.
        delta (bool): If True, also add columns with delta lags.
        
    Returns:
        pd.DataFrame: DataFrame with additional lagged columns (and delta columns if delta=True).
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'Year', 'quarter', 'month',
                           'dayofyear', 'dayofweek', 'bitcoin_price_7d_future',
                           'bitcoin_price_7d_log_return', 'fng_classification']
    
    
    df_with_lags = df.copy()
    
    # Exclude columns that are in the exclusion list or already contain '_lag'
    cols_to_lag = [col for col in df.columns 
                   if col not in exclude_columns and '_lag' not in col]
    
     # Only add the single lag specified by num_lags
    lag = num_lags
    shifted = df[cols_to_lag].shift(lag)
    lagged_names = [f'{col}_lag{lag}' for col in shifted.columns]
    shifted.columns = lagged_names
    df_with_lags = pd.concat([df_with_lags, shifted], axis=1)

    # Optionally compute delta lags (difference between current and lagged values)
    if delta:
        delta_values = df[cols_to_lag].values - shifted.values
        delta_names = [f'{col}_delta_lag{lag}' for col in cols_to_lag]
        delta_df = pd.DataFrame(delta_values, index=df.index, columns=delta_names)
        df_with_lags = pd.concat([df_with_lags, delta_df], axis=1)
    
    return df_with_lags

# Funcion para agregar rolling features como ser medias moviles, EMAs, SD moviles, max, min. 
def add_rolling_features(df, window, exclude_columns=None, ema_span=None):
    """
    Adds rolling window features (mean, std, max, min) and EMA for each column in the DataFrame 
    that is not in the exclude_columns list, using vectorized operations.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a DateTime index.
        window (int): Size of the rolling window (number of periods) for the rolling stats.
        exclude_columns (list): Columns to exclude from rolling and EMA calculations.
        ema_span (int): Span for EMA calculation. If None, defaults to the rolling window value.
        
    Returns:
        pd.DataFrame: DataFrame with original columns plus new rolling and EMA feature columns.
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'Year', 'quarter', 'month',
                           'dayofyear', 'dayofweek', 'bitcoin_price_7d_future',
                           'bitcoin_price_7d_log_return', 'fng_classification']
    
        
    if ema_span is None:
        ema_span = window  # Default EMA span to the rolling window size

    # Determine columns for which to calculate features (excluding columns with '_lag_' or '_roll_')
    cols_to_calc = [col for col in df.columns 
                    if col not in exclude_columns and '_lag' not in col and '_roll_' not in col and '_ema_' not in col]

    # Compute rolling statistics for all selected columns
    rolling_stats = df[cols_to_calc].rolling(window=window, closed='right').agg(['mean', 'std', 'max', 'min'])
    # Flatten MultiIndex columns
    rolling_stats.columns = [f"{col}_roll_{stat}_{window}" for col, stat in rolling_stats.columns]

    # Compute exponential moving average (EMA) for each selected column
    ema_df = df[cols_to_calc].ewm(span=ema_span, adjust=False).mean()
    ema_df.columns = [f"{col}_ema_{ema_span}" for col in ema_df.columns]

    # Concatenate original DataFrame with the new features
    df_new = pd.concat([df, rolling_stats, ema_df], axis=1)
    return df_new



    
    
    
    
    
    
    
    
    
    
    
    
    return


























































