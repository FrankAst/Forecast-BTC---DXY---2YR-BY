import pandas as pd


# Funcion para generar los splits
def walk_forward_splits(df, train_up_to_date, number_of_val_windows, window_length = 7 ):
    
    # crear train / tests splits
    
    """
    Creates walk-forward CV splits. For each starting training cutoff in `train_up_to_date`,
    the function produces a series of train/test splits. Starting with training data up to the given date,
    it predicts the closing value for the next `window_length` days, then incorporates those days into training,
    then predicts the next window, repeating for `number_of_val_windows` windows.
    
    Parameters:
      df (pd.DataFrame): DataFrame with a DateTime index.
      train_up_to_date: lista con inicios de CV.
      number_of_val_windows: cantidad de ventanas a evaluar en cada CV.
      window_length: cantidad de dias en la ventana de evaluacion.
      
      
    Returns:
      dict: A dictionary where each key is a starting date from `train_up_to_date` and its value is a list 
            of (train, test) tuples for each CV split.
    """
    
    
    # Ensure the index is datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Dictionary to hold splits for each starting date
    cv_splits = {}
    
    # Loop over each provided starting date
    for start_date in train_up_to_date:
        # Convert to Timestamp if necessary
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        splits_for_date = []
        
        # Initial training: all data up to and including the start_date
        train = df[df.index <= start_date].copy()
        # Set current training end as the last date in the training set (which should be start_date)
        current_train_end = train.index.max()
        
        # Create walk-forward splits
        for window in range(number_of_val_windows):
            # Define test window: next 'window_length' days after current_train_end
            test_start = current_train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=window_length - 1)
            
            # Get test set
            test = df[(df.index >= test_start) & (df.index <= test_end)].copy()
            # If no test data is available, break out of the loop
            if test.empty:
                break
            
            # Append the (train, test) tuple
            splits_for_date.append((train.copy(), test.copy()))
            
            # Update the training set to include the new test window
            train = pd.concat([train, test]).sort_index()
            # Update current training end
            current_train_end = train.index.max()
        
        cv_splits[start_date] = splits_for_date
        
    return cv_splits


def training_strategy(df, target, train_up_to_date = ["2024-12-31"], number_of_val_windows = 8):
    
    '''
    Definir training set. Hasta 2024 incluido por default.
    Definir set de validacion en 2025 ~ 8 semanas = 2 meses por defecto.
    Definir final test set 2025
    Definir future set 2025 (target es NaN)
    '''
    
    # Create future_set with rows where the target column is NaN
    future_set = df[df[target].isnull()].copy()
    
    # Exclude future_set from df to create available_data
    available_data = df[~df.index.isin(future_set.index)].copy()
    
    
    # Generamos los splits de train y splits para CV walk forward.     
    cv_splits = walk_forward_splits(available_data, train_up_to_date, number_of_val_windows)
    
    return cv_splits , future_set
    
    
    





















