import lightgbm  as lgb
from sklearn.metrics import mean_squared_error
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (MAPE)."""
    # Avoid division by zero by replacing zeros with a very small number
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def cv_training(params, splits, target, seed):
    
    '''
    params = best params from BO 
    splits = set of splits for train/test from the walk forward cv.
    target = variable to predict
    seed = seed.
    
    '''
    
    results = []  # List to store RMSE results per fold
    predictions_list = []  # List to store predictions + actual values
    
    for cutoff_date, splits in splits.items():
        logger.info(f"CV splits starting from: {cutoff_date}")
        
        for i, (train, test) in enumerate(splits, start=1):
            
            X_train = train.drop(columns=target)
            y_train = train[target]
            X_test = test.drop(columns=target)
            y_test = test[target]

            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)
            
            # Train the model, early stopping if validation performance does not improve
            gbm = lgb.train(params, train_data, 
                            valid_sets=[valid_data],
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=50),  # New syntax for early stopping
                                lgb.log_evaluation(period=0)  # Disable evaluation logging
                ])
        
            # Predict on validation set
            preds = gbm.predict(X_test)
            
             # Compute RMSE
            rmse = mean_squared_error(y_test, preds)**0.5
            mape_val = mape(y_test.values, preds)
            logger.info(f"  Fold {i}: RMSE = {rmse:.4f}, MAPE = {mape_val:.4f}%")

            # Store RMSE results
            results.append({"cutoff_date": cutoff_date, 
                            "fold": i, 
                            "fold start date": X_test.index.min().date(), 
                            "rmse": rmse,
                            "mape": mape_val
                            })

            # Store predictions alongside actual values
            fold_predictions = pd.DataFrame({
                "cutoff_date": cutoff_date,
                "fold": i,
                "Date": X_test.index, #.min().date(),
                "actual": y_test.values,
                "predicted": preds
            })
            predictions_list.append(fold_predictions)

    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    # Combine all predictions into a single DataFrame
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    return results_df, predictions_df
            
   

    