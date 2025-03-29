import lightgbm  as lgb
from sklearn.metrics import mean_squared_error
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def final_training(train_set, val_set, future_set, params, target, seed):
    
    
    # Preparacion de training sets
    X_train = train_set.drop(columns = target)
    y_train = train_set[target]
    
    X_val = val_set.drop(columns=target)
    y_val = val_set[target]
    
    # Preparacion de future set
    X_future = future_set.drop(columns=target)
    
    # Creamos sets de LGBM
    train_data = lgb.Dataset(X_train, label= y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Entrenamos el modelo
    final_model = lgb.train(params,
                            train_data,
                            valid_sets = [val_data],
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=50)]                            
                            )
    
    # Guardamos predicciones
    future_predictions = final_model.predict(X_future)
    
    # Guardamos en dataset.
    predictions_df =future_set.copy()
    predictions_df[target] = future_predictions
    
    return predictions_df
