import pandas as pd


from feature_engineering import *
from training_strategy import *
from bayesian_opt import *
from cross_validation_training import *
from final_train import *

# Loading my API_KEY
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
SEED = os.getenv("SEED")



if __name__ == "main":
    
    # Fecha final dataset de entramiento BO
    last_date = pd.to_datetime("2024-12-31 00:00:00")
    
    
    ## Leo el dataset
    df = pd.read_csv("../data/preprocessed/transformed_df.csv")
    
    ## Aplico variables historicas
    
    # Lags & Deltas 
    df = add_lags(df, num_lags = 1, delta = True)
    df = add_lags(df, num_lags = 2, delta = True)
    df = add_lags(df, num_lags = 3, delta = True)
    df = add_lags(df, num_lags = 7, delta = True)
    
    # Operaciones de ventana
    df = add_rolling_features(df, window = 3)
    df = add_rolling_features(df, window = 7)
    df = add_rolling_features(df, window = 21)
    
    
    ## Training strategy -- train_up_to_date = ["2024-12-31"]
    splits , future = training_strategy(df, target = "bitcoin_price_7d_future")
    
    # Splt set 1: train until 2024-12-31 / test from 2025-01-01 to 2025-01-07
    train_1, val_1 = splits[last_date][0]
    
    ## Optimizacion Bayesiana
    best_params = bayesian_optimization(nr_trials = 50, 
                                        train = train_1, 
                                        val_set = val_1, 
                                        target = "bitcoin_price_7d_future",
                                        seed = SEED) 
    
    ## CV y metricas
    results, predictions = cv_training(params = best_params, splits = splits, target = "bitcoin_price_7d_future", seed = SEED )
    
    
    
    # Final train
    ## Sets
    train_final, future = training_strategy(df, target = "bitcoin_price_7d_future", final_train = True)
    
    train_final_bo = train_final.iloc[:-7]
    val_final_bo = train_final.iloc[-7:]
    
    ## BO
    final_best_params = bayesian_optimization(nr_trials = 50,
                                              train_set = train_final_bo,
                                              val_set = val_final_bo,
                                              target = "bitcoin_price_7d_future",
                                              seed = SEED)
    
    future_predictions = final_training(train_final_bo, val_final_bo, future, final_best_params, target = "bitcoin_price_7d_future", seed = SEED)
    
    
    
    
    
