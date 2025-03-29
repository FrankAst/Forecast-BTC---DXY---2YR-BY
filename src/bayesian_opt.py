import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd


def objective(trial, X_train, y_train, X_val, y_val, seed):
    # Suggest hyperparameters from various distributions:
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'seed': seed,             # ensure reproducibility in LightGBM
        'deterministic': True     # optional: ensures deterministic behavior
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    
    # Train the model, early stopping if validation performance does not improve
    gbm = lgb.train(param, train_data, 
                    valid_sets=[valid_data],
                    #verbose=False,
                   callbacks=[
                        lgb.early_stopping(stopping_rounds=50),  # New syntax for early stopping
                        lgb.log_evaluation(period=0)  # Disable evaluation logging
        ])
    
    # Predict on validation set
    preds = gbm.predict(X_val)
    # Compute RMSE as the objective metric
    rmse = mean_squared_error(y_val, preds)
                              
    return rmse

# Create a study object. We want to minimize RMSE.
def bayesian_optimization(train_set, val_set, target,seed, nr_trials = 50):
    
    X_train = train_set.drop(columns = target, axis =1)
    y_train = train_set[target]
    
    X_val = val_set.drop(columns = target, axis =1)
    y_val = val_set[target]
    
    # Creamos el study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler = sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, seed = seed), n_trials = nr_trials)  # Increase n_trials for more thorough search

    # Extract the best hyperparameters
    best_params = study.best_params

    return best_params


