import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from feature_engineering import *
from training_strategy import *
from bayesian_opt import *
from canarios_asesinos import *
from cross_validation_training import *
from final_train import *
from analisis_resultados import *

# Loading my API_KEY
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
SEED = os.getenv("SEED")



def main(tgt = 'bitcoin_price_7d_log_return'):
    
    # Fecha final dataset de entramiento BO
    last_date = pd.to_datetime("2025-02-28 00:00:00")
    
    
    ## Leo el dataset
    df = pd.read_csv("../data/preprocessed/transformed_df.csv")
    
    ## Aplico variables historicas
    
    # Lags & Deltas 
    df = add_lags(df, num_lags = 7, delta = True)
    df = add_lags(df, num_lags = 14, delta = True)
    df = add_lags(df, num_lags = 21, delta = True)
    
    # Operaciones de ventana
    df = add_rolling_features(df, window = 7)
    df = add_rolling_features(df, window = 21)
    
    # Check for duplicated columns
    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        df = df.loc[:, ~df.columns.duplicated()]
    
    
    ## Training strategy -- train_up_to_date = ["2024-12-31"]
    splits , future = training_strategy(df, train_up_to_date = [last_date],  target = tgt)
     
    # Splt set 1: train until 2024-12-31 / test from 2025-01-01 to 2025-01-07
    train_1, val_1 = splits[last_date][0]
    
    '''
    ## Optimizacion Bayesiana
    best_params = bayesian_optimization(nr_trials = 50, 
                                        train = train_1, 
                                        val_set = val_1, 
                                        target = tgt,
                                        seed = SEED) 
    
    ## Canarios asesinos
    
    selected_features = select_canary_features(train_1, target = tgt, params = best_params, n_canaries = 5)
    
    ## CV y metricas
    results, predictions = cv_training(params = best_params, splits = splits, target = tgt, seed = SEED, variables_depuradas = selected_features)
    
    
    # Final train
    ## Sets
    train_final, future = training_strategy(df, target = tgt, final_train = True)
    
    # Select only the features that were selected in the canary feature selection
    if selected_features is not None:
            train_final = train_final[selected_features + ['bitcoin_price_7d_log_return']]
            future = future[selected_features + ['bitcoin_price_7d_log_return']]
    
    # Set de entrenamiento y de validacion finales.
    train_final_bo = train_final.iloc[:-7]
    val_final_bo = train_final.iloc[-7:]
    
    ## BO
    final_best_params = bayesian_optimization(nr_trials = 50,
                                              train_set = train_final_bo,
                                              val_set = val_final_bo,
                                              target = tgt,
                                              seed = SEED)
    
    ## Entrenamiento final y predicciones de futuro.
    future_predictions, model = final_training(train_final_bo, val_final_bo, future, final_best_params, target = tgt, seed = SEED)
    
    
    ################################### Resultados ###################################
     
    gain_importance , split_importance, final_df, real_vs_pred_chart = results(predictions, model, target)
    
    # Save results into the folder ../results_wf
    output_dir = "../results_wf"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardamos los importance plots as pdf
    from matplotlib.backends.backend_pdf import PdfPages
    fi_pdf_path = os.path.join(output_dir, "feature_importance.pdf")
    with PdfPages(fi_pdf_path) as pdf:
        pdf.savefig(gain_importance.figure)
        pdf.savefig(split_importance.figure)
    
    # Guardamos el grafico de predicciones
    pred_pdf_path = os.path.join(output_dir, "real_vs_predictions.pdf")
    real_vs_pred_chart.savefig(pred_pdf_path)
    
    # Guardamos el dataframe final
    final_df_path = os.path.join(output_dir, "final_df.csv")
    final_df.to_csv(final_df_path, index = True)
    
'''
    
if __name__ == "__main__":
    
    target = 'bitcoin_price_7d_log_return'
    main(tgt = target)

    
