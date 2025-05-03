import pandas as pd
import lightgbm as lgb
from pull_data import fetch_bitcoin_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_updated_data(df):
    """
    Fetches BTC price data for the prediction period and returns a date->price dict.
    """
    logger.info("Starting fetch_updated_data")
    try:
        start_date = df.index[df['label'] == 'prediction'][0]
        end_date = df.index[df['label'] == 'prediction'][-1]
        logger.info(f"Fetching data from {start_date} to {end_date}")
    except IndexError:
        logger.warning("No prediction labels found in DataFrame. Returning empty dict.")
        return {}

    # Fetch the updated data
    updated_data = fetch_bitcoin_data(start_date=start_date, end_date=end_date)[['date', 'bitcoin_price']]
    updated_data['date'] = pd.to_datetime(updated_data['date'])
    logger.debug(f"Fetched {len(updated_data)} rows of updated data")

    # Convert to dict
    updated_data_dict = dict(zip(updated_data['date'], updated_data['bitcoin_price']))
    logger.info("Completed fetch_updated_data")
    return updated_data_dict
    
    
def plot_bitcoin_predictions(true_df, pred_df, final_df):
    """
    Plots true, predicted, and real BTC prices; returns the Figure.
    """
    logger.info("Creating bitcoin predictions plot")
    fig, ax = plt.subplots(figsize=(12, 6))

    # True data
    sns.lineplot(
        data=true_df, x=true_df.index, y='bitcoin_price',
        label='True Data', marker='o', linestyle='-', ax=ax
    )
    logger.debug(f"Plotted true data with {len(true_df)} points")

    # Predictions
    sns.lineplot(
        data=pred_df, x=pred_df.index, y='bitcoin_price',
        label='Prediction', marker='o', linestyle='--', ax=ax
    )
    logger.debug(f"Plotted predictions with {len(pred_df)} points")

    # Real data points
    real_dates = final_df.index[~final_df['real_data'].isna()]
    real_values = final_df.loc[real_dates, 'real_data']
    ax.scatter(real_dates, real_values, label='Real Data', s=100, zorder=5)
    logger.debug(f"Plotted real data with {len(real_values)} points")

    # Format x-axis
    ax.set_xticks(final_df.index)
    ax.set_xticklabels(final_df.index.strftime('%Y-%m-%d'), rotation=45)

    # Customize
    ax.set_title('Bitcoin Price: True, Predictions, and Real', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bitcoin Price', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Label', fontsize=12)
    plt.tight_layout()

    logger.info("Finished plot creation")
    return fig


def results(predictions, model, target):
    """
    Analyze model results: feature importance, prediction table, and plot.
    Returns gain plot, split plot, final DataFrame, and Figure.
    """
    logger.info("Starting results analysis")

    # Feature importances
    logger.info("Generating feature importance plots")
    gain_imp_plot = lgb.plot_importance(model, max_num_features=25, importance_type='gain', figsize=(10, 6))
    split_imp_plot = lgb.plot_importance(model, max_num_features=25, importance_type='split', figsize=(10, 6))

    # Prepare prediction table
    logger.info("Building predictions DataFrame")
    result_df = predictions[['bitcoin_price']].copy()
    result_df['label'] = 'true_data'

    # Derive future predictions
    if target == 'bitcoin_price_7d_log_return':
        predictions_df = predictions[['bitcoin_price']].copy()
        predictions_df['bitcoin_price'] = (
            predictions['bitcoin_price'] * np.exp(predictions['bitcoin_price_7d_log_return'])
        )
        predictions_df['label'] = 'prediction'
        predictions_df.index = predictions_df.index + pd.Timedelta(days=7)
    else:
        predictions_df = predictions[['bitcoin_price_7d_future']].copy()
        predictions_df.rename(columns={'bitcoin_price_7d_future': 'bitcoin_price'}, inplace=True)
        predictions_df['label'] = 'prediction'
        predictions_df.index = predictions_df.index + pd.Timedelta(days=7)

    final_df = pd.concat([result_df, predictions_df]).sort_index().round(2)
    logger.debug(f"Combined final_df with {len(final_df)} rows")

    # Add real_data column
    final_df['real_data'] = np.nan
    updated_dict = fetch_updated_data(final_df)
    for date, val in updated_dict.items():
        final_df.loc[date, 'real_data'] = val
    logger.debug(f"Inserted real_data for {len(updated_dict)} dates")

    # Subsets for plotting
    true_df = final_df[final_df['label'] == 'true_data']
    pred_df = final_df[final_df['label'] == 'prediction']

    # Plot results
    real_pred_chart = plot_bitcoin_predictions(true_df, pred_df, final_df)

    # Percentage difference
    final_df['% difference'] = (
        100 * (abs(final_df['real_data'] - final_df['bitcoin_price']) / final_df['real_data'])
    ).round(2)
    logger.info("Results analysis completed")

    return gain_imp_plot, split_imp_plot, final_df, real_pred_chart
