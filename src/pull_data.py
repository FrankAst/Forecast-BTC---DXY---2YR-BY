import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging

# Loading my API_KEY
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
API_KEY = os.getenv("FRED_API_KEY_F")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================
# FRED API (for 2-Year Treasury Yield)
# ==============================================
FRED_API_KEY =  API_KEY # Replace with your FRED API key

def fetch_fred_data(series_id, start_date='2015-01-01', end_date=None):
    """
    Fetch historical data from FRED API.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    url = f'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    logging.info(f"Fetching data from FRED API for series {series_id} from {start_date} to {end_date}")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logging.error(f"Failed to fetch data from FRED API: {response.status_code} - {response.text}")
        return pd.DataFrame()
    
    data = response.json()['observations']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    logging.info(f"Successfully fetched {len(df)} rows for series {series_id}")
    return df[['date', 'value']].rename(columns={'value': series_id})

# ==============================================
# yfinance (for Bitcoin Data)
# ==============================================
def fetch_bitcoin_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical Bitcoin data using yfinance.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    logging.info(f"Fetching Bitcoin data from yfinance from {start_date} to {end_date}")
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    if btc.empty:
        logging.error("Failed to fetch Bitcoin data from yfinance")
        return pd.DataFrame()
    
    btc.reset_index(inplace=True)
    btc.rename(columns={'Date': 'date', 'Close': 'bitcoin_price'}, inplace=True)
    logging.info(f"Successfully fetched {len(btc)} rows of Bitcoin data")
    return btc[['date', 'bitcoin_price']]

# ==============================================
# yfinance (for DXY - US Dollar Index)
# ==============================================
def fetch_dxy_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical DXY data using yfinance.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    logging.info(f"Fetching DXY data from yfinance from {start_date} to {end_date}")
    dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date)
    if dxy.empty:
        logging.error("Failed to fetch DXY data from yfinance")
        return pd.DataFrame()
    
    dxy.reset_index(inplace=True)
    dxy.rename(columns={'Date': 'date', 'Close': 'dxy'}, inplace=True)
    logging.info(f"Successfully fetched {len(dxy)} rows of DXY data")
    return dxy[['date', 'dxy']]

# ==============================================
# Fetch All Data
# ==============================================
def fetch_all_data():
    logging.info("Starting to fetch all data")
    # Fetch 2-Year Treasury Yield
    dgs2 = fetch_fred_data('DGS2')  # 2-Year Treasury Yield

    # Fetch Bitcoin Data
    btc = fetch_bitcoin_data()

    # Fetch DXY (US Dollar Index)
    dxy = fetch_dxy_data()

    # Merge all data into a single DataFrame
    data_frames = [dgs2, btc, dxy]
    
    """
    merged_data = data_frames[0]
    for df in data_frames[1:]:
        merged_data = pd.merge(merged_data, df, on='date', how='outer')
    
    logging.info(f"Successfully merged data into a single DataFrame with {len(merged_data)} rows")
    return merged_data.sort_values('date').reset_index(drop=True) """

    return data_frames
    
# ==============================================
# Run the Code
# ==============================================
if __name__ == '__main__':
    logging.info("Starting the data fetching process")
    # Fetch all data
    data = fetch_all_data()
    
    if len(data)>1:
        # Save to CSV
        data[0].to_csv('../data/dsg.csv', index=False)
        data[1].to_csv('../data/btc.csv', index=False)
        data[2].to_csv('../data/dxy.csv', index=False)
        logging.info("Data saved successfuly '")
    else:
        logging.error("No data fetched, CSV file not created")