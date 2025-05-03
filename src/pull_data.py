import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from io import StringIO


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
# FRED API (FEDERAL FUNDS RATE)
# ==============================================

def fetch_fedfunds_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical Federal Funds Rate data from FRED using series 'FEDFUNDS'.
    """
    logging.info(f"Fetching Federal Funds Rate data from {start_date} to {end_date if end_date else datetime.today().strftime('%Y-%m-%d')}")
    return fetch_fred_data('FEDFUNDS', start_date=start_date, end_date=end_date)

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
    return btc #[['date', 'bitcoin_price']]

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
    return dxy#[['date', 'dxy']]


# ==============================================
# yfinance (for VIX - Volatility Index)
# ==============================================

def fetch_vix_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical VIX data from Yahoo Finance using ticker '^VIX'.
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    logging.info(f"Fetching VIX data from {start_date} to {end_date}")
    vix = yf.download('^VIX', start=start_date, end=end_date)
    if vix.empty:
        logging.error("Failed to fetch VIX data from yfinance")
        return pd.DataFrame()
    vix.reset_index(inplace=True)
    vix.rename(columns={'Date': 'date', 'Close': 'vix_close'}, inplace=True)
    logging.info(f"Successfully fetched {len(vix)} rows of VIX data")
    return vix


# ==============================================
# yfinance (for Gold Data)
# ==============================================

def fetch_gold_data(start_date='2015-01-01', end_date=None):
    """
    Fetch historical Gold data from Yahoo Finance using ticker 'GC=F' (Gold Futures).
    """
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    logging.info(f"Fetching Gold data from {start_date} to {end_date}")
    gold = yf.download('GC=F', start=start_date, end=end_date)
    if gold.empty:
        logging.error("Failed to fetch Gold data from yfinance")
        return pd.DataFrame()
    gold.reset_index(inplace=True)
    gold.rename(columns={'Date': 'date', 'Close': 'gold_price'}, inplace=True)
    logging.info(f"Successfully fetched {len(gold)} rows of Gold data")
    return gold


# ==============================================
# Fear and greed index (Alternative.me)
# ==============================================

def fetch_fear_and_greed_index():
    """
    Fetch the Fear and Greed Index data from Alternative.me API.
    """
    logging.info("Fetching Fear and Greed Index data")
    try:
        # 1. Fetch the raw CSV (in‑memory)
        url = "https://api.alternative.me/fng/?limit=0&format=csv"
        response = requests.get(url)
        response.raise_for_status()
        logging.info("Successfully fetched raw Fear and Greed Index data")

        # 2. Split into lines and keep only the rows with exactly 2 commas (i.e. date,value,classification)
        #    and drop any “metadata” lines
        lines = response.text.splitlines()
        data_lines = [
            line for line in lines
            if line.count(",") == 2
            and "metadata" not in line.lower()
        ]
        logging.info(f"Filtered {len(data_lines)} valid rows from the raw data")

        # 3. Stitch back into one CSV-formatted string
        csv_data = "\n".join(data_lines)

        # 4. Read into pandas (first row is header)
        fng_df = pd.read_csv(StringIO(csv_data), names=["date", "fng_value", "fng_classification"], header=0)
        
        logging.info(f"Successfully parsed Fear and Greed Index data into DataFrame with {len(fng_df)} rows")

        return fng_df
    except Exception as e:
        logging.error(f"Failed to fetch Fear and Greed Index data: {e}")
        return pd.DataFrame()

# ==============================================
# Fetch All Data
# ==============================================
def fetch_all_data():
    logging.info("Starting to fetch all data")
    
   # Fetch FRED data
    dgs = fetch_fred_data('DGS2')   # 2-Year Treasury Yield
    fedfunds = fetch_fedfunds_data()  # Effective Federal Funds Rate
    
    # Fetch data from Yahoo Finance
    btc = fetch_bitcoin_data()
    dxy = fetch_dxy_data()
    vix = fetch_vix_data()
    gold = fetch_gold_data()
    
    # Fetch fear and greed index from Alternative.me
    fng = fetch_fear_and_greed_index()
    
    # Organize into a dictionary for clarity
    data = {
        "dgs": dgs,
        "fedfunds": fedfunds,
        "bitcoin": btc,
        "dxy": dxy,
        "vix": vix,
        "gold": gold,
        "fng": fng
    }
    
    logging.info("Successfully fetched all data")
    return data
    
# ==============================================
# Run the Code
# ==============================================
if __name__ == '__main__':
    
    logging.info("Starting the data fetching process")
    all_data = fetch_all_data()
    
    # Save datasets to CSV files if available
    for key, df in all_data.items():
        if not df.empty:
            output_path = f'../data/{key}.csv'
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {key} data to CSV at {output_path}")
        else:
            logging.error(f"No data fetched for {key}. CSV not created.")