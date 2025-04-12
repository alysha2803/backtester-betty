import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from tqdm import tqdm

# ==================== CONFIGURATION ====================
# Base URL for REST API
BASE_URL = "https://api.datasource.cybotrade.rs"

# Your API key - replace with your actual key
API_KEY = "apikey"

# Symbols to fetch
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOLS_COINBASE = ["BTC-USD", "ETH-USD"]  # Coinbase uses different format

# Exchanges and market types
EXCHANGES = {
    "binance": ["spot", "linear"],
    "bybit": ["spot", "linear", "inverse"],
    "coinbase": [""]  # Coinbase has a different API structure
}

# Time intervals to fetch
INTERVALS = ["1h", "4h", "1d"]
COINBASE_INTERVALS = ["1h", "1d"]  # Coinbase doesn't support 4h intervals


# Onchain data providers and endpoints
ONCHAIN_PROVIDERS = {
    "cryptoquant": [
        "btc/exchange-flows/inflow?exchange=all_exchange&window=hour",
        "btc/exchange-flows/outflow?exchange=all_exchange&window=hour",
        "btc/market-data/open-interest?exchange=all_exchange&window=hour",
        "btc/market-data/coinbase-premium-index?window=hour",
        "eth/exchange-flows/inflow?exchange=all_exchange&window=hour",
        "eth/exchange-flows/outflow?exchange=all_exchange&window=hour",
        "eth/market-data/open-interest?exchange=all_exchange&window=hour"
    ],
    "glassnode": [
        "market/deltacap_usd?a=BTC&i=1h",
        "market/mvrv_z_score?a=BTC&i=24h",
        "blockchain/utxo_created_value_median?a=BTC&c=usd&i=24h",
        "market/mvrv_z_score?a=ETH&i=24h"
    ],
    "coinglass": [
        "futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=1h",
        "futures/openInterest/ohlc-history?exchange=Bybit&symbol=BTCUSDT&interval=1h",
        "futures/openInterest/ohlc-history?exchange=Binance&symbol=ETHUSDT&interval=1h",
        "futures/openInterest/ohlc-history?exchange=Bybit&symbol=ETHUSDT&interval=1h"
    ]
}

# Define output directories
DATA_DIR = "cybotrade_data"
MARKET_DATA_DIR = f"{DATA_DIR}/market_data"
ONCHAIN_DATA_DIR = f"{DATA_DIR}/onchain_data"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MARKET_DATA_DIR, exist_ok=True)
os.makedirs(ONCHAIN_DATA_DIR, exist_ok=True)
# Add this after the initial directory creation
for provider in ONCHAIN_PROVIDERS.keys():
    os.makedirs(f"{ONCHAIN_DATA_DIR}/{provider}", exist_ok=True)
    # Create subdirectories for each provider
    for endpoint in ONCHAIN_PROVIDERS[provider]:
        # Extract path elements
        path_parts = endpoint.split('/')
        if len(path_parts) > 1:
            subdir = f"{ONCHAIN_DATA_DIR}/{provider}/{path_parts[0]}"
            os.makedirs(subdir, exist_ok=True)
            if len(path_parts) > 2:
                subdir = f"{subdir}/{path_parts[1]}"
                os.makedirs(subdir, exist_ok=True)

# ==================== API REQUEST HELPERS ====================

def make_request(endpoint, params=None, max_retries=5, retry_delay=5):
    """
    Make a request to the Cybotrade API with retry logic and rate limit handling
    
    Args:
        endpoint: API endpoint path
        params: Query parameters
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries in seconds
    
    Returns:
        JSON response data
    """
    headers = {"X-API-Key": API_KEY}
    url = f"{BASE_URL}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                # Get rate limit reset time from headers
                reset_timestamp = int(response.headers.get('X-Api-Limit-Reset-Timestamp', time.time() * 1000 + 60000))
                current_time = int(time.time() * 1000)
                sleep_time = max(1, (reset_timestamp - current_time) / 1000)
                
                print(f"Rate limit hit. Sleeping for {sleep_time:.2f} seconds. Attempt {attempt + 1}/{max_retries}")
                time.sleep(sleep_time)
                continue
            
            # Handle other errors
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds. Attempt {attempt + 1}/{max_retries}")
                    time.sleep(sleep_time)
                    continue
                response.raise_for_status()
            
            # Parse successful response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                print(f"Request error: {e}. Retrying in {sleep_time} seconds. Attempt {attempt + 1}/{max_retries}")
                time.sleep(sleep_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise
    
    return None

def fetch_with_pagination(endpoint, params, data_key="data"):
    """
    Fetch data with pagination to get complete historical data
    
    Args:
        endpoint: API endpoint path
        params: Base query parameters
        data_key: Key for data in response
    
    Returns:
        List of all data points across pages
    """
    all_data = []
    current_params = params.copy()
    
    # Calculate end time if not provided
    if "end_time" not in current_params:
        current_params["end_time"] = int(datetime.now().timestamp() * 1000)
    
    # Calculate start time if not provided (3 years ago)
    if "start_time" not in current_params:
        # Start time is 3 years ago
        current_params["start_time"] = int((datetime.now() - timedelta(days=3*365)).timestamp() * 1000)
    
    # Set initial conditions
    more_data = True
    batch_size = 1000  # Adjust batch size as needed
    
    # Use original start and end times for progress tracking
    total_timespan = current_params["end_time"] - current_params["start_time"]
    progress_bar = tqdm(total=100, desc=f"Fetching {endpoint}")
    
    while more_data:
        # Adjust batch parameters - use end_time and limit, but keep other essential parameters
        query_params = {k: v for k, v in current_params.items() if k != "start_time"}
        query_params["limit"] = batch_size
        
        # Make API request
        try:
            response = make_request(endpoint, query_params)
            
            if not response or not response.get(data_key):
                # No data returned
                break
            
            # Get data from response
            data_batch = response[data_key]
            if not data_batch:
                break
                
            # Append data to result
            all_data.extend(data_batch)
            
            # Update progress bar
            if len(data_batch) > 0:
                time_covered = current_params["end_time"] - data_batch[-1]["start_time"]
                progress = min(100, int(100 * time_covered / total_timespan))
                progress_bar.update(progress - progress_bar.n)
            
            # Check if we need to fetch more data
            if len(data_batch) < batch_size:
                # Received fewer items than requested, likely at the end
                more_data = False
            elif min(item["start_time"] for item in data_batch) <= current_params["start_time"]:
                # We've reached our target start time
                more_data = False
            else:
                # Update end_time to fetch earlier data
                oldest_timestamp = min(item["start_time"] for item in data_batch)
                current_params["end_time"] = oldest_timestamp - 1
        
        except Exception as e:
            print(f"Error in pagination: {e}")
            break
    
    progress_bar.close()
    return all_data

# ==================== DATA FETCHING FUNCTIONS ====================

def fetch_market_data(exchange, market_type, symbol, interval):
    """
    Fetch OHLCV data for a specific symbol and interval
    
    Args:
        exchange: Exchange name (e.g., 'binance', 'bybit')
        market_type: Market type (e.g., 'spot', 'linear')
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '1h', '1d')
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {exchange}-{market_type} {symbol} {interval} data")
    
    # Adjust symbol for Coinbase
    api_symbol = symbol
    if exchange == "coinbase":
        if symbol == "BTCUSDT":
            api_symbol = "BTC-USD"
        elif symbol == "ETHUSDT":
            api_symbol = "ETH-USD"
    
    # Define endpoint
    endpoint = f"/{exchange}{'-' + market_type if market_type else ''}/candle"
    
    # Define parameters
    params = {
        "symbol": api_symbol,
        "interval": interval,
        "limit": 10000  # Maximum limit per request
    }
    
    # Use pagination to fetch all data
    candle_data = fetch_with_pagination(endpoint, params)
    
    if not candle_data:
        print(f"No data returned for {exchange}-{market_type} {symbol} {interval}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(candle_data)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
    
    # Sort by time
    df = df.sort_values('start_time').reset_index(drop=True)
    
    # Save to CSV
    filename = f"{MARKET_DATA_DIR}/{exchange}_{market_type}_{symbol}_{interval}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    
    return df

def fetch_onchain_data(provider, endpoint):
    """
    Fetch on-chain data from providers
    
    Args:
        provider: Provider name (e.g., 'cryptoquant', 'glassnode')
        endpoint: Specific endpoint path
    
    Returns:
        DataFrame with on-chain data
    """
    print(f"Fetching {provider} data: {endpoint}")
    
    # Define API endpoint
    api_endpoint = f"/{provider}/{endpoint}"
    
    # Define parameters
    params = {
        "limit": 10000,
        "flatten": True
    }
    
    # Use pagination to fetch all data
    onchain_data = fetch_with_pagination(api_endpoint, params)
    
    if not onchain_data:
        print(f"No data returned for {provider} {endpoint}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(onchain_data)
    
    # Convert timestamp to datetime
    if 'start_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
    
    # Sort by time
    if 'start_time' in df.columns:
        df = df.sort_values('start_time').reset_index(drop=True)
    
    # Create sanitized filename
    safe_endpoint = endpoint.replace('?', '_').replace('&', '_').replace('=', '_')
    filepath = f"{ONCHAIN_DATA_DIR}/{provider}/{safe_endpoint}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} records to {filepath}")  # Changed 'filename' to 'filepath'
    
    return df

# ==================== MAIN EXECUTION ====================

def fetch_all_data():
    """
    Fetch all market and on-chain data
    """
    # Start time
    start_time = time.time()
    print(f"Starting data fetch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will fetch 3 years of historical data for BTC and ETH from multiple sources.\n")
    
    # 1. Fetch market data
    print("\n== FETCHING MARKET DATA ==")
    market_data = {}
    
    for symbol in SYMBOLS:
        market_data[symbol] = {}
        
        for exchange, market_types in EXCHANGES.items():
            for market_type in market_types:
                # Use the appropriate intervals list based on the exchange
                interval_list = COINBASE_INTERVALS if exchange == "coinbase" else INTERVALS
                for interval in interval_list:
                    # Handle Coinbase special case with symbols
                    if exchange == "coinbase":
                        actual_symbol = "BTC-USD" if symbol == "BTCUSDT" else "ETH-USD"
                        key = f"{exchange}_{market_type}_{actual_symbol}_{interval}"
                    else:
                        key = f"{exchange}_{market_type}_{symbol}_{interval}"
                    
                    df = fetch_market_data(exchange, market_type, symbol, interval)
                    if not df.empty:
                        market_data[symbol][key] = df
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
    
    # 2. Fetch on-chain data
    print("\n== FETCHING ON-CHAIN DATA ==")
    onchain_data = {}
    
    for provider, endpoints in ONCHAIN_PROVIDERS.items():
        onchain_data[provider] = {}
        
        for endpoint in endpoints:
            key = endpoint.replace('?', '_').replace('&', '_').replace('=', '_')
            df = fetch_onchain_data(provider, endpoint)
            if not df.empty:
                onchain_data[provider][key] = df
            
            # Small delay to avoid rate limits
            time.sleep(1)
    
    # Finished
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nData fetching completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Data is saved in the '{DATA_DIR}' directory")
    
    # Return the data dictionary
    return {
        "market_data": market_data,
        "onchain_data": onchain_data
    }

def load_all_data():
    """
    Load all previously fetched data from CSV files
    
    Returns:
        Dictionary of DataFrames with all loaded data
    """
    data = {
        "market_data": {},
        "onchain_data": {}
    }
    
    # Load market data
    for filename in os.listdir(MARKET_DATA_DIR):
        if filename.endswith(".csv"):
            parts = filename.replace(".csv", "").split("_")
            
            if len(parts) >= 4:
                exchange = parts[0]
                market_type = parts[1]
                symbol = parts[2]
                interval = parts[3]
                
                # Initialize nested dictionaries if needed
                if symbol not in data["market_data"]:
                    data["market_data"][symbol] = {}
                
                key = f"{exchange}_{market_type}_{symbol}_{interval}"
                data["market_data"][symbol][key] = pd.read_csv(f"{MARKET_DATA_DIR}/{filename}")
                
                # Convert timestamp to datetime
                if 'start_time' in data["market_data"][symbol][key].columns:
                    data["market_data"][symbol][key]['datetime'] = pd.to_datetime(
                        data["market_data"][symbol][key]['start_time'], unit='ms'
                    )
    
    # Load on-chain data
    for provider in ONCHAIN_PROVIDERS.keys():
        provider_dir = f"{ONCHAIN_DATA_DIR}/{provider}"
        data["onchain_data"][provider] = {}
        
        if os.path.exists(provider_dir):
            for filename in os.listdir(provider_dir):
                if filename.endswith(".csv"):
                    key = filename.replace(".csv", "")
                    data["onchain_data"][provider][key] = pd.read_csv(f"{provider_dir}/{filename}")
                    
                    # Convert timestamp to datetime
                    if 'start_time' in data["onchain_data"][provider][key].columns:
                        data["onchain_data"][provider][key]['datetime'] = pd.to_datetime(
                            data["onchain_data"][provider][key]['start_time'], unit='ms'
                        )
    
    return data

# ==================== SAMPLE FUNCTIONS FOR WORKING WITH THE DATA ====================

def prepare_features(data, symbol="BTCUSDT", interval="1h"):
    """
    Prepare features for model training
    
    Args:
        data: Dictionary of data from load_all_data()
        symbol: Symbol to prepare features for
        interval: Time interval to use
    
    Returns:
        DataFrame with prepared features
    """
    # Find the best market data source
    market_sources = [s for s in data["market_data"].get(symbol, {}).keys() if interval in s]
    if not market_sources:
        print(f"No market data found for {symbol} at {interval} interval")
        return pd.DataFrame()
    
    # Prefer Binance spot data if available
    main_source = next((s for s in market_sources if "binance_spot" in s), market_sources[0])
    
    # Get the main price data
    price_data = data["market_data"][symbol][main_source].copy()
    
    # Create basic price features
    price_data['returns'] = price_data['close'].pct_change()
    price_data['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
    price_data['volatility_24h'] = price_data['returns'].rolling(24).std()
    
    # Create technical indicators
    price_data['ma_7'] = price_data['close'].rolling(7).mean()
    price_data['ma_25'] = price_data['close'].rolling(25).mean()
    price_data['ma_99'] = price_data['close'].rolling(99).mean()
    
    # Add RSI
    delta = price_data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    price_data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Create date features
    price_data['hour'] = price_data['datetime'].dt.hour
    price_data['day_of_week'] = price_data['datetime'].dt.dayofweek
    price_data['month'] = price_data['datetime'].dt.month
    
    # You can add more features from on-chain data here by merging on datetime
    
    # Clean up missing values
    price_data = price_data.dropna()
    
    return price_data

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Fetch all data
    print("Do you want to fetch new data or use existing data?")
    print("1. Fetch new data (may take 30+ minutes)")
    print("2. Use existing data (if available)")
    
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        all_data = fetch_all_data()
    else:
        # Try to load existing data
        try:
            all_data = load_all_data()
            print("Successfully loaded existing data")
            
            # Check if we have BTC and ETH data
            symbols_found = list(all_data["market_data"].keys())
            if not symbols_found:
                print("No market data found. Please fetch new data.")
                all_data = fetch_all_data()
            else:
                print(f"Found data for: {', '.join(symbols_found)}")
                
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Fetching new data instead...")
            all_data = fetch_all_data()
    
    # Example: Prepare features for model training
    print("\n== PREPARING FEATURES ==")
    btc_features = prepare_features(all_data, symbol="BTCUSDT", interval="1h")
    eth_features = prepare_features(all_data, symbol="ETHUSDT", interval="1h")
    
    # Show data summary
    print("\n== DATA SUMMARY ==")
    
    if not btc_features.empty:
        print(f"BTC data: {len(btc_features)} rows from {btc_features['datetime'].min()} to {btc_features['datetime'].max()}")
        print("Sample BTC features:")
        print(btc_features.head())
    
    if not eth_features.empty:
        print(f"ETH data: {len(eth_features)} rows from {eth_features['datetime'].min()} to {eth_features['datetime'].max()}")
        print("Sample ETH features:")
        print(eth_features.head())
    
    print("\nAll operations completed successfully!")
    print("The data is now ready to be used for backtesting and model training.")
