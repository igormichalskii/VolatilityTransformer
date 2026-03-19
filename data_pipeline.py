import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_data(symbol: str="SPY", vix_symbol: str="^VIX", start_date: str="2010-01-01"):
    print(f"Fetching data for {symbol} and {vix_symbol} starting from {start_date}...")

    spy = yf.Ticker(symbol).history(start=start_date)[['Close']]
    spy.rename(columns={'Close': 'close'}, inplace=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    vix = yf.Ticker(vix_symbol).history(start=start_date)[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

    df = spy.join(vix, how='inner')
    print("Calculating returns and volatility...")
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

    clean_df = df.dropna()
    output_file = f"{symbol}_TRANSFORMER_clean.parquet"
    clean_df.to_parquet(output_file)

    print(f"Data cleaned and saved to {output_file} | Shape: {clean_df.shape}")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_data()