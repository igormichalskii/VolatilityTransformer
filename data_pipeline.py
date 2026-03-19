import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_hf_data(symbol="SPY", vix_symbol="^VIX"):
    print(f"1. Fetching intraday data for {symbol} and {vix_symbol}. (1-Hour Bars)...")

    spy = yf.download(symbol, period="720d", interval="1h")[['Close']]
    spy.rename(columns={'Close': 'close'}, inplace=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    vix = yf.download(vix_symbol, period="720d", interval="1h")[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

    print("2. Aligning market microstructure (1-Day Cycles)...")
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)

    df = spy.join(vix, how='inner')

    # Calculate hourly log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # THE UPGRADE: 7-Row Rolling Target (Exactly 1 full trading day of Yahoo data)
    df['realized_vol'] = df['log_return'].rolling(window=7).std() * np.sqrt(1764)

    clean_df = df.dropna()
    output_file = f"{symbol}_HF_clean.parquet"
    clean_df.to_parquet(output_file)

    print(f"3. Microstructure secured. Shape: {clean_df.shape}. Saved to {output_file}.")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_hf_data()