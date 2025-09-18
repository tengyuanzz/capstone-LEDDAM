
import yfinance as yf
import pandas as pd

# List of ticker symbols
tickers = ["AAPL"]


def download_data():
    for ticker in tickers:
        df_xlstm = yf.download(ticker)
        df_xlstm = df_xlstm.reset_index(drop=True)
        # df_xlstm = df_xlstm[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")  # Get max historical data

        # Save CSV to datasets folder
        # df_xlstm.to_csv(rf"datasets/{ticker}_xlstm.csv", index=False, header=False)
        # df.to_csv(rf"datasets/{ticker}.csv")
        
        df_xlstm.to_csv(rf"datasets/AAPL_xlstm_raw.csv", index=False, header=False)
        # df.to_csv(rf"datasets/test.csv")

def make_sentiment_data():
    df_data = pd.read_csv(
        'datasets/AAPL_xlstm_date.csv',
        header= None,
        names=["Date", "Open", "High", "Low", "Close", "Volume"],
    )

    df_data['Date'] = pd.to_datetime(df_data['Date'], errors='coerce', utc=True)
    df_data['Date'] = df_data['Date'].dt.date
    df_data['Date'] = pd.to_datetime(df_data['Date'])

    df_sentiment = pd.read_csv('datasets/AAPL daily sentiment.csv')
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])

    df_comb = df_data.merge(
        df_sentiment,
        on='Date',
        how='outer'
    )
    start_date = pd.to_datetime('2014-01-01')
    end_date = pd.to_datetime('2018-12-31')
    df_comb = df_comb[(df_comb['Date'] >= start_date) & (df_comb['Date'] <= end_date)]

    
    df_comb['ohclv_missing'] = df_comb['Close'].isnull() # ohclv missing
    df_comb['ohclv_present'] = df_comb.loc[~df_comb['ohclv_missing'], 'Date'] # ohclv present
    df_comb['next_valid_ohclv'] = df_comb['ohclv_present'].bfill() # backfill the latet valid dates

    avg_sentiment = df_comb[df_comb['polarity'].notna()].groupby('next_valid_ohclv')['polarity'].mean()
    df_final = df_comb[~df_comb['ohclv_missing']].copy() # filter out the rows with ohclv data
    
    df_final['polarity'] = df_final['Date'].map(avg_sentiment) # map the average sentiment to the ohclv data
    df_comb=df_comb.drop(columns=['ohclv_missing','ohclv_present','next_valid_ohclv'])
    df_final=df_final.drop(columns=['ohclv_missing','ohclv_present','next_valid_ohclv'])

    df_final.to_csv(rf"datasets/AAPL_xlstm_sentiment_comb.csv", index=False, header=False)

make_sentiment_data()