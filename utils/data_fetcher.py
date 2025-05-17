import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch stock data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str, default "1y"
        Period to fetch data for (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    interval : str, default "1d"
        Interval between data points (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the stock data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        # Check if data is empty
        if df.empty:
            return pd.DataFrame()
        
        # Handle timezone for consistent indexing
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_multiple_stocks_data(tickers, period="1y", interval="1d"):
    """
    Fetch data for multiple stocks.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    period : str, default "1y"
        Period to fetch data for
    interval : str, default "1d"
        Interval between data points
    
    Returns:
    --------
    dict
        Dictionary with tickers as keys and DataFrames as values
    """
    result = {}
    for ticker in tickers:
        result[ticker] = get_stock_data(ticker, period, interval)
    return result

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_market_indices():
    """
    Fetch data for major market indices.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing index data
    """
    indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225']
    index_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'FTSE 100', 'Nikkei 225']
    
    data = []
    
    for idx, name in zip(indices, index_names):
        try:
            index_data = yf.Ticker(idx).history(period="2d")
            
            if not index_data.empty and len(index_data) >= 2:
                current = index_data['Close'].iloc[-1]
                previous = index_data['Close'].iloc[-2]
                change = current - previous
                percent_change = (change / previous) * 100
                
                data.append({
                    'Index': name,
                    'Price': current,
                    'Change': change,
                    'Change (%)': percent_change
                })
        except Exception as e:
            print(f"Error fetching data for {idx}: {e}")
    
    if not data:
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=['Index', 'Price', 'Change', 'Change (%)'])
    
    return pd.DataFrame(data)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(ticker):
    """
    Fetch detailed information about a stock.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    dict
        Dictionary containing stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        result = {
            'symbol': ticker,
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'previous_close': info.get('previousClose', 'N/A')
        }
        
        return result
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {
            'symbol': ticker,
            'name': 'N/A',
            'error': str(e)
        }
