import streamlit as st
import yfinance as yf

def display_stock_selector(default_ticker="AAPL"):
    """
    Display a stock ticker selection input with suggestions.
    
    Parameters:
    -----------
    default_ticker : str, default "AAPL"
        Default ticker to display
    
    Returns:
    --------
    str
        Selected ticker symbol
    """
    # Common stock tickers that can be suggested
    popular_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
        "V", "WMT", "JNJ", "PG", "UNH", "HD", "BAC", "XOM", "PFE", "NFLX"
    ]
    
    # Allow user to select from popular tickers or enter a custom one
    selection_method = st.radio(
        "Select stock by",
        ["Popular Tickers", "Custom Ticker"],
        horizontal=True
    )
    
    if selection_method == "Popular Tickers":
        ticker = st.selectbox(
            "Choose a stock ticker",
            popular_tickers
        )
    else:
        ticker = st.text_input(
            "Enter stock ticker symbol",
            value=default_ticker
        ).upper()
    
    # Display basic info about the selected ticker
    if ticker:
        try:
            # Try to get basic info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if 'shortName' in info:
                st.caption(f"Selected: {info.get('shortName', ticker)} ({ticker})")
                
                # Display sector and industry if available
                if 'sector' in info and 'industry' in info:
                    st.caption(f"Sector: {info['sector']} | Industry: {info['industry']}")
        except:
            # If there's an error, just display the ticker
            st.caption(f"Selected ticker: {ticker}")
    
    return ticker

def display_multi_stock_selector(default_tickers=None, max_stocks=3):
    """
    Display a multi-stock ticker selection input.
    
    Parameters:
    -----------
    default_tickers : list, default None
        Default tickers to display
    max_stocks : int, default 3
        Maximum number of stocks that can be selected
    
    Returns:
    --------
    list
        List of selected ticker symbols
    """
    if default_tickers is None:
        default_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Ensure we don't exceed the max stocks in defaults
    default_tickers = default_tickers[:max_stocks]
    
    # Common stock tickers that can be suggested
    popular_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
        "V", "WMT", "JNJ", "PG", "UNH", "HD", "BAC", "XOM", "PFE", "NFLX"
    ]
    
    st.subheader(f"Select Stocks (up to {max_stocks})")
    
    selected_tickers = []
    
    for i in range(max_stocks):
        # Default value
        default_value = default_tickers[i] if i < len(default_tickers) else ""
        
        # Input field
        ticker = st.text_input(
            f"Stock {i+1}",
            value=default_value
        ).upper()
        
        # Add to list if not empty
        if ticker:
            selected_tickers.append(ticker)
    
    # Suggestion for popular tickers
    with st.expander("Popular Tickers"):
        st.write("Click to copy: " + ", ".join(popular_tickers))
    
    return selected_tickers
