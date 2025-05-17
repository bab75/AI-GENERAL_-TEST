import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64
import io
import json

from utils.data_fetcher import get_stock_data
from utils.technical_indicators import add_all_indicators
from components.stock_selector import display_stock_selector
from components.timeframe_selector import display_timeframe_selector
from utils.export_utils import download_link

st.set_page_config(
    page_title="Export Data - Stock Analysis Platform",
    page_icon="💾",
    layout="wide"
)

def main():
    st.title("💾 Export Data")
    
    st.markdown("""
    Export stock data, technical indicators, and charts for further analysis or record-keeping.
    This tool allows you to download data in various formats and export visualizations as images.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Export Settings")
        
        # Stock selection
        ticker = display_stock_selector()
        
        # Timeframe selection
        period, interval = display_timeframe_selector()
        
        # Data to include
        st.subheader("Data to Include")
        include_price = st.checkbox("Price Data", value=True)
        include_indicators = st.checkbox("Technical Indicators", value=True)
        include_stats = st.checkbox("Summary Statistics", value=True)
        
        # Export format
        st.subheader("Export Format")
        export_format = st.radio(
            "Select format",
            ["CSV", "Excel", "JSON"],
            index=0
        )
        
        # Fetch button
        fetch_button = st.button("Fetch Data", type="primary")
    
    # Main content
    if ticker and fetch_button:
        st.header(f"Data for {ticker}")
        
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                # Get stock data
                df = get_stock_data(ticker, period=period, interval=interval)
                
                if df.empty:
                    st.error(f"No data available for {ticker} with the selected parameters.")
                    return
                
                # Add technical indicators if requested
                if include_indicators:
                    df = add_all_indicators(df)
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Display price chart
                if include_price:
                    st.subheader("Price Chart")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"
                        )
                    )
                    
                    fig.update_layout(
                        title=f"{ticker} - Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display summary statistics
                if include_stats:
                    st.subheader("Summary Statistics")
                    
                    # Calculate daily returns
                    df['Daily_Return'] = df['Close'].pct_change()
                    
                    # Basic stats
                    start_date = df.index[0].strftime('%Y-%m-%d')
                    end_date = df.index[-1].strftime('%Y-%m-%d')
                    days = (df.index[-1] - df.index[0]).days
                    
                    price_start = df['Close'].iloc[0]
                    price_end = df['Close'].iloc[-1]
                    price_change = price_end - price_start
                    price_change_pct = (price_change / price_start) * 100
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric(
                            label="Date Range",
                            value=f"{start_date} to {end_date}",
                            delta=f"{days} days"
                        )
                    
                    with stats_col2:
                        st.metric(
                            label="Price Change",
                            value=f"${price_change:.2f}",
                            delta=f"{price_change_pct:.2f}%"
                        )
                    
                    with stats_col3:
                        st.metric(
                            label="Volatility (Annualized)",
                            value=f"{df['Daily_Return'].std() * np.sqrt(252) * 100:.2f}%"
                        )
                    
                    # More detailed stats
                    stats_data = {
                        'Metric': [
                            'Start Price', 'End Price', 'Min Price', 'Max Price',
                            'Average Price', 'Average Volume', 'Total Volume',
                            'Daily Return (Avg)', 'Daily Return (Min)', 'Daily Return (Max)',
                            'Daily Return (Std)'
                        ],
                        'Value': [
                            f"${price_start:.2f}", f"${price_end:.2f}",
                            f"${df['Low'].min():.2f}", f"${df['High'].max():.2f}",
                            f"${df['Close'].mean():.2f}", f"{df['Volume'].mean():.0f}",
                            f"{df['Volume'].sum():.0f}", f"{df['Daily_Return'].mean() * 100:.2f}%",
                            f"{df['Daily_Return'].min() * 100:.2f}%", f"{df['Daily_Return'].max() * 100:.2f}%",
                            f"{df['Daily_Return'].std() * 100:.2f}%"
                        ]
                    }
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.table(stats_df)
                
                # Prepare data for export
                export_df = df.copy()
                
                # Create export buttons
                st.subheader("Export Options")
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    if export_format == "CSV":
                        csv = export_df.to_csv()
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{ticker}_{period}_{interval}.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            export_df.to_excel(writer, sheet_name=f"{ticker} Data")
                            
                            if include_stats:
                                stats_df.to_excel(writer, sheet_name="Summary Stats")
                        
                        st.download_button(
                            label="Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"{ticker}_{period}_{interval}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    
                    elif export_format == "JSON":
                        # Convert datetime index to strings
                        export_df.index = export_df.index.strftime('%Y-%m-%d %H:%M:%S')
                        json_data = export_df.to_json(orient="index")
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"{ticker}_{period}_{interval}.json",
                            mime="application/json"
                        )
                
                with export_col2:
                    # Export chart as HTML
                    if include_price:
                        fig_html = fig.to_html(include_plotlyjs="cdn")
                        st.download_button(
                            label="Download Chart (HTML)",
                            data=fig_html,
                            file_name=f"{ticker}_chart_{period}_{interval}.html",
                            mime="text/html"
                        )
                
                with export_col3:
                    # Link to external resources
                    st.markdown(f"### External Resources")
                    st.markdown(f"[View on Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")
                    st.markdown(f"[View on TradingView](https://www.tradingview.com/symbols/{ticker}/)")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif not ticker:
        st.info("Please select a stock ticker to begin.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This tool provides data exports for educational and personal use. The data is sourced from Yahoo Finance 
    through the yfinance library. Please check the terms of service for any data provider you use and ensure compliance 
    with their requirements.
    """)

if __name__ == "__main__":
    main()
