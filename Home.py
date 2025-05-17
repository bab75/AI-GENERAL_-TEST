import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from utils.data_fetcher import get_stock_data, get_market_indices
import os
import zipfile
from io import BytesIO

st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



def main():
    # Add a reset button to the sidebar
    with st.sidebar:
        st.sidebar.markdown("## Analysis Options")
        if st.button("🔄 Reset All Analysis", use_container_width=True, 
                   help="Clear all session data and reset the application"):
            # Clear all session state data
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
       
    st.title("📊 Stock Analysis Platform")
    
    # Introduction section
    st.markdown("""
    Welcome to the comprehensive Stock Analysis Platform! This tool provides advanced features for:
    
    * **Technical Analysis**: 20+ indicators including RSI, MACD, Bollinger Bands
    * **Multi-Stock Comparison**: Compare up to 3 stocks side-by-side
    * **Options Chain Analysis**: Analyze options data and Greeks
    * **Data Export**: Download data in various formats
    
    Get started by exploring the market overview below or navigate to a specific analysis page using the sidebar.
    """)
    
    # Market overview section
    st.header("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://pixabay.com/get/g1a3d12bda1246f5387fa336c41c2683069304a32abaa9aa80090d87667823041feb646ca6646d9c595d53ae7a3855d13b74fc5395b09f35420282004f35d0c39_1280.jpg", 
                 caption="Stock Market Dashboard", use_container_width=True)
        
        # Show major indices
        st.subheader("Major Indices")
        indices = get_market_indices()
        st.dataframe(indices, use_container_width=True)
        
    with col2:
        st.image("https://pixabay.com/get/g2d9d68b0df5f4c1b8882a924814daca0387cec51833c1538e03f7f819926d3984a57fb0825964a62e01cb394ec5f752f4eb81566e90259289f71658dd2406ca0_1280.jpg", 
                 caption="Financial Charts", use_container_width=True)
        
        # Quick stock lookup
        st.subheader("Quick Stock Lookup")
        default_ticker = "AAPL"
        ticker = st.text_input("Enter a ticker symbol", default_ticker).upper()
        
        if ticker:
            try:
                # Get stock info
                stock_info = yf.Ticker(ticker).info
                
                # Extract relevant info
                name = stock_info.get('shortName', 'N/A')
                current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 'N/A'))
                previous_close = stock_info.get('previousClose', 'N/A')
                
                if current_price != 'N/A' and previous_close != 'N/A':
                    change = current_price - previous_close
                    percent_change = (change / previous_close) * 100
                    
                    # Display basic stock info
                    st.metric(
                        label=f"{name} ({ticker})",
                        value=f"${current_price:.2f}",
                        delta=f"{change:.2f} ({percent_change:.2f}%)"
                    )
                else:
                    st.error(f"Could not retrieve complete data for {ticker}")
                    
                # Display mini chart
                data = get_stock_data(ticker, period="1mo")
                if not data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="",
                        yaxis_title="Price ($)",
                        title=f"{ticker} - 1 Month"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error retrieving data: {e}")
    
    # Feature showcase
    st.header("Platform Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.image("https://pixabay.com/get/g76c2465550d563b78dce4e3f9bcb675a2b92333af2fa2114167cc5f717e1ea94ea1bba2ed1e996a0ba0987bd8241bca37e6613dc5b7fdcd89e2558914b6efc82_1280.jpg", caption="Technical Analysis")
        st.markdown("""
        ### Technical Analysis
        * 20+ Technical Indicators
        * Multiple Timeframes
        * Interactive Charts
        * Pattern Recognition
        """)
        st.page_link("pages/1_Technical_Analysis.py", label="Go to Technical Analysis", icon="📈")
    
    with feature_col2:
        st.image("https://pixabay.com/get/g95127649948187153c80f07d217c3a036a744d0d52b39e190b81caf6fa7d3aea1e563effb7c3bce5f2333ef75d1d258723c070e332e98cb64307a346a185ca9d_1280.jpg", caption="Stock Comparison")
        st.markdown("""
        ### Multi-Stock Comparison
        * Side-by-Side Comparison
        * Correlation Analysis
        * Performance Metrics
        * Synchronized Charts
        """)
        st.page_link("pages/2_Multi_Stock_Comparison.py", label="Compare Stocks", icon="🔍")
    
    with feature_col3:
        st.image("https://pixabay.com/get/gbbe713eff09de32189dd604837386d5ab066da33c24cb2ea4941817f44264b3466aa3b529cb86734b367afa9e34c56bef028ddbdcc64f84e8982ec25fc8f3eb6_1280.jpg", caption="Options Analysis")
        st.markdown("""
        ### Options Analysis
        * Options Chain Visualization
        * Greeks Calculations
        * Strategy Analysis
        * Expiration Selection
        """)
        st.page_link("pages/3_Options_Analysis.py", label="Analyze Options", icon="🧮")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by yfinance, Streamlit, Plotly, and Pandas")
    st.markdown("© 2023 Stock Analysis Platform | Data provided for informational purposes only")

if __name__ == "__main__":
    main()
