import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.data_fetcher import get_stock_data
from utils.technical_indicators import add_all_indicators
from components.stock_selector import display_stock_selector
from components.timeframe_selector import display_timeframe_selector
from components.indicator_selector import display_indicator_selector

st.set_page_config(
    page_title="Technical Analysis - Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Technical Analysis")
    
    st.markdown("""
    Analyze stocks using over 20 technical indicators. Select a stock, timeframe, and indicators to visualize.
    The platform provides a comprehensive suite of tools for technical analysis, helping you make informed trading decisions.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Analysis Settings")
        
        # Stock selection
        ticker = display_stock_selector()
        
        # Timeframe selection
        period, interval = display_timeframe_selector()
        
        # Technical indicators selection
        selected_indicators = display_indicator_selector()
        
        # Apply button
        analyze_button = st.button("Analyze", type="primary")
    
    # Main content area
    if ticker and analyze_button:
        st.header(f"Technical Analysis for {ticker}")
        
        try:
            with st.spinner(f"Fetching and analyzing data for {ticker}..."):
                # Get stock data
                df = get_stock_data(ticker, period=period, interval=interval)
                
                if df.empty:
                    st.error(f"No data available for {ticker} with the selected parameters.")
                    return
                
                # Add all indicators to the dataframe
                df = add_all_indicators(df)
                
                # Get basic stock info
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    current_price = df['Close'].iloc[-1]
                    previous_close = df['Close'].iloc[-2] if len(df) > 1 else None
                    
                    if previous_close is not None:
                        change = current_price - previous_close
                        percent_change = (change / previous_close) * 100
                        st.metric(
                            label=f"{ticker} - Current Price",
                            value=f"${current_price:.2f}",
                            delta=f"{change:.2f} ({percent_change:.2f}%)"
                        )
                    else:
                        st.metric(
                            label=f"{ticker} - Current Price",
                            value=f"${current_price:.2f}"
                        )
                
                with info_col2:
                    volume = df['Volume'].iloc[-1]
                    avg_volume = df['Volume'].mean()
                    volume_change = ((volume - avg_volume) / avg_volume) * 100
                    
                    st.metric(
                        label="Volume",
                        value=f"{volume:,.0f}",
                        delta=f"{volume_change:.2f}% from avg"
                    )
                
                with info_col3:
                    high_52w = df['High'].max()
                    low_52w = df['Low'].min()
                    st.metric(
                        label="Range",
                        value=f"${high_52w:.2f} - ${low_52w:.2f}"
                    )
                
                # Create main price chart
                st.subheader("Price Chart")
                
                # Determine how many rows we need for the subplots
                indicator_count = len(selected_indicators)
                row_count = 1 + (indicator_count if indicator_count > 0 else 0)
                
                # Create subplot grid
                row_heights = [0.6] + [0.4 / indicator_count] * indicator_count if indicator_count > 0 else [1]
                fig = make_subplots(
                    rows=row_count, 
                    cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=row_heights
                )
                
                # Add price candlestick chart with improved hover template
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="Price",
                        hoverinfo="text",
                        hovertext=[
                            f"Date: {date.strftime('%Y-%m-%d')}<br>" +
                            f"Open: ${open:.2f}<br>" +
                            f"High: ${high:.2f}<br>" +
                            f"Low: ${low:.2f}<br>" +
                            f"Close: ${close:.2f}<br>" +
                            f"Volume: {volume:,}"
                            for date, open, high, low, close, volume in zip(
                                df.index, df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
                            )
                        ]
                    ),
                    row=1, col=1
                )
                
                # Volume as bar chart at the bottom of price chart
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name="Volume",
                        marker_color='rgba(100, 100, 255, 0.3)',
                        opacity=0.5,
                        yaxis="y2"
                    ),
                    row=1, col=1
                )
                
                # Add volume axis to price chart
                fig.update_layout(
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    )
                )
                
                # Add selected indicators
                current_row = 2
                for indicator in selected_indicators:
                    if indicator == 'RSI':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['RSI'],
                                name="RSI"
                            ),
                            row=current_row, col=1
                        )
                        
                        # Add overbought/oversold lines
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                    
                    elif indicator == 'MACD':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['MACD'],
                                name="MACD"
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['MACD_signal'],
                                name="Signal"
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df['MACD_hist'],
                                name="MACD Histogram",
                                marker_color='rgba(180, 180, 255, 0.5)'
                            ),
                            row=current_row, col=1
                        )
                    
                    elif indicator == 'Bollinger Bands':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['Close'],
                                name="Close",
                                line=dict(color='yellow')
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['BB_upper'],
                                name="Upper Band",
                                line=dict(color='rgba(250, 0, 0, 0.7)')
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['BB_middle'],
                                name="Middle Band",
                                line=dict(dash='dash', color='rgba(150, 150, 150, 0.7)')
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['BB_lower'],
                                name="Lower Band",
                                line=dict(color='rgba(0, 250, 0, 0.7)')
                            ),
                            row=current_row, col=1
                        )
                    
                    elif indicator == 'Stochastic Oscillator':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['%K'],
                                name="%K"
                            ),
                            row=current_row, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['%D'],
                                name="%D"
                            ),
                            row=current_row, col=1
                        )
                        
                        # Add overbought/oversold lines
                        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
                    
                    elif indicator == 'ATR':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['ATR'],
                                name="ATR"
                            ),
                            row=current_row, col=1
                        )
                    
                    elif indicator == 'OBV':
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['OBV'],
                                name="OBV"
                            ),
                            row=current_row, col=1
                        )
                    
                    current_row += 1
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} - Technical Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=800,
                    template="plotly_dark",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical summary
                st.subheader("Technical Summary")
                
                # Get the most recent values for the indicators
                last_row = df.iloc[-1]
                
                # Prepare summary table
                summary_data = []
                
                if 'RSI' in df.columns:
                    rsi_value = last_row['RSI']
                    if rsi_value > 70:
                        rsi_signal = "Overbought"
                    elif rsi_value < 30:
                        rsi_signal = "Oversold"
                    else:
                        rsi_signal = "Neutral"
                    
                    summary_data.append({
                        "Indicator": "RSI",
                        "Value": f"{rsi_value:.2f}",
                        "Signal": rsi_signal
                    })
                
                if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                    macd = last_row['MACD']
                    signal = last_row['MACD_signal']
                    
                    if macd > signal:
                        macd_signal = "Bullish"
                    else:
                        macd_signal = "Bearish"
                    
                    summary_data.append({
                        "Indicator": "MACD",
                        "Value": f"{macd:.2f}",
                        "Signal": macd_signal
                    })
                
                if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                    price = last_row['Close']
                    upper = last_row['BB_upper']
                    lower = last_row['BB_lower']
                    
                    if price > upper:
                        bb_signal = "Overbought"
                    elif price < lower:
                        bb_signal = "Oversold"
                    else:
                        bb_signal = "Neutral"
                    
                    summary_data.append({
                        "Indicator": "Bollinger Bands",
                        "Value": f"{price:.2f} (UB: {upper:.2f}, LB: {lower:.2f})",
                        "Signal": bb_signal
                    })
                
                if '%K' in df.columns and '%D' in df.columns:
                    k = last_row['%K']
                    d = last_row['%D']
                    
                    if k > 80 and d > 80:
                        stoch_signal = "Overbought"
                    elif k < 20 and d < 20:
                        stoch_signal = "Oversold"
                    elif k > d:
                        stoch_signal = "Bullish Crossover"
                    elif k < d:
                        stoch_signal = "Bearish Crossover"
                    else:
                        stoch_signal = "Neutral"
                    
                    summary_data.append({
                        "Indicator": "Stochastic",
                        "Value": f"K: {k:.2f}, D: {d:.2f}",
                        "Signal": stoch_signal
                    })
                
                # Show summary table
                if summary_data:
                    st.table(pd.DataFrame(summary_data))
                
                # Show raw data in expandable section
                with st.expander("View Raw Data"):
                    st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif not ticker:
        st.info("Please select a stock ticker to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** Technical analysis should be used as one of many tools in your investment decision process. 
    Past performance is not indicative of future results.
    """)

if __name__ == "__main__":
    main()
