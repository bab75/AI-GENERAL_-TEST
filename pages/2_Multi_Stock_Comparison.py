import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.data_fetcher import get_stock_data
from utils.technical_indicators import add_selected_indicators
from components.timeframe_selector import display_timeframe_selector

st.set_page_config(
    page_title="Multi-Stock Comparison - Stock Analysis Platform",
    page_icon="🔍",
    layout="wide"
)

def get_correlation_matrix(stocks_data):
    """Calculate correlation matrix between stocks."""
    corr_data = {}
    
    for ticker, data in stocks_data.items():
        if not data.empty:
            corr_data[ticker] = data['Close']
    
    if len(corr_data) > 1:
        return pd.DataFrame(corr_data).corr()
    return pd.DataFrame()

def normalize_data(df, column='Close'):
    """Normalize data to percentage change from first value."""
    if df.empty:
        return df
    
    first_value = df[column].iloc[0]
    df[f'Normalized_{column}'] = (df[column] / first_value - 1) * 100
    return df

def calculate_performance_metrics(df):
    """Calculate various performance metrics."""
    if df.empty or len(df) < 2:
        return {}
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate metrics
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    volatility = df['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
    sharpe = total_return / volatility if volatility != 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = (1 + df['Daily_Return'].fillna(0)).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    return {
        'Total Return (%)': total_return,
        'Annualized Volatility (%)': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown
    }

def main():
    st.title("🔍 Multi-Stock Comparison")
    
    st.markdown("""
    Compare up to 3 stocks side-by-side to analyze their relative performance, correlation, and technical indicators.
    This tool helps you understand how different stocks move in relation to each other and spot potential trends or divergences.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Comparison Settings")
        
        # Stock selection (up to 3)
        st.subheader("Select Stocks (up to 3)")
        
        ticker1 = st.text_input("Stock 1", "AAPL").upper()
        ticker2 = st.text_input("Stock 2", "MSFT").upper()
        ticker3 = st.text_input("Stock 3", "GOOGL").upper()
        
        # Filter out empty tickers
        tickers = [t for t in [ticker1, ticker2, ticker3] if t]
        
        if len(tickers) < 1:
            st.error("Please enter at least one ticker.")
        
        # Timeframe selection
        period, interval = display_timeframe_selector()
        
        # Indicators for comparison
        st.subheader("Select Indicators")
        selected_indicators = st.multiselect(
            "Choose indicators for comparison",
            ["RSI", "MACD", "Bollinger Bands", "Volume"],
            default=["RSI"]
        )
        
        # Compare button
        compare_button = st.button("Compare Stocks", type="primary")
    
    # Main content
    if len(tickers) > 0 and compare_button:
        st.header(f"Comparison of {', '.join(tickers)}")
        
        try:
            with st.spinner("Fetching and analyzing data..."):
                # Get data for all tickers
                stocks_data = {}
                for ticker in tickers:
                    df = get_stock_data(ticker, period=period, interval=interval)
                    
                    if not df.empty:
                        # Add selected indicators
                        df = add_selected_indicators(df, selected_indicators)
                        
                        # Add to the dictionary
                        stocks_data[ticker] = df
                
                if not stocks_data:
                    st.error("Could not retrieve data for any of the selected stocks.")
                    return
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["Price Comparison", "Correlation Analysis", "Performance Metrics", "Technical Indicators"])
                
                with tab1:
                    st.subheader("Price Comparison")
                    
                    # Create normalized price chart for comparison
                    fig = go.Figure()
                    
                    for ticker, data in stocks_data.items():
                        if not data.empty:
                            normalized_data = normalize_data(data.copy())
                            # Create hover text with original price information
                            hover_text = [
                                f"Date: {date.strftime('%Y-%m-%d')}<br>" +
                                f"Stock: {ticker}<br>" +
                                f"Change: {norm_val:.2f}%<br>" +
                                f"Close: ${close:.2f}<br>" +
                                f"Volume: {vol:,}"
                                for date, norm_val, close, vol in zip(
                                    data.index, 
                                    normalized_data['Normalized_Close'],
                                    data['Close'],
                                    data['Volume']
                                )
                            ]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=normalized_data.index,
                                    y=normalized_data['Normalized_Close'],
                                    name=ticker,
                                    mode='lines',
                                    hoverinfo='text',
                                    hovertext=hover_text
                                )
                            )
                    
                    fig.update_layout(
                        title="Normalized Price Comparison (% Change)",
                        xaxis_title="Date",
                        yaxis_title="% Change",
                        height=500,
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
                    
                    # Volume comparison
                    st.subheader("Volume Comparison")
                    
                    # Create subplots for volume
                    fig_volume = make_subplots(
                        rows=len(stocks_data),
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=[f"{ticker} - Volume" for ticker in stocks_data.keys()]
                    )
                    
                    for i, (ticker, data) in enumerate(stocks_data.items(), 1):
                        if not data.empty:
                            # Create hover text with more information
                            hover_text = [
                                f"Date: {date.strftime('%Y-%m-%d')}<br>" +
                                f"Stock: {ticker}<br>" +
                                f"Volume: {volume:,}<br>" +
                                f"Close: ${close:.2f}"
                                for date, volume, close in zip(
                                    data.index, 
                                    data['Volume'],
                                    data['Close']
                                )
                            ]
                            
                            fig_volume.add_trace(
                                go.Bar(
                                    x=data.index,
                                    y=data['Volume'],
                                    name=f"{ticker} Volume",
                                    marker_color=f'rgba({50*i}, {100*i}, 255, 0.7)',
                                    hoverinfo='text',
                                    hovertext=hover_text
                                ),
                                row=i, col=1
                            )
                    
                    fig_volume.update_layout(
                        height=300 * len(stocks_data),
                        template="plotly_dark",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                with tab2:
                    st.subheader("Correlation Analysis")
                    
                    # Calculate correlation matrix
                    corr_matrix = get_correlation_matrix(stocks_data)
                    
                    if not corr_matrix.empty:
                        # Display correlation matrix
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            aspect="auto",
                            title="Price Correlation Matrix"
                        )
                        
                        fig_corr.update_layout(
                            height=400,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Scatter plot matrix for more detailed view
                        st.subheader("Scatter Plot Matrix")
                        
                        # Prepare data for scatter plot
                        scatter_data = {}
                        for ticker, data in stocks_data.items():
                            if not data.empty:
                                scatter_data[ticker] = data['Close']
                        
                        if len(scatter_data) > 1:
                            scatter_df = pd.DataFrame(scatter_data)
                            
                            fig_scatter = px.scatter_matrix(
                                scatter_df,
                                dimensions=list(scatter_df.columns),
                                title="Stock Price Relationships"
                            )
                            
                            fig_scatter.update_layout(
                                height=500,
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Need at least two stocks with data for scatter plot matrix.")
                    else:
                        st.info("Need at least two stocks with data to calculate correlation.")
                
                with tab3:
                    st.subheader("Performance Metrics")
                    
                    # Calculate and display performance metrics
                    metrics_data = []
                    
                    for ticker, data in stocks_data.items():
                        if not data.empty:
                            metrics = calculate_performance_metrics(data.copy())
                            metrics['Stock'] = ticker
                            metrics_data.append(metrics)
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        # Reorder columns to have Stock first
                        if 'Stock' in metrics_df.columns:
                            cols = ['Stock'] + [col for col in metrics_df.columns if col != 'Stock']
                            metrics_df = metrics_df[cols]
                        
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Create bar chart for visual comparison
                        st.subheader("Visual Comparison")
                        
                        # Set up metrics to display
                        metrics_to_plot = [c for c in metrics_df.columns if c != 'Stock']
                        
                        for metric in metrics_to_plot:
                            fig_bar = px.bar(
                                metrics_df,
                                x='Stock',
                                y=metric,
                                title=f"{metric} by Stock",
                                color='Stock'
                            )
                            
                            fig_bar.update_layout(
                                height=300,
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("Could not calculate performance metrics with the available data.")
                
                with tab4:
                    st.subheader("Technical Indicators Comparison")
                    
                    for indicator in selected_indicators:
                        st.markdown(f"### {indicator} Comparison")
                        
                        fig_indicator = go.Figure()
                        
                        if indicator == "RSI":
                            for ticker, data in stocks_data.items():
                                if 'RSI' in data.columns:
                                    fig_indicator.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['RSI'],
                                            name=f"{ticker} RSI"
                                        )
                                    )
                            
                            # Add overbought/oversold lines
                            fig_indicator.add_hline(y=70, line_dash="dash", line_color="red")
                            fig_indicator.add_hline(y=30, line_dash="dash", line_color="green")
                            
                            fig_indicator.update_layout(
                                title="RSI Comparison",
                                xaxis_title="Date",
                                yaxis_title="RSI",
                                height=400
                            )
                        
                        elif indicator == "MACD":
                            # For MACD, create separate subplots for each stock
                            macd_fig = make_subplots(
                                rows=len(stocks_data),
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=[f"{ticker} - MACD" for ticker in stocks_data.keys()]
                            )
                            
                            for i, (ticker, data) in enumerate(stocks_data.items(), 1):
                                if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                                    macd_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['MACD'],
                                            name=f"{ticker} MACD"
                                        ),
                                        row=i, col=1
                                    )
                                    
                                    macd_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['MACD_signal'],
                                            name=f"{ticker} Signal"
                                        ),
                                        row=i, col=1
                                    )
                                    
                                    macd_fig.add_trace(
                                        go.Bar(
                                            x=data.index,
                                            y=data['MACD_hist'],
                                            name=f"{ticker} Histogram"
                                        ),
                                        row=i, col=1
                                    )
                            
                            macd_fig.update_layout(
                                height=300 * len(stocks_data),
                                template="plotly_dark",
                                showlegend=True
                            )
                            
                            st.plotly_chart(macd_fig, use_container_width=True)
                            continue
                        
                        elif indicator == "Bollinger Bands":
                            # For Bollinger Bands, create separate subplots for each stock
                            bb_fig = make_subplots(
                                rows=len(stocks_data),
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=[f"{ticker} - Bollinger Bands" for ticker in stocks_data.keys()]
                            )
                            
                            for i, (ticker, data) in enumerate(stocks_data.items(), 1):
                                if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                                    bb_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['Close'],
                                            name=f"{ticker} Close",
                                            line=dict(color='yellow')
                                        ),
                                        row=i, col=1
                                    )
                                    
                                    bb_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['BB_upper'],
                                            name=f"{ticker} Upper Band",
                                            line=dict(color='rgba(250, 0, 0, 0.7)')
                                        ),
                                        row=i, col=1
                                    )
                                    
                                    bb_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['BB_middle'],
                                            name=f"{ticker} Middle Band",
                                            line=dict(dash='dash', color='rgba(150, 150, 150, 0.7)')
                                        ),
                                        row=i, col=1
                                    )
                                    
                                    bb_fig.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['BB_lower'],
                                            name=f"{ticker} Lower Band",
                                            line=dict(color='rgba(0, 250, 0, 0.7)')
                                        ),
                                        row=i, col=1
                                    )
                            
                            bb_fig.update_layout(
                                height=300 * len(stocks_data),
                                template="plotly_dark",
                                showlegend=True
                            )
                            
                            st.plotly_chart(bb_fig, use_container_width=True)
                            continue
                        
                        elif indicator == "Volume":
                            # Skip as we already have volume comparison
                            continue
                        
                        # Display the indicator figure if not already handled
                        if fig_indicator.data:
                            fig_indicator.update_layout(
                                template="plotly_dark",
                                height=400,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig_indicator, use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif len(tickers) < 1:
        st.info("Please enter at least one stock ticker to begin comparison.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This tool provides a comparative view of multiple stocks. Use the insights gained to identify stocks that might 
    move together, stocks that might be outperforming their peers, or stocks showing divergent technical signals.
    """)

if __name__ == "__main__":
    main()
