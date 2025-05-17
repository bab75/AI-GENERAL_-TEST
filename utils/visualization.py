import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_price_chart(df, ticker, include_volume=True):
    """
    Create a price chart for the given stock data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data
    ticker : str
        Stock ticker symbol
    include_volume : bool, default True
        Whether to include volume in the chart
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the price chart
    """
    if df.empty:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"secondary_y": include_volume}]]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        secondary_y=False
    )
    
    # Add volume if requested
    if include_volume and 'Volume' in df.columns:
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume",
                marker_color='rgba(100, 100, 255, 0.3)',
                opacity=0.5
            ),
            secondary_y=True
        )
    
    # Set titles and labels
    fig.update_layout(
        title=f"{ticker} - Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    if include_volume:
        fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    return fig

def create_indicator_chart(df, indicator, ticker):
    """
    Create a chart for the given technical indicator.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing indicator data
    indicator : str
        Name of the indicator to plot
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the indicator chart
    """
    if df.empty:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add traces based on indicator type
    if indicator == 'RSI':
        # Create RSI plot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        # Set y-axis range
        fig.update_layout(yaxis=dict(range=[0, 100]))
    
    elif indicator == 'MACD':
        # Create MACD plot
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'))
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram', marker_color='rgba(180, 180, 255, 0.5)'))
    
    elif indicator == 'Bollinger Bands':
        # Create Bollinger Bands plot
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band', line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band', line=dict(color='green')))
    
    elif indicator == 'Stochastic':
        # Create Stochastic plot
        fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K'))
        fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D'))
        
        # Add overbought/oversold lines
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.add_hline(y=20, line_dash="dash", line_color="green")
        
        # Set y-axis range
        fig.update_layout(yaxis=dict(range=[0, 100]))
    
    elif indicator == 'ATR':
        # Create ATR plot
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR'))
    
    elif indicator == 'Moving Averages':
        # Create Moving Averages plot
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red')))
    
    # Set titles and labels
    fig.update_layout(
        title=f"{ticker} - {indicator}",
        xaxis_title="Date",
        yaxis_title=indicator,
        height=400,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix):
    """
    Create a heatmap for the correlation matrix.
    
    Parameters:
    -----------
    correlation_matrix : pandas.DataFrame
        Correlation matrix for stocks
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the correlation heatmap
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Price Correlation Matrix"
    )
    
    fig.update_layout(
        height=500,
        template="plotly_dark"
    )
    
    return fig

def create_performance_comparison(normalized_data):
    """
    Create a chart comparing the normalized performance of multiple stocks.
    
    Parameters:
    -----------
    normalized_data : dict
        Dictionary with tickers as keys and DataFrames with normalized price as values
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the performance comparison chart
    """
    fig = go.Figure()
    
    for ticker, data in normalized_data.items():
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Normalized_Close'],
                name=ticker,
                mode='lines'
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
    
    return fig

def create_options_payoff_diagram(stock_prices, payoff, breakeven=None, current_price=None, strike=None):
    """
    Create a payoff diagram for options strategies.
    
    Parameters:
    -----------
    stock_prices : array-like
        Array of stock prices to plot
    payoff : array-like
        Array of payoff values corresponding to stock prices
    breakeven : float, optional
        Breakeven price point
    current_price : float, optional
        Current stock price
    strike : float, optional
        Strike price for the option
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the options payoff diagram
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=stock_prices,
            y=payoff,
            mode='lines',
            name='Payoff',
            line=dict(width=3)
        )
    )
    
    # Add breakeven point if provided
    if breakeven is not None:
        fig.add_vline(
            x=breakeven,
            line_width=2,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Breakeven: ${breakeven:.2f}",
            annotation_position="top left"
        )
    
    # Add current price marker if provided
    if current_price is not None:
        fig.add_vline(
            x=current_price,
            line_width=2,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top right"
        )
    
    # Add strike price marker if provided
    if strike is not None:
        fig.add_vline(
            x=strike,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Strike: ${strike:.2f}",
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        title="Options Payoff Diagram",
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Profit/Loss ($)",
        height=500,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
