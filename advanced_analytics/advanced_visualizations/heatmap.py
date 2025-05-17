"""
Advanced Heatmap Visualization Module

This module provides enhanced heatmap visualization capabilities for stock analysis,
including correlation heatmaps, sector performance heatmaps, and volatility heatmaps.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_correlation_heatmap(data: Dict[str, pd.DataFrame], 
                               column: str = 'Close',
                               period: str = 'all',
                               normalize: bool = True,
                               title: str = 'Stock Price Correlation Heatmap') -> go.Figure:
    """
    Create a correlation heatmap for multiple stocks.
    
    Args:
        data: Dictionary with ticker symbols as keys and DataFrames as values
        column: Column to use for correlation (default: 'Close')
        period: Time period to analyze ('all' or number of days)
        normalize: Whether to normalize prices before correlation
        title: Plot title
        
    Returns:
        Plotly figure object with correlation heatmap
    """
    # Create a new DataFrame with the closing prices of each stock
    df_combined = pd.DataFrame()
    
    for ticker, df in data.items():
        if df.empty or column not in df.columns:
            continue
            
        # Limit to the specified period
        if period != 'all' and period.isdigit():
            df = df.tail(int(period))
            
        # Use the specified column or fall back to 'Close'
        values = df[column].copy()
        
        # Normalize if requested
        if normalize:
            values = values / values.iloc[0] if not values.empty else values
            
        df_combined[ticker] = values
    
    # Calculate correlation matrix
    correlation_matrix = df_combined.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        zmin=-1, zmax=1,
        colorscale='RdBu_r',
        colorbar=dict(title=dict(text='Correlation')),
        hoverongaps=False
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=900,
        xaxis=dict(title='', tickangle=-45),
        yaxis=dict(title=''),
        margin=dict(l=60, r=60, t=80, b=80)
    )
    
    # Add annotations (correlation values)
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            if i == j:
                annotations.append(dict(
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    text='1.00',
                    showarrow=False,
                    font=dict(color='white' if abs(value) > 0.5 else 'black')
                ))
            else:
                annotations.append(dict(
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    text=f'{value:.2f}',
                    showarrow=False,
                    font=dict(color='white' if abs(value) > 0.5 else 'black')
                ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_sector_performance_heatmap(data: pd.DataFrame, 
                                     sector_column: str = 'Sector',
                                     performance_column: str = 'Price_Change_Pct',
                                     title: str = 'Sector Performance Heatmap') -> go.Figure:
    """
    Create a heatmap showing performance by sector.
    
    Args:
        data: DataFrame with stock data including sector and performance metrics
        sector_column: Column name for sector
        performance_column: Column name for performance metric
        title: Plot title
        
    Returns:
        Plotly figure object with sector performance heatmap
    """
    if sector_column not in data.columns or performance_column not in data.columns:
        logger.error(f"Required columns not found in data")
        return go.Figure()
    
    # Group by sector and calculate average performance
    sector_performance = data.groupby(sector_column)[performance_column].mean().reset_index()
    
    # Sort by performance (descending)
    sector_performance = sector_performance.sort_values(performance_column, ascending=False)
    
    # Create diverging colorscale
    max_abs_val = max(abs(sector_performance[performance_column].min()), 
                      abs(sector_performance[performance_column].max()))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[sector_performance[performance_column].values],
        x=sector_performance[sector_column].values,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-max_abs_val,
        zmax=max_abs_val,
        colorbar=dict(title='Avg % Change'),
        hovertemplate='Sector: %{x}<br>Performance: %{z:.2f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=400,
        width=1000,
        xaxis=dict(title='', tickangle=-45),
        yaxis=dict(title='', showticklabels=False),
        margin=dict(l=60, r=60, t=80, b=120)
    )
    
    # Add annotations (performance values)
    annotations = []
    for i, (sector, performance) in enumerate(zip(sector_performance[sector_column], 
                                                sector_performance[performance_column])):
        color = 'white' if abs(performance) > max_abs_val * 0.3 else 'black'
        annotations.append(dict(
            x=sector,
            y=0,
            text=f'{performance:.2f}%',
            showarrow=False,
            font=dict(color=color, size=12)
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_multi_timeframe_correlation_heatmap(data: Dict[str, pd.DataFrame], 
                                               column: str = 'Close',
                                               timeframes: List[int] = [30, 90, 180, 365],
                                               title: str = 'Multi-Timeframe Correlation Analysis') -> go.Figure:
    """
    Create a multi-panel correlation heatmap for different timeframes.
    
    Args:
        data: Dictionary with ticker symbols as keys and DataFrames as values
        column: Column to use for correlation
        timeframes: List of timeframes in days to analyze
        title: Plot title
        
    Returns:
        Plotly figure object with multi-timeframe correlation heatmaps
    """
    # Determine number of rows based on timeframes
    rows = (len(timeframes) + 1) // 2  # Ceiling division
    cols = min(2, len(timeframes))
    
    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, 
                        subplot_titles=[f'{tf}-Day Correlation' for tf in timeframes])
    
    # Process each timeframe
    for i, timeframe in enumerate(timeframes):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Create a new DataFrame with the closing prices of each stock
        df_combined = pd.DataFrame()
        
        for ticker, df in data.items():
            if df.empty or column not in df.columns:
                continue
                
            # Limit to the specified timeframe
            values = df[column].tail(timeframe).copy()
            
            # Normalize
            values = values / values.iloc[0] if not values.empty else values
                
            df_combined[ticker] = values
        
        # Calculate correlation matrix
        correlation_matrix = df_combined.corr()
        
        # Add heatmap to subplot
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                zmin=-1, zmax=1,
                colorscale='RdBu_r',
                showscale=i == 0,  # Only show colorbar for first heatmap
                hoverongaps=False,
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update axis labels
        fig.update_xaxes(title_text='', tickangle=-45, row=row, col=col)
        fig.update_yaxes(title_text='', row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=300 * rows,
        width=1200,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

def create_volatility_heatmap(data: Dict[str, pd.DataFrame], 
                             timeframe: int = 30,
                             title: str = 'Stock Volatility Heatmap') -> go.Figure:
    """
    Create a heatmap showing volatility of multiple stocks over time.
    
    Args:
        data: Dictionary with ticker symbols as keys and DataFrames as values
        timeframe: Timeframe in days to analyze
        title: Plot title
        
    Returns:
        Plotly figure object with volatility heatmap
    """
    # Create a new DataFrame for volatilities
    volatility_df = pd.DataFrame()
    
    for ticker, df in data.items():
        if df.empty or 'Close' not in df.columns:
            continue
            
        # Calculate daily returns
        returns = df['Close'].pct_change()
        
        # Calculate rolling volatility (20-day window, annualized)
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Limit to the specified timeframe
        volatility = volatility.tail(timeframe)
        
        # Add to the volatility DataFrame
        volatility_df[ticker] = volatility
    
    # Prepare data for heatmap (reshape to wide format)
    volatility_df = volatility_df.reset_index()
    
    # Convert datetime index to string dates for plotting
    if pd.api.types.is_datetime64_any_dtype(volatility_df['index']):
        volatility_df['date'] = volatility_df['index'].dt.strftime('%Y-%m-%d')
    else:
        volatility_df['date'] = volatility_df['index']
    
    # Drop the original index column
    volatility_df = volatility_df.drop('index', axis=1)
    
    # Melt the DataFrame to long format for heatmap
    volatility_long = pd.melt(
        volatility_df, 
        id_vars=['date'], 
        value_vars=[col for col in volatility_df.columns if col != 'date'],
        var_name='Stock', 
        value_name='Volatility'
    )
    
    # Create heatmap using Plotly Express
    fig = px.density_heatmap(
        volatility_long,
        x='Stock',
        y='date',
        z='Volatility',
        color_continuous_scale='Viridis',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=900,
        xaxis=dict(title='', tickangle=-45),
        yaxis=dict(title='', autorange='reversed'),
        coloraxis_colorbar=dict(title='Volatility'),
        margin=dict(l=60, r=60, t=80, b=80)
    )
    
    return fig

def create_returns_heatmap(data: Dict[str, pd.DataFrame], 
                          timeframe: int = 252,  # Default to approx. 1 year
                          freq: str = 'W',  # Weekly
                          title: str = 'Weekly Returns Heatmap') -> go.Figure:
    """
    Create a calendar heatmap showing returns over time.
    
    Args:
        data: Dictionary with a single ticker symbol as key and DataFrame as value
        timeframe: Timeframe in days to analyze
        freq: Frequency for returns ('D' for daily, 'W' for weekly, 'M' for monthly)
        title: Plot title
        
    Returns:
        Plotly figure object with returns calendar heatmap
    """
    if not data or len(data) != 1:
        logger.error("Data should contain exactly one stock")
        return go.Figure()
    
    # Get the ticker and DataFrame
    ticker = list(data.keys())[0]
    df = data[ticker]
    
    if df.empty or 'Close' not in df.columns:
        logger.error("Invalid data for returns heatmap")
        return go.Figure()
    
    # Limit to the specified timeframe
    df = df.tail(timeframe).copy()
    
    # Ensure the DataFrame has a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.error("DataFrame index must be datetime for returns heatmap")
        return go.Figure()
    
    # Calculate returns based on specified frequency
    if freq == 'D':
        returns = df['Close'].pct_change().dropna() * 100
    elif freq == 'W':
        returns = df['Close'].resample('W').last().pct_change().dropna() * 100
    elif freq == 'M':
        returns = df['Close'].resample('M').last().pct_change().dropna() * 100
    else:
        logger.error(f"Unsupported frequency: {freq}")
        return go.Figure()
    
    # Prepare data for calendar heatmap
    if freq == 'W':
        # Extract year and week number
        year_week = pd.DataFrame({
            'Year': returns.index.year,
            'Week': returns.index.isocalendar().week,
            'Return': returns.values
        })
        
        # Create pivot table
        pivot_table = year_week.pivot(index='Year', columns='Week', values='Return')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='Return %'),
            hovertemplate='Year: %{y}<br>Week: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Weekly Returns Heatmap',
            height=500,
            width=1100,
            xaxis=dict(title='Week of Year', dtick=1),
            yaxis=dict(title='Year', dtick=1),
            margin=dict(l=60, r=60, t=80, b=60)
        )
    
    elif freq == 'M':
        # Extract year and month
        year_month = pd.DataFrame({
            'Year': returns.index.year,
            'Month': returns.index.month,
            'Return': returns.values
        })
        
        # Create pivot table
        pivot_table = year_month.pivot(index='Year', columns='Month', values='Return')
        
        # Map month numbers to names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[month_names[i-1] for i in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='Return %'),
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Monthly Returns Heatmap',
            height=500,
            width=900,
            xaxis=dict(title='Month'),
            yaxis=dict(title='Year', dtick=1),
            margin=dict(l=60, r=60, t=80, b=60)
        )
    
    else:  # Daily returns
        # Extract year, month, and day
        daily_returns = pd.DataFrame({
            'Year': returns.index.year,
            'Month': returns.index.month,
            'Day': returns.index.day,
            'Return': returns.values
        })
        
        # Create calendar heatmap using px.density_heatmap
        fig = px.density_heatmap(
            daily_returns,
            x='Day',
            y='Month',
            z='Return',
            facet_col='Year',
            facet_col_wrap=1,
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title=f'{ticker} Daily Returns Heatmap'
        )
        
        # Update layout
        fig.update_layout(
            height=300 * len(daily_returns['Year'].unique()),
            width=1000,
            margin=dict(l=60, r=60, t=80, b=60)
        )
    
    return fig