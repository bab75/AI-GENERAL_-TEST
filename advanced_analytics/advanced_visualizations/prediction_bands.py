"""
Prediction Bands Visualization Module

This module provides visualization capabilities for showing price prediction bands,
confidence intervals, and probabilistic forecasts for stock prices.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Union, Optional
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_bollinger_bands_chart(data: pd.DataFrame, 
                               window: int = 20,
                               num_std: float = 2.0,
                               column: str = 'Close',
                               title: str = 'Bollinger Bands',
                               show_volume: bool = True) -> go.Figure:
    """
    Create a chart with Bollinger Bands.
    
    Args:
        data: DataFrame with price data
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        column: Column to use for calculations
        title: Chart title
        show_volume: Whether to include volume
        
    Returns:
        Plotly figure object with Bollinger Bands
    """
    if column not in data.columns:
        logger.error(f"Column '{column}' not found in data")
        return go.Figure()
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2 if show_volume else 1, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03 if show_volume else 0,
        row_heights=[0.8, 0.2] if show_volume else [1],
        subplot_titles=None
    )
    
    # Calculate Bollinger Bands
    data = data.copy()
    data['ma'] = data[column].rolling(window=window).mean()
    data['std'] = data[column].rolling(window=window).std()
    data['upper_band'] = data['ma'] + (data['std'] * num_std)
    data['lower_band'] = data['ma'] - (data['std'] * num_std)
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=column,
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add moving average
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['ma'],
            mode='lines',
            name=f'{window}-day MA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # Add upper band
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['upper_band'],
            mode='lines',
            name=f'Upper Band ({num_std} σ)',
            line=dict(color='green', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add lower band
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['lower_band'],
            mode='lines',
            name=f'Lower Band ({num_std} σ)',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add filled area between bands
    fig.add_trace(
        go.Scatter(
            x=data.index.tolist() + data.index.tolist()[::-1],
            y=data['upper_band'].tolist() + data['lower_band'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 176, 246, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Band Range',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add volume trace if requested
    if show_volume and 'Volume' in data.columns:
        # Add colored volume bars
        colors = []
        for i in range(len(data)):
            if i > 0:
                if data[column].iloc[i] > data[column].iloc[i-1]:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('green')  # Default for first bar
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.5
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800 if show_volume else 600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    # Update y-axis for price
    fig.update_yaxes(title_text='Price', row=1, col=1)
    
    # Update y-axis for volume
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

def create_confidence_interval_chart(data: pd.DataFrame,
                                   forecast_data: pd.DataFrame,
                                   column: str = 'Close',
                                   title: str = 'Price Forecast with Confidence Intervals') -> go.Figure:
    """
    Create a chart with price forecast and confidence intervals.
    
    Args:
        data: DataFrame with historical price data
        forecast_data: DataFrame with forecast data including confidence intervals
        column: Column to use for historical data
        title: Chart title
        
    Returns:
        Plotly figure object with forecast and confidence intervals
    """
    if column not in data.columns:
        logger.error(f"Column '{column}' not found in historical data")
        return go.Figure()
    
    if 'forecast' not in forecast_data.columns:
        logger.error("'forecast' column not found in forecast data")
        return go.Figure()
    
    # Check if confidence interval columns exist
    has_ci = all(col in forecast_data.columns for col in ['lower_90', 'upper_90', 'lower_80', 'upper_80'])
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        )
    )
    
    # Add confidence intervals if available
    if has_ci:
        # Add 90% confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                y=forecast_data['upper_90'].tolist() + forecast_data['lower_90'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='90% Confidence Interval',
                showlegend=True
            )
        )
        
        # Add 80% confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                y=forecast_data['upper_80'].tolist() + forecast_data['lower_80'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='80% Confidence Interval',
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    return fig

def create_monte_carlo_simulation_chart(data: pd.DataFrame,
                                      num_simulations: int = 100,
                                      days_to_forecast: int = 30,
                                      percentiles: List[int] = [5, 25, 50, 75, 95],
                                      column: str = 'Close',
                                      title: str = 'Monte Carlo Price Simulation') -> go.Figure:
    """
    Create a Monte Carlo simulation chart for price forecasting.
    
    Args:
        data: DataFrame with historical price data
        num_simulations: Number of Monte Carlo simulations to run
        days_to_forecast: Number of days to forecast
        percentiles: List of percentiles to show
        column: Column to use for historical data
        title: Chart title
        
    Returns:
        Plotly figure object with Monte Carlo simulations
    """
    if column not in data.columns:
        logger.error(f"Column '{column}' not found in data")
        return go.Figure()
    
    # Calculate daily returns
    returns = data[column].pct_change().dropna()
    
    # Calculate statistics
    mu = returns.mean()
    sigma = returns.std()
    
    # Get last price
    last_price = data[column].iloc[-1]
    
    # Generate simulations
    simulation_df = pd.DataFrame()
    
    for i in range(num_simulations):
        # Generate random returns
        daily_returns = np.random.normal(mu, sigma, days_to_forecast)
        
        # Create price series
        price_series = [last_price]
        
        for j in range(days_to_forecast):
            # Calculate new price
            price_series.append(price_series[-1] * (1 + daily_returns[j]))
        
        # Store simulation
        simulation_df[f'sim_{i}'] = price_series
    
    # Create date index for forecast
    last_date = data.index[-1]
    if pd.api.types.is_datetime64_any_dtype(last_date):
        date_range = pd.date_range(start=last_date, periods=days_to_forecast + 1)
    else:
        # If not datetime, use integers
        date_range = range(int(last_date), int(last_date) + days_to_forecast + 1)
    
    # Set index
    simulation_df.index = date_range
    
    # Calculate percentiles
    percentile_df = pd.DataFrame(index=simulation_df.index)
    
    for p in percentiles:
        percentile_df[f'p_{p}'] = simulation_df.apply(lambda x: np.percentile(x, p), axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add a sample of simulation paths
    for i in range(min(10, num_simulations)):
        fig.add_trace(
            go.Scatter(
                x=simulation_df.index,
                y=simulation_df[f'sim_{i}'],
                mode='lines',
                name=f'Simulation {i+1}',
                opacity=0.3,
                line=dict(color='lightgrey'),
                showlegend=(i == 0)  # Only show legend for the first one
            )
        )
    
    # Add percentiles
    colors = ['red', 'orange', 'green', 'orange', 'red']
    
    for i, p in enumerate(percentiles):
        fig.add_trace(
            go.Scatter(
                x=percentile_df.index,
                y=percentile_df[f'p_{p}'],
                mode='lines',
                name=f'{p}th Percentile',
                line=dict(color=colors[i], width=2, dash='dash')
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    return fig

def create_technical_bands_chart(data: pd.DataFrame,
                               bands_type: str = 'keltner',  # 'keltner', 'donchian', or 'psar'
                               column: str = 'Close',
                               title: str = None,
                               show_volume: bool = True) -> go.Figure:
    """
    Create a chart with technical bands (Keltner Channels, Donchian Channels, or Parabolic SAR).
    
    Args:
        data: DataFrame with price data (must include OHLC data)
        bands_type: Type of bands to show ('keltner', 'donchian', or 'psar')
        column: Column to use for price trace
        title: Chart title (if None, will be set based on bands_type)
        show_volume: Whether to include volume
        
    Returns:
        Plotly figure object with technical bands
    """
    # Check required columns based on bands type
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Required columns {required_cols} not found in data")
        return go.Figure()
    
    # Create title if not provided
    if title is None:
        if bands_type.lower() == 'keltner':
            title = 'Keltner Channels'
        elif bands_type.lower() == 'donchian':
            title = 'Donchian Channels'
        elif bands_type.lower() == 'psar':
            title = 'Parabolic SAR'
        else:
            title = 'Technical Bands'
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2 if show_volume else 1, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03 if show_volume else 0,
        row_heights=[0.8, 0.2] if show_volume else [1],
        subplot_titles=None
    )
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=column,
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Calculate and add bands based on type
    if bands_type.lower() == 'keltner':
        # Calculate Keltner Channels
        data = data.copy()
        
        # Calculate typical price
        data['tp'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate moving average of typical price
        data['ma'] = data['tp'].rolling(window=20).mean()
        
        # Calculate Average True Range
        data['tr'] = np.maximum(
            np.maximum(
                data['High'] - data['Low'],
                abs(data['High'] - data['Close'].shift(1))
            ),
            abs(data['Low'] - data['Close'].shift(1))
        )
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # Calculate Keltner Channels
        data['upper_band'] = data['ma'] + (data['atr'] * 2)
        data['lower_band'] = data['ma'] - (data['atr'] * 2)
        
        # Add moving average
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['ma'],
                mode='lines',
                name='Middle Line (EMA)',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        # Add upper band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['upper_band'],
                mode='lines',
                name='Upper Channel',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add lower band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['lower_band'],
                mode='lines',
                name='Lower Channel',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add filled area between bands
        fig.add_trace(
            go.Scatter(
                x=data.index.tolist() + data.index.tolist()[::-1],
                y=data['upper_band'].tolist() + data['lower_band'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 176, 246, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Channel Range',
                showlegend=False
            ),
            row=1, col=1
        )
    
    elif bands_type.lower() == 'donchian':
        # Calculate Donchian Channels
        data = data.copy()
        window = 20
        
        # Calculate upper and lower bands
        data['upper_band'] = data['High'].rolling(window=window).max()
        data['lower_band'] = data['Low'].rolling(window=window).min()
        data['middle_band'] = (data['upper_band'] + data['lower_band']) / 2
        
        # Add middle band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['middle_band'],
                mode='lines',
                name='Middle Line',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        # Add upper band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['upper_band'],
                mode='lines',
                name='Upper Channel',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add lower band
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['lower_band'],
                mode='lines',
                name='Lower Channel',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Add filled area between bands
        fig.add_trace(
            go.Scatter(
                x=data.index.tolist() + data.index.tolist()[::-1],
                y=data['upper_band'].tolist() + data['lower_band'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 176, 246, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Channel Range',
                showlegend=False
            ),
            row=1, col=1
        )
    
    elif bands_type.lower() == 'psar':
        # Calculate Parabolic SAR
        data = data.copy()
        
        # Parameters
        af_start = 0.02
        af_step = 0.02
        af_max = 0.2
        
        # Initialize arrays
        length = len(data)
        psar = np.zeros(length)
        ep = np.zeros(length)
        af = np.zeros(length)
        trend = np.zeros(length)
        
        # Find initial trend
        if data['Close'].iloc[1] > data['Close'].iloc[0]:
            trend[1] = 1  # Uptrend
            psar[1] = data['Low'].iloc[0]
            ep[1] = data['High'].iloc[1]
        else:
            trend[1] = -1  # Downtrend
            psar[1] = data['High'].iloc[0]
            ep[1] = data['Low'].iloc[1]
        
        af[1] = af_start
        
        # Calculate PSAR values
        for i in range(2, length):
            # Copy previous values
            psar[i] = psar[i-1]
            ep[i] = ep[i-1]
            af[i] = af[i-1]
            trend[i] = trend[i-1]
            
            # Update PSAR
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Previous uptrend
                # Ensure PSAR is below lows
                psar[i] = min(psar[i], data['Low'].iloc[i-2], data['Low'].iloc[i-1])
                
                # Check for reversal
                if psar[i] > data['Low'].iloc[i]:
                    trend[i] = -1  # Reverse to downtrend
                    psar[i] = ep[i-1]  # Set PSAR to previous extreme point
                    ep[i] = data['Low'].iloc[i]  # Set extreme point to current low
                    af[i] = af_start  # Reset acceleration factor
                else:
                    trend[i] = 1  # Continue uptrend
                    # Update extreme point and acceleration factor
                    if data['High'].iloc[i] > ep[i-1]:
                        ep[i] = data['High'].iloc[i]
                        af[i] = min(af[i-1] + af_step, af_max)
            else:  # Previous downtrend
                # Ensure PSAR is above highs
                psar[i] = max(psar[i], data['High'].iloc[i-2], data['High'].iloc[i-1])
                
                # Check for reversal
                if psar[i] < data['High'].iloc[i]:
                    trend[i] = 1  # Reverse to uptrend
                    psar[i] = ep[i-1]  # Set PSAR to previous extreme point
                    ep[i] = data['High'].iloc[i]  # Set extreme point to current high
                    af[i] = af_start  # Reset acceleration factor
                else:
                    trend[i] = -1  # Continue downtrend
                    # Update extreme point and acceleration factor
                    if data['Low'].iloc[i] < ep[i-1]:
                        ep[i] = data['Low'].iloc[i]
                        af[i] = min(af[i-1] + af_step, af_max)
        
        # Store PSAR values in DataFrame
        data['psar'] = psar
        data['trend'] = trend
        
        # Create lists for uptrend and downtrend dots
        uptrend_x = []
        uptrend_y = []
        downtrend_x = []
        downtrend_y = []
        
        for i in range(1, length):
            if trend[i] == 1:  # Uptrend
                uptrend_x.append(data.index[i])
                uptrend_y.append(psar[i])
            else:  # Downtrend
                downtrend_x.append(data.index[i])
                downtrend_y.append(psar[i])
        
        # Add uptrend PSAR dots
        fig.add_trace(
            go.Scatter(
                x=uptrend_x,
                y=uptrend_y,
                mode='markers',
                name='PSAR (Uptrend)',
                marker=dict(
                    symbol='circle',
                    size=5,
                    color='green'
                )
            ),
            row=1, col=1
        )
        
        # Add downtrend PSAR dots
        fig.add_trace(
            go.Scatter(
                x=downtrend_x,
                y=downtrend_y,
                mode='markers',
                name='PSAR (Downtrend)',
                marker=dict(
                    symbol='circle',
                    size=5,
                    color='red'
                )
            ),
            row=1, col=1
        )
    
    # Add volume trace if requested
    if show_volume and 'Volume' in data.columns:
        # Add colored volume bars (green for up days, red for down days)
        colors = []
        for i in range(len(data)):
            if i > 0 and data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                colors.append('green')
            else:
                colors.append('red')
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.5
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800 if show_volume else 600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    # Update y-axis for price
    fig.update_yaxes(title_text='Price', row=1, col=1)
    
    # Update y-axis for volume
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

def create_support_resistance_chart(data: pd.DataFrame,
                                  levels: List[float] = None,
                                  auto_detect: bool = True,
                                  n_levels: int = 5,
                                  title: str = 'Support and Resistance Levels',
                                  show_volume: bool = True) -> go.Figure:
    """
    Create a chart with support and resistance levels.
    
    Args:
        data: DataFrame with price data
        levels: List of manually specified price levels
        auto_detect: Whether to auto-detect levels
        n_levels: Number of levels to auto-detect
        title: Chart title
        show_volume: Whether to include volume
        
    Returns:
        Plotly figure object with support and resistance levels
    """
    # Check required columns
    if 'Close' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
        logger.error("Required columns (Close, High, Low) not found in data")
        return go.Figure()
    
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2 if show_volume else 1, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03 if show_volume else 0,
        row_heights=[0.8, 0.2] if show_volume else [1],
        subplot_titles=None
    )
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Auto-detect levels
    if auto_detect:
        # Use KDE (Kernel Density Estimation) to identify price clusters
        kde = stats.gaussian_kde(
            np.concatenate([data['High'].values, data['Low'].values])
        )
        
        # Create a range of price values
        price_range = np.linspace(
            data['Low'].min() * 0.95,
            data['High'].max() * 1.05,
            1000
        )
        
        # Calculate density
        density = kde(price_range)
        
        # Find peaks (local maxima) in the density
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density)
        
        # Combine with manually specified levels
        if levels:
            all_levels = np.concatenate([price_range[peaks], np.array(levels)])
        else:
            all_levels = price_range[peaks]
        
        # Sort levels and select top n_levels
        level_densities = kde(all_levels)
        level_idx = np.argsort(level_densities)[::-1]  # Sort by density (descending)
        selected_levels = all_levels[level_idx[:n_levels]]
        
        # Add to levels list
        levels = selected_levels.tolist()
    
    # Add levels
    if levels:
        # Calculate closest price for each level
        current_price = data['Close'].iloc[-1]
        level_distances = [abs(level - current_price) / current_price * 100 for level in levels]
        
        # Sort levels by distance to current price
        level_info = sorted(zip(levels, level_distances), key=lambda x: x[1])
        
        # Add horizontal lines for levels
        for i, (level, distance) in enumerate(level_info):
            # Color coding
            if level > current_price:
                # Resistance level
                color = 'rgba(255, 0, 0, 0.5)'
                name = f"Resistance: ${level:.2f} ({distance:.1f}% away)"
            else:
                # Support level
                color = 'rgba(0, 255, 0, 0.5)'
                name = f"Support: ${level:.2f} ({distance:.1f}% away)"
            
            # Add line
            fig.add_shape(
                type='line',
                x0=data.index[0],
                x1=data.index[-1],
                y0=level,
                y1=level,
                line=dict(
                    color=color,
                    width=2,
                    dash='dash'
                )
            )
            
            # Add annotation
            fig.add_annotation(
                x=data.index[-1],
                y=level,
                text=name,
                showarrow=False,
                xshift=10,
                align='left',
                font=dict(
                    color='black',
                    size=10
                ),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=color,
                borderwidth=1
            )
    
    # Add volume trace if requested
    if show_volume and 'Volume' in data.columns:
        # Add colored volume bars (green for up days, red for down days)
        colors = []
        for i in range(len(data)):
            if data['Close'].iloc[i] > data['Open'].iloc[i]:
                colors.append('green')
            else:
                colors.append('red')
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.5
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800 if show_volume else 600,
        width=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    # Update y-axis for price
    fig.update_yaxes(title_text='Price', row=1, col=1)
    
    # Update y-axis for volume
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig