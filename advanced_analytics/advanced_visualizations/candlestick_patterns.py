"""
Advanced Candlestick Pattern Visualization Module

This module provides enhanced visualization capabilities for candlestick chart patterns,
highlighting recognized patterns and providing contextual information.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Union, Optional
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from advanced_analytics import pattern_recognition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_candlestick_chart(data: pd.DataFrame, 
                           title: str = 'Candlestick Chart',
                           show_volume: bool = True,
                           rangeslider: bool = True) -> go.Figure:
    """
    Create a basic candlestick chart with optional volume bars.
    
    Args:
        data: DataFrame with OHLCV data (must have Open, High, Low, Close columns)
        title: Chart title
        show_volume: Whether to include volume bars
        rangeslider: Whether to include a range slider
        
    Returns:
        Plotly figure object with candlestick chart
    """
    # Check if required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Required columns {required_cols} not found in data")
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
    
    # Add volume trace if requested
    if show_volume and 'Volume' in data.columns:
        # Add colored volume bars (green for up days, red for down days)
        colors = ['green' if data['Close'][i] > data['Open'][i] else 'red' 
                  for i in range(len(data))]
        
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
        xaxis_rangeslider_visible=rangeslider,
        height=800 if show_volume else 600,
        width=1000,
        showlegend=False,
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume' if show_volume else None
    )
    
    # Update y-axis for price
    fig.update_yaxes(title_text='Price', row=1, col=1)
    
    # Update y-axis for volume
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig

def create_pattern_candlestick_chart(data: pd.DataFrame, 
                                   pattern_result: Dict,
                                   title: str = 'Candlestick Pattern Analysis',
                                   show_volume: bool = True) -> go.Figure:
    """
    Create a candlestick chart with highlighted patterns.
    
    Args:
        data: DataFrame with OHLCV data
        pattern_result: Dictionary with pattern detection results
        title: Chart title
        show_volume: Whether to include volume bars
        
    Returns:
        Plotly figure object with candlestick chart and highlighted patterns
    """
    # Create base candlestick chart
    fig = create_candlestick_chart(data, title=title, show_volume=show_volume, rangeslider=False)
    
    # If no pattern or invalid pattern result, return base chart
    if not pattern_result or pattern_result.get('pattern', 'none') == 'none':
        fig.update_layout(title=f"{title} (No patterns detected)")
        return fig
    
    # Extract pattern details
    pattern_name = pattern_result.get('pattern', 'none')
    confidence = pattern_result.get('confidence', 0)
    
    # Add pattern name and confidence to title
    pattern_title = f"{title} - {pattern_name.replace('_', ' ').title()} (Confidence: {confidence:.2f})"
    fig.update_layout(title=pattern_title)
    
    # Get pattern-specific shapes to highlight on the chart
    shapes = []
    annotations = []
    
    # Helper function to add a horizontal line
    def add_horizontal_line(y_value, x_range, color='rgba(255, 0, 0, 0.5)', width=1.5, dash='solid'):
        return dict(
            type='line',
            x0=x_range[0],
            x1=x_range[-1],
            y0=y_value,
            y1=y_value,
            line=dict(color=color, width=width, dash=dash)
        )
    
    # Helper function to add a vertical line
    def add_vertical_line(x_value, y_range, color='rgba(0, 0, 255, 0.5)', width=1.5, dash='solid'):
        return dict(
            type='line',
            x0=x_value,
            x1=x_value,
            y0=y_range[0],
            y1=y_range[1],
            line=dict(color=color, width=width, dash=dash)
        )
    
    # Helper function to add a rectangle
    def add_rectangle(x_range, y_range, color='rgba(0, 255, 0, 0.2)', line_color='rgba(0, 255, 0, 0.8)'):
        return dict(
            type='rect',
            x0=x_range[0],
            x1=x_range[-1],
            y0=y_range[0],
            y1=y_range[1],
            fillcolor=color,
            line=dict(color=line_color, width=1)
        )
    
    # Handle head and shoulders pattern
    if 'head_and_shoulders' in pattern_name:
        # Extract key points
        is_inverse = 'inverse' in pattern_name
        
        # Indices of shoulders and head
        left_idx = pattern_result.get('left_shoulder_idx', 0)
        head_idx = pattern_result.get('head_idx', 0)
        right_idx = pattern_result.get('right_shoulder_idx', 0)
        
        # Get dates and prices at these points
        indexes = data.index.tolist()
        if left_idx < len(indexes) and head_idx < len(indexes) and right_idx < len(indexes):
            left_date = indexes[left_idx]
            head_date = indexes[head_idx]
            right_date = indexes[right_idx]
            
            # Prices for regular or inverse pattern
            if is_inverse:
                left_price = data['Low'].iloc[left_idx]
                head_price = data['Low'].iloc[head_idx]
                right_price = data['Low'].iloc[right_idx]
                neckline = pattern_result.get('neckline', max(data['High'].iloc[left_idx:right_idx+1]))
            else:
                left_price = data['High'].iloc[left_idx]
                head_price = data['High'].iloc[head_idx]
                right_price = data['High'].iloc[right_idx]
                neckline = pattern_result.get('neckline', min(data['Low'].iloc[left_idx:right_idx+1]))
            
            # Add markers for shoulders and head
            fig.add_trace(go.Scatter(
                x=[left_date, head_date, right_date],
                y=[left_price, head_price, right_price],
                mode='markers',
                marker=dict(size=12, color='purple', symbol='circle'),
                name='Key Points',
                hoverinfo='text',
                hovertext=[
                    f"Left Shoulder: {left_price:.2f}",
                    f"Head: {head_price:.2f}",
                    f"Right Shoulder: {right_price:.2f}"
                ]
            ), row=1, col=1)
            
            # Add line connecting shoulders
            fig.add_trace(go.Scatter(
                x=[left_date, right_date],
                y=[left_price, right_price],
                mode='lines',
                line=dict(color='purple', width=2, dash='dash'),
                name='Shoulder Line',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Add head line connecting to shoulders line
            mid_date = indexes[left_idx + (right_idx - left_idx) // 2]
            mid_price = (left_price + right_price) / 2
            fig.add_trace(go.Scatter(
                x=[mid_date, head_date],
                y=[mid_price, head_price],
                mode='lines',
                line=dict(color='purple', width=2, dash='dash'),
                name='Head Line',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Add neckline
            fig.add_shape(
                add_horizontal_line(neckline, 
                                   [indexes[left_idx], indexes[right_idx]], 
                                   color='rgba(255, 165, 0, 0.8)', 
                                   width=2, 
                                   dash='dot')
            )
            
            # Add neckline annotation
            fig.add_annotation(
                x=indexes[right_idx],
                y=neckline,
                text="Neckline",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=0
            )
            
            # Add pattern annotation
            pattern_text = "Inverse Head & Shoulders (Bullish)" if is_inverse else "Head & Shoulders (Bearish)"
            fig.add_annotation(
                x=head_date,
                y=head_price * (0.95 if is_inverse else 1.05),
                text=pattern_text,
                showarrow=False,
                font=dict(size=14, color='purple'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='purple',
                borderwidth=1
            )
    
    # Handle double top/bottom pattern
    elif 'double_top' in pattern_name or 'double_bottom' in pattern_name:
        is_bottom = 'bottom' in pattern_name
        
        # Extract key points
        first_idx = pattern_result.get('first_peak_idx' if not is_bottom else 'first_trough_idx', 0)
        second_idx = pattern_result.get('second_peak_idx' if not is_bottom else 'second_trough_idx', 0)
        middle_idx = pattern_result.get('trough_idx' if not is_bottom else 'peak_idx', 0)
        
        # Get dates and prices at these points
        indexes = data.index.tolist()
        if first_idx < len(indexes) and second_idx < len(indexes) and middle_idx < len(indexes):
            first_date = indexes[first_idx]
            second_date = indexes[second_idx]
            middle_date = indexes[middle_idx]
            
            # Prices for tops or bottoms
            if is_bottom:
                first_price = data['Low'].iloc[first_idx]
                second_price = data['Low'].iloc[second_idx]
                middle_price = data['High'].iloc[middle_idx]
                level = pattern_result.get('support_level', (first_price + second_price) / 2)
            else:
                first_price = data['High'].iloc[first_idx]
                second_price = data['High'].iloc[second_idx]
                middle_price = data['Low'].iloc[middle_idx]
                level = pattern_result.get('resistance_level', (first_price + second_price) / 2)
            
            # Add markers for peaks/troughs
            fig.add_trace(go.Scatter(
                x=[first_date, middle_date, second_date],
                y=[first_price, middle_price, second_price],
                mode='markers',
                marker=dict(size=12, color='blue', symbol='circle'),
                name='Key Points',
                hoverinfo='text',
                hovertext=[
                    f"First {'Bottom' if is_bottom else 'Top'}: {first_price:.2f}",
                    f"Middle {'Peak' if is_bottom else 'Trough'}: {middle_price:.2f}",
                    f"Second {'Bottom' if is_bottom else 'Top'}: {second_price:.2f}"
                ]
            ), row=1, col=1)
            
            # Add line connecting the tops/bottoms
            fig.add_trace(go.Scatter(
                x=[first_date, second_date],
                y=[first_price, second_price],
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                name=f"{'Support' if is_bottom else 'Resistance'} Line",
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Add support/resistance line
            fig.add_shape(
                add_horizontal_line(level, 
                                   [first_date, second_date], 
                                   color='rgba(0, 0, 255, 0.8)' if is_bottom else 'rgba(255, 0, 0, 0.8)', 
                                   width=2, 
                                   dash='dot')
            )
            
            # Add level annotation
            fig.add_annotation(
                x=second_date,
                y=level,
                text="Support Level" if is_bottom else "Resistance Level",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=0
            )
            
            # Add pattern annotation
            pattern_text = "Double Bottom (Bullish)" if is_bottom else "Double Top (Bearish)"
            fig.add_annotation(
                x=middle_date,
                y=middle_price * (1.05 if is_bottom else 0.95),
                text=pattern_text,
                showarrow=False,
                font=dict(size=14, color='blue'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='blue',
                borderwidth=1
            )
    
    # Handle triangle patterns
    elif 'triangle' in pattern_name:
        # Extract key points
        is_ascending = 'ascending' in pattern_name
        is_descending = 'descending' in pattern_name
        is_symmetrical = 'symmetrical' in pattern_name
        
        # Get the peaks and troughs to draw trend lines
        peaks = pattern_result.get('recent_peaks', [])
        troughs = pattern_result.get('recent_troughs', [])
        
        # Support and resistance trend lines
        resistance_slope = pattern_result.get('resistance_slope', 0)
        support_slope = pattern_result.get('support_slope', 0)
        
        # Only proceed if we have valid peaks and troughs
        if peaks and troughs and len(peaks) >= 2 and len(troughs) >= 2:
            # Get dates at these points
            indexes = data.index.tolist()
            
            # Filter valid indices
            peaks = [p for p in peaks if p < len(indexes)]
            troughs = [t for t in troughs if t < len(indexes)]
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Dates
                peak_dates = [indexes[p] for p in peaks]
                trough_dates = [indexes[t] for t in troughs]
                
                # Prices
                peak_prices = [data['High'].iloc[p] for p in peaks]
                trough_prices = [data['Low'].iloc[t] for t in troughs]
                
                # Add markers for peaks and troughs
                fig.add_trace(go.Scatter(
                    x=peak_dates,
                    y=peak_prices,
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='triangle-down'),
                    name='Peaks',
                    hoverinfo='text',
                    hovertext=[f"Peak: {p:.2f}" for p in peak_prices]
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=trough_dates,
                    y=trough_prices,
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='triangle-up'),
                    name='Troughs',
                    hoverinfo='text',
                    hovertext=[f"Trough: {t:.2f}" for t in trough_prices]
                ), row=1, col=1)
                
                # Add resistance trend line
                if peaks:
                    # Create dates for trend line
                    first_date_idx = min(peaks)
                    last_date_idx = max(peaks)
                    x_values = np.array([first_date_idx, last_date_idx])
                    
                    # Calculate y values based on slope
                    first_y = data['High'].iloc[first_date_idx]
                    y_values = first_y + resistance_slope * (x_values - first_date_idx)
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=[indexes[x] for x in x_values],
                        y=y_values,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Resistance Trend',
                        hoverinfo='skip'
                    ), row=1, col=1)
                
                # Add support trend line
                if troughs:
                    # Create dates for trend line
                    first_date_idx = min(troughs)
                    last_date_idx = max(troughs)
                    x_values = np.array([first_date_idx, last_date_idx])
                    
                    # Calculate y values based on slope
                    first_y = data['Low'].iloc[first_date_idx]
                    y_values = first_y + support_slope * (x_values - first_date_idx)
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=[indexes[x] for x in x_values],
                        y=y_values,
                        mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
                        name='Support Trend',
                        hoverinfo='skip'
                    ), row=1, col=1)
                
                # Add pattern annotation
                if is_ascending:
                    pattern_text = "Ascending Triangle (Bullish)"
                    annotation_color = 'green'
                elif is_descending:
                    pattern_text = "Descending Triangle (Bearish)"
                    annotation_color = 'red'
                else:  # Symmetrical
                    pattern_text = "Symmetrical Triangle (Neutral)"
                    annotation_color = 'blue'
                
                # Place annotation near the middle of the pattern
                middle_idx = (min(peaks + troughs) + max(peaks + troughs)) // 2
                middle_date = indexes[middle_idx] if middle_idx < len(indexes) else indexes[-1]
                middle_price = data['Close'].iloc[middle_idx] if middle_idx < len(data) else data['Close'].iloc[-1]
                
                fig.add_annotation(
                    x=middle_date,
                    y=middle_price,
                    text=pattern_text,
                    showarrow=False,
                    font=dict(size=14, color=annotation_color),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor=annotation_color,
                    borderwidth=1
                )
    
    # Handle cup and handle pattern
    elif 'cup_and_handle' in pattern_name:
        # Extract key points
        left_peak_idx = pattern_result.get('left_peak_idx', 0)
        right_peak_idx = pattern_result.get('right_peak_idx', 0)
        cup_low_idx = pattern_result.get('cup_low_idx', 0)
        resistance_level = pattern_result.get('resistance_level', 0)
        
        # Get dates and prices at these points
        indexes = data.index.tolist()
        if left_peak_idx < len(indexes) and right_peak_idx < len(indexes) and cup_low_idx < len(indexes):
            left_date = indexes[left_peak_idx]
            right_date = indexes[right_peak_idx]
            low_date = indexes[cup_low_idx]
            
            left_price = data['High'].iloc[left_peak_idx]
            right_price = data['High'].iloc[right_peak_idx]
            low_price = data['Low'].iloc[cup_low_idx]
            
            # Create dates for the handle (if available in the data)
            handle_dates = []
            handle_prices = []
            
            if right_peak_idx + 1 < len(indexes):
                # Look for a small dip after the right peak (the handle)
                for i in range(right_peak_idx + 1, min(right_peak_idx + 20, len(indexes))):
                    handle_dates.append(indexes[i])
                    handle_prices.append(data['Close'].iloc[i])
            
            # Add markers for cup points
            fig.add_trace(go.Scatter(
                x=[left_date, low_date, right_date],
                y=[left_price, low_price, right_price],
                mode='markers',
                marker=dict(size=12, color='orange', symbol='circle'),
                name='Cup Points',
                hoverinfo='text',
                hovertext=[
                    f"Left Cup Rim: {left_price:.2f}",
                    f"Cup Bottom: {low_price:.2f}",
                    f"Right Cup Rim: {right_price:.2f}"
                ]
            ), row=1, col=1)
            
            # Add U-shaped line for the cup
            # Create smooth curve for cup
            cup_dates = pd.date_range(start=left_date, end=right_date, periods=50)
            cup_x_norm = np.linspace(0, 1, 50)
            
            # Create U-shape using parabola
            cup_y = -4 * (cup_x_norm - 0.5)**2 + 1
            cup_prices = low_price + cup_y * (resistance_level - low_price)
            
            fig.add_trace(go.Scatter(
                x=cup_dates,
                y=cup_prices,
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='Cup',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Add resistance line
            fig.add_shape(
                add_horizontal_line(resistance_level, 
                                   [left_date, handle_dates[-1] if handle_dates else right_date], 
                                   color='rgba(255, 165, 0, 0.8)', 
                                   width=2, 
                                   dash='dot')
            )
            
            # Add handle if available
            if handle_dates and handle_prices:
                fig.add_trace(go.Scatter(
                    x=handle_dates,
                    y=handle_prices,
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name='Handle',
                    hoverinfo='text',
                    hovertext="Handle"
                ), row=1, col=1)
            
            # Add pattern annotation
            fig.add_annotation(
                x=low_date,
                y=low_price * 0.95,
                text="Cup and Handle (Bullish)",
                showarrow=False,
                font=dict(size=14, color='orange'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='orange',
                borderwidth=1
            )
    
    # Handle flag and pennant patterns
    elif 'flag' in pattern_name or 'pennant' in pattern_name:
        # Extract pattern details
        is_bullish = 'bullish' in pattern_name
        is_pennant = 'pennant' in pattern_name
        
        pole_start_idx = pattern_result.get('pole_start_idx', 0)
        pole_end_idx = pattern_result.get('pole_end_idx', 0)
        
        # Get dates at these points
        indexes = data.index.tolist()
        if pole_start_idx < len(indexes) and pole_end_idx < len(indexes):
            pole_start_date = indexes[pole_start_idx]
            pole_end_date = indexes[pole_end_idx]
            
            pole_start_price = data['Close'].iloc[pole_start_idx]
            pole_end_price = data['Close'].iloc[pole_end_idx]
            
            # Add pole
            fig.add_trace(go.Scatter(
                x=[pole_start_date, pole_end_date],
                y=[pole_start_price, pole_end_price],
                mode='lines',
                line=dict(color='blue', width=3),
                name='Flagpole',
                hoverinfo='text',
                hovertext=f"Flagpole: {abs(pole_end_price - pole_start_price):.2f} points"
            ), row=1, col=1)
            
            # Add flag or pennant
            if pole_end_idx + 1 < len(indexes):
                flag_start_date = pole_end_date
                flag_end_idx = min(pole_end_idx + 20, len(indexes) - 1)
                flag_end_date = indexes[flag_end_idx]
                
                if is_pennant:
                    # Draw converging lines for pennant
                    upper_slope = pattern_result.get('upper_slope', 0)
                    lower_slope = pattern_result.get('lower_slope', 0)
                    
                    # Calculate upper and lower trend lines
                    upper_start = data['High'].iloc[pole_end_idx]
                    lower_start = data['Low'].iloc[pole_end_idx]
                    
                    upper_end = upper_start + upper_slope * (flag_end_idx - pole_end_idx)
                    lower_end = lower_start + lower_slope * (flag_end_idx - pole_end_idx)
                    
                    # Draw upper trend line
                    fig.add_trace(go.Scatter(
                        x=[flag_start_date, flag_end_date],
                        y=[upper_start, upper_end],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dash'),
                        name='Upper Trend',
                        hoverinfo='skip'
                    ), row=1, col=1)
                    
                    # Draw lower trend line
                    fig.add_trace(go.Scatter(
                        x=[flag_start_date, flag_end_date],
                        y=[lower_start, lower_end],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dash'),
                        name='Lower Trend',
                        hoverinfo='skip'
                    ), row=1, col=1)
                else:
                    # Draw parallel lines for flag
                    flag_slope = pattern_result.get('flag_slope', 0)
                    
                    # Calculate channel lines
                    mid_price = data['Close'].iloc[pole_end_idx]
                    channel_width = abs(data['High'].iloc[pole_end_idx] - data['Low'].iloc[pole_end_idx])
                    
                    upper_start = mid_price + channel_width / 2
                    lower_start = mid_price - channel_width / 2
                    
                    upper_end = upper_start + flag_slope * (flag_end_idx - pole_end_idx)
                    lower_end = lower_start + flag_slope * (flag_end_idx - pole_end_idx)
                    
                    # Draw upper channel line
                    fig.add_trace(go.Scatter(
                        x=[flag_start_date, flag_end_date],
                        y=[upper_start, upper_end],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
                        name='Upper Channel',
                        hoverinfo='skip'
                    ), row=1, col=1)
                    
                    # Draw lower channel line
                    fig.add_trace(go.Scatter(
                        x=[flag_start_date, flag_end_date],
                        y=[lower_start, lower_end],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
                        name='Lower Channel',
                        hoverinfo='skip'
                    ), row=1, col=1)
            
            # Add pattern annotation
            pattern_text = (
                f"{'Bullish' if is_bullish else 'Bearish'} "
                f"{'Pennant' if is_pennant else 'Flag'}"
            )
            annotation_color = 'green' if is_bullish else 'red'
            
            # Place annotation near the middle of the pattern
            middle_date = indexes[(pole_start_idx + pole_end_idx) // 2]
            middle_price = (pole_start_price + pole_end_price) / 2
            
            fig.add_annotation(
                x=middle_date,
                y=middle_price,
                text=pattern_text,
                showarrow=False,
                font=dict(size=14, color=annotation_color),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=annotation_color,
                borderwidth=1
            )
    
    # Handle wedge patterns
    elif 'wedge' in pattern_name:
        # Extract key points
        is_rising = 'rising' in pattern_name
        is_falling = 'falling' in pattern_name
        
        # Get slopes and angles
        upper_slope = pattern_result.get('upper_slope', 0)
        lower_slope = pattern_result.get('lower_slope', 0)
        upper_angle = pattern_result.get('upper_angle', 0)
        lower_angle = pattern_result.get('lower_angle', 0)
        
        # Determine wedge direction
        direction = "Rising" if is_rising else "Falling"
        
        # Get start and end indices for wedge
        # Use a reasonable portion of the data for the wedge
        wedge_length = min(30, len(data) // 3)
        start_idx = max(0, len(data) - wedge_length)
        end_idx = len(data) - 1
        
        if start_idx < end_idx:
            # Get dates
            indexes = data.index.tolist()
            start_date = indexes[start_idx]
            end_date = indexes[end_idx]
            
            # Get starting prices for trend lines
            upper_start = data['High'].iloc[start_idx]
            lower_start = data['Low'].iloc[start_idx]
            
            # Calculate end prices based on slopes
            delta_x = end_idx - start_idx
            upper_end = upper_start + upper_slope * delta_x
            lower_end = lower_start + lower_slope * delta_x
            
            # Draw upper trend line
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[upper_start, upper_end],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='Upper Trend',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Draw lower trend line
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[lower_start, lower_end],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='Lower Trend',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Add pattern annotation
            pattern_text = f"{direction} Wedge ({'Bearish' if is_rising else 'Bullish'})"
            annotation_color = 'red' if is_rising else 'green'
            
            # Place annotation near the middle of the pattern
            middle_date = indexes[(start_idx + end_idx) // 2]
            middle_price = (data['High'].iloc[start_idx] + data['Low'].iloc[start_idx]) / 2
            
            fig.add_annotation(
                x=middle_date,
                y=middle_price,
                text=pattern_text,
                showarrow=False,
                font=dict(size=14, color=annotation_color),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=annotation_color,
                borderwidth=1
            )
    
    # Update layout
    fig.update_layout(
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

def create_multiple_pattern_chart(data: pd.DataFrame, 
                                patterns: List[Dict],
                                title: str = 'Multiple Pattern Detection',
                                show_volume: bool = True) -> go.Figure:
    """
    Create a candlestick chart highlighting multiple detected patterns.
    
    Args:
        data: DataFrame with OHLCV data
        patterns: List of pattern detection result dictionaries
        title: Chart title
        show_volume: Whether to include volume bars
        
    Returns:
        Plotly figure object with candlestick chart and highlighted patterns
    """
    # Filter valid patterns
    valid_patterns = [p for p in patterns if p and p.get('pattern', 'none') != 'none']
    
    if not valid_patterns:
        # No valid patterns, return basic chart
        return create_candlestick_chart(data, title=f"{title} (No patterns detected)", 
                                       show_volume=show_volume, rangeslider=False)
    
    # Create base candlestick chart
    fig = create_candlestick_chart(data, title=title, show_volume=show_volume, rangeslider=False)
    
    # Add patterns to the annotation text
    pattern_names = [p.get('pattern', '').replace('_', ' ').title() for p in valid_patterns]
    confidence_values = [p.get('confidence', 0) for p in valid_patterns]
    
    # Create pattern annotation text
    patterns_text = "<br>".join([f"{name} (Conf: {conf:.2f})" for name, conf in zip(pattern_names, confidence_values)])
    
    # Update chart title
    fig.update_layout(title=f"{title}<br><sup>{patterns_text}</sup>")
    
    # Limited to top 3 patterns by confidence to avoid clutter
    top_patterns = sorted(valid_patterns, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
    
    # Use different colors for each pattern
    pattern_colors = ['rgba(255, 0, 0, 0.3)', 'rgba(0, 0, 255, 0.3)', 'rgba(0, 255, 0, 0.3)']
    
    # Add shaded regions for each pattern
    for i, pattern in enumerate(top_patterns):
        pattern_name = pattern.get('pattern', '')
        
        # Extract indices based on pattern type
        highlight_indices = []
        
        if 'head_and_shoulders' in pattern_name:
            left_idx = pattern.get('left_shoulder_idx', 0)
            right_idx = pattern.get('right_shoulder_idx', 0)
            if left_idx > 0 and right_idx > left_idx:
                highlight_indices = list(range(left_idx, right_idx + 1))
                
        elif 'double' in pattern_name:
            first_idx = pattern.get('first_peak_idx' if 'top' in pattern_name else 'first_trough_idx', 0)
            second_idx = pattern.get('second_peak_idx' if 'top' in pattern_name else 'second_trough_idx', 0)
            if first_idx > 0 and second_idx > first_idx:
                highlight_indices = list(range(first_idx, second_idx + 1))
                
        elif 'triangle' in pattern_name:
            peaks = pattern.get('recent_peaks', [])
            troughs = pattern.get('recent_troughs', [])
            if peaks and troughs:
                all_points = peaks + troughs
                min_idx = min(all_points)
                max_idx = max(all_points)
                highlight_indices = list(range(min_idx, max_idx + 1))
                
        elif 'cup_and_handle' in pattern_name:
            left_idx = pattern.get('left_peak_idx', 0)
            right_idx = pattern.get('right_peak_idx', 0)
            if left_idx > 0 and right_idx > left_idx:
                highlight_indices = list(range(left_idx, right_idx + 10))  # Include handle
                
        elif 'flag' in pattern_name or 'pennant' in pattern_name:
            pole_start_idx = pattern.get('pole_start_idx', 0)
            pole_end_idx = pattern.get('pole_end_idx', 0)
            if pole_start_idx > 0 and pole_end_idx > pole_start_idx:
                flag_end_idx = min(pole_end_idx + 20, len(data) - 1)
                highlight_indices = list(range(pole_start_idx, flag_end_idx + 1))
                
        elif 'wedge' in pattern_name:
            # Use last 20-30 candles for wedge
            wedge_length = min(30, len(data) // 3)
            start_idx = max(0, len(data) - wedge_length)
            end_idx = len(data) - 1
            highlight_indices = list(range(start_idx, end_idx + 1))
        
        # Highlight pattern region if indices are available
        if highlight_indices:
            # Get dates for the highlight region
            indexes = data.index.tolist()
            valid_indices = [idx for idx in highlight_indices if idx < len(indexes)]
            
            if valid_indices:
                # Get date range
                start_date = indexes[min(valid_indices)]
                end_date = indexes[max(valid_indices)]
                
                # Get price range
                price_section = data.iloc[valid_indices]
                min_price = price_section['Low'].min() * 0.99  # Add some margin
                max_price = price_section['High'].max() * 1.01  # Add some margin
                
                # Add shaded rectangle
                fig.add_shape(
                    type='rect',
                    x0=start_date,
                    x1=end_date,
                    y0=min_price,
                    y1=max_price,
                    fillcolor=pattern_colors[i % len(pattern_colors)],
                    opacity=0.2,
                    layer='below',
                    line=dict(width=0)
                )
                
                # Add pattern label
                fig.add_annotation(
                    x=indexes[valid_indices[len(valid_indices)//2]],
                    y=max_price,
                    text=pattern_name.replace('_', ' ').title(),
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    borderwidth=1
                )
    
    return fig

def create_pattern_comparison_chart(ticker: str, 
                                  patterns: List[Dict],
                                  data: pd.DataFrame,
                                  title: str = 'Pattern Detection History') -> go.Figure:
    """
    Create a chart comparing patterns detected over time.
    
    Args:
        ticker: Stock ticker symbol
        patterns: List of pattern detection result dictionaries with timestamps
        data: DataFrame with price data
        title: Chart title
        
    Returns:
        Plotly figure object with pattern comparison chart
    """
    # Filter valid patterns
    valid_patterns = [p for p in patterns if p and p.get('pattern', 'none') != 'none']
    
    if not valid_patterns:
        # Create a basic price chart with no patterns
        fig = go.Figure(data=go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price'
        ))
        
        fig.update_layout(
            title=f"{title} - No patterns detected",
            height=600,
            width=1000,
            xaxis_title='Date',
            yaxis_title='Price'
        )
        
        return fig
    
    # Create a figure with price chart
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
    ))
    
    # Create pattern categories and colors
    pattern_categories = {
        'head_and_shoulders': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'inverse_head_and_shoulders': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'double_top': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'double_bottom': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'ascending_triangle': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'descending_triangle': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'symmetrical_triangle': {'color': 'blue', 'symbol': 'diamond', 'trend': 'neutral'},
        'cup_and_handle': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'bullish_flag': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'bearish_flag': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'bullish_pennant': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'},
        'bearish_pennant': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'rising_wedge': {'color': 'red', 'symbol': 'triangle-down', 'trend': 'bearish'},
        'falling_wedge': {'color': 'green', 'symbol': 'triangle-up', 'trend': 'bullish'}
    }
    
    # Add each pattern as a marker
    for pattern in valid_patterns:
        pattern_name = pattern.get('pattern', '')
        confidence = pattern.get('confidence', 0)
        date = pattern.get('date', data.index[-1])  # Default to last date if not provided
        
        # Get pattern category properties
        props = pattern_categories.get(pattern_name, 
                                     {'color': 'gray', 'symbol': 'circle', 'trend': 'unknown'})
        
        # Get price at this date
        price = data.loc[data.index <= date, 'Close'].iloc[-1] if not data.empty else 0
        
        # Add marker
        fig.add_trace(go.Scatter(
            x=[date],
            y=[price],
            mode='markers',
            marker=dict(
                size=10 + confidence * 10,  # Size based on confidence
                color=props['color'],
                symbol=props['symbol'],
                line=dict(width=1, color='black')
            ),
            name=pattern_name.replace('_', ' ').title(),
            hoverinfo='text',
            hovertext=f"{pattern_name.replace('_', ' ').title()}<br>Confidence: {confidence:.2f}<br>Date: {date}"
        ))
    
    # Add marker for bullish patterns
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='green', symbol='triangle-up'),
        name='Bullish Patterns'
    ))
    
    # Add marker for bearish patterns
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='red', symbol='triangle-down'),
        name='Bearish Patterns'
    ))
    
    # Add marker for neutral patterns
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='diamond'),
        name='Neutral Patterns'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {ticker}",
        height=600,
        width=1000,
        xaxis_title='Date',
        yaxis_title='Price',
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