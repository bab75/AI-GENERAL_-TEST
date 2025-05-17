"""
Advanced Analytics Page

This page provides access to advanced analytics features including:
- Anomaly detection
- Pattern recognition
- Advanced visualizations
- Enhanced report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import logging
from datetime import datetime
from scipy import stats

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from components import stock_selector, timeframe_selector
from utils import data_fetcher

# Import advanced analytics modules
from advanced_analytics import anomaly_detection, pattern_recognition, report_generation
from advanced_analytics.advanced_visualizations import (
    heatmap, network_graph, candlestick_patterns, prediction_bands
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "advanced_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    """Main function for the Advanced Analytics page"""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    # Page title and description
    st.title("Advanced Analytics")
    st.write("""
    This page provides access to advanced analytics features including anomaly detection, 
    pattern recognition, advanced visualizations, and enhanced report generation.
    """)
    
    # Sidebar
    st.sidebar.title("Advanced Analytics")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Anomaly Detection",
            "Pattern Recognition",
            "Advanced Visualizations",
            "Enhanced Report Generation"
        ]
    )
    
    # Run the selected analysis
    if analysis_type == "Anomaly Detection":
        run_anomaly_detection()
    elif analysis_type == "Pattern Recognition":
        run_pattern_recognition()
    elif analysis_type == "Advanced Visualizations":
        run_advanced_visualizations()
    elif analysis_type == "Enhanced Report Generation":
        run_enhanced_reporting()

def run_anomaly_detection():
    """Run the anomaly detection analysis"""
    
    st.header("Anomaly Detection")
    st.write("""
    This feature uses statistical and machine learning methods to detect anomalies in stock price
    and volume data. Anomalies can include unusual price movements, volume spikes, volatility changes,
    and more.
    """)
    
    # Select stock
    ticker = stock_selector.display_stock_selector()
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Anomaly detection parameters
    st.subheader("Anomaly Detection Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        z_score_threshold = st.slider(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="Higher values detect only more extreme anomalies"
        )
        
        volume_change_threshold = st.slider(
            "Volume Change Threshold (%)",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Minimum percentage increase to qualify as a volume spike"
        )
    
    with col2:
        window_size = st.slider(
            "Analysis Window Size",
            min_value=5,
            max_value=30,
            value=20,
            step=1,
            help="Window size for moving average calculations"
        )
        
        volatility_threshold = st.slider(
            "Volatility Threshold",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help="Threshold for detecting volatility anomalies"
        )
    
    # AI Interpretation option
    use_ai = st.checkbox(
        "Use AI for Enhanced Interpretation", 
        value=False,
        help="Use AI to provide deeper analysis of detected anomalies (requires API key)"
    )
    
    # Check if API keys are available if AI is requested
    if use_ai:
        if not check_api_keys():
            st.warning("To use AI interpretation, you need to provide API keys for OpenAI or Anthropic.")
            use_ai = False
    
    # Create anomaly detector configuration
    config = {
        'z_score_threshold': z_score_threshold,
        'window_size': window_size,
        'volume_change_threshold': volume_change_threshold / 100,  # Convert to decimal
        'volatility_threshold': volatility_threshold,
        'use_ai': use_ai
    }
    
    # Run analysis button
    if st.button("Detect Anomalies"):
        with st.spinner("Fetching data and detecting anomalies..."):
            # Fetch data
            data = data_fetcher.get_stock_data(ticker, period=period, interval=interval)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Run anomaly detection
            results = anomaly_detection.detect_anomalies(ticker, data, config)
            
            if 'error' in results:
                st.error(f"Error detecting anomalies: {results['error']}")
                return
            
            # Display results
            display_anomaly_results(results, data)

def display_anomaly_results(results, data):
    """Display the results of anomaly detection"""
    
    # Summary information
    st.subheader("Anomaly Detection Results")
    
    anomaly_count = results.get('anomaly_count', 0)
    
    if anomaly_count == 0:
        st.info("No anomalies detected in the selected time period.")
        return
    
    st.info(f"Detected {anomaly_count} anomalies in the selected time period.")
    
    # Get anomaly data
    anomaly_data = results.get('anomaly_data', pd.DataFrame())
    
    if not anomaly_data.empty:
        # Create a consolidated dataframe for the most severe anomalies
        anomaly_dates = list(results.get('interpretations', {}).get('anomalies', {}).keys())
        
        # Create a new dataframe with columns for display with anomaly score
        display_df = anomaly_data.copy()
        
        # Create a price chart with anomalies highlighted
        price_fig = go.Figure()
        
        # Add price trace
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=1)
        ))
        
        # Add markers for anomalies
        price_fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data['Close'],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='circle',
                line=dict(color='red', width=1)
            ),
            name='Anomalies',
            hoverinfo='text',
            hovertext=[f"Date: {idx}<br>Close: ${row['Close']:.2f}<br>Anomaly Score: {row['anomaly_score']}" 
                     for idx, row in anomaly_data.iterrows()]
        ))
        
        # Update layout
        price_fig.update_layout(
            title=f"Price Chart with Detected Anomalies",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Display anomaly details
        st.subheader("Anomaly Details")
        
        # Create tabs for each anomaly
        tabs = st.tabs([str(date) for date in anomaly_dates[:5]])  # Show top 5 anomalies
        
        for i, (tab, date) in enumerate(zip(tabs, anomaly_dates[:5])):
            with tab:
                anomaly_info = results.get('interpretations', {}).get('anomalies', {}).get(date, {})
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Date:** {date}")
                    st.write(f"**Anomaly Score:** {anomaly_info.get('anomaly_score', 'N/A')}")
                    st.write(f"**Close Price:** ${anomaly_info.get('close_price', 'N/A'):.2f}")
                    st.write(f"**Volume:** {anomaly_info.get('volume', 'N/A'):,}")
                    
                    # Anomaly types
                    anomaly_types = anomaly_info.get('anomaly_types', [])
                    if anomaly_types:
                        st.write("**Anomaly Types:**")
                        for atype in anomaly_types:
                            st.write(f"- {atype}")
                    
                    # Interpretation
                    st.write("**Interpretation:**")
                    st.write(anomaly_info.get('interpretation', 'No interpretation available.'))
                    
                    # AI Interpretation (if available)
                    if 'ai_interpretation' in anomaly_info:
                        st.write("**AI Interpretation:**")
                        st.info(anomaly_info.get('ai_interpretation', 'No AI interpretation available.'))
                
                with col2:
                    # Get a window of data around the anomaly
                    anomaly_date = date
                    
                    try:
                        # Convert to timestamp if string
                        if isinstance(anomaly_date, str):
                            anomaly_date = pd.Timestamp(anomaly_date)
                        
                        # Find the index position of the anomaly date
                        idx = data.index.get_indexer([anomaly_date], method='nearest')[0]
                        
                        # Get a window around the anomaly
                        start_idx = max(0, idx - 10)
                        end_idx = min(len(data) - 1, idx + 10)
                        
                        window_data = data.iloc[start_idx:end_idx]
                        
                        # Create a figure
                        fig = go.Figure()
                        
                        # Add price trace
                        fig.add_trace(go.Scatter(
                            x=window_data.index,
                            y=window_data['Close'],
                            mode='lines+markers',
                            name='Price',
                            line=dict(color='blue', width=1)
                        ))
                        
                        # Add markers for anomaly
                        fig.add_trace(go.Scatter(
                            x=[anomaly_date],
                            y=[data.loc[data.index == anomaly_date, 'Close'].values[0]],
                            mode='markers',
                            marker=dict(
                                size=12,
                                color='red',
                                symbol='star'
                            ),
                            name='Anomaly'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Price Around Anomaly",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting anomaly window: {str(e)}")
        
        # Display anomaly data table
        with st.expander("View All Anomaly Data"):
            # Format anomaly_data for display
            display_columns = ['Close', 'Volume', 'anomaly_score']
            
            # Add anomaly type columns if available
            for col in anomaly_data.columns:
                if col.endswith('_anomaly') and col not in display_columns:
                    display_columns.append(col)
            
            st.dataframe(anomaly_data[display_columns])

def run_pattern_recognition():
    """Run the pattern recognition analysis"""
    
    st.header("Pattern Recognition")
    st.write("""
    This feature detects common chart patterns such as head and shoulders, double tops/bottoms, 
    triangles, flags, and more. It provides visualizations and interpretations of the detected patterns.
    """)
    
    # Select stock
    ticker = stock_selector.display_stock_selector()
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Pattern recognition parameters
    st.subheader("Pattern Recognition Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smoothing_period = st.slider(
            "Smoothing Period",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Window size for smoothing price data"
        )
        
        threshold_pct = st.slider(
            "Threshold Percentage",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Percentage threshold for pattern recognition"
        )
    
    with col2:
        peak_distance = st.slider(
            "Minimum Peak Distance",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help="Minimum distance between peaks/troughs"
        )
        
        pattern_window = st.slider(
            "Analysis Window Size",
            min_value=60,
            max_value=252,
            value=120,
            step=10,
            help="Window size for pattern detection (number of days)"
        )
    
    # Create pattern recognizer configuration
    config = {
        'smoothing_period': smoothing_period,
        'peak_distance': peak_distance,
        'threshold_pct': threshold_pct,
        'pattern_window': pattern_window
    }
    
    # Run analysis button
    if st.button("Detect Patterns"):
        with st.spinner("Fetching data and detecting patterns..."):
            # Fetch data
            data = data_fetcher.get_stock_data(ticker, period=period, interval=interval)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Run pattern recognition
            results = pattern_recognition.detect_patterns(ticker, data, config)
            
            if 'error' in results:
                st.error(f"Error detecting patterns: {results['error']}")
                return
            
            # Display results
            display_pattern_results(results, data, ticker)

def display_pattern_results(results, data, ticker):
    """Display the results of pattern recognition"""
    
    # Get primary pattern
    primary_pattern = results.get('primary_pattern', {})
    all_patterns = results.get('all_patterns', [])
    
    if primary_pattern.get('pattern', 'none') == 'none':
        st.info("No significant patterns detected in the selected time period.")
        
        # Show basic candlestick chart
        fig = candlestick_patterns.create_candlestick_chart(
            data, 
            title=f"{ticker} - No Significant Patterns Detected",
            show_volume=True
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Display primary pattern
    st.subheader("Primary Pattern Detected")
    
    pattern_name = primary_pattern.get('pattern', '').replace('_', ' ').title()
    confidence = primary_pattern.get('confidence', 0)
    
    st.write(f"**Pattern:** {pattern_name}")
    st.write(f"**Confidence:** {confidence:.2f}")
    
    # Display pattern visualization
    fig = candlestick_patterns.create_pattern_candlestick_chart(
        data, 
        primary_pattern,
        title=f"{ticker} - {pattern_name} Pattern",
        show_volume=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display pattern details
    with st.expander("Pattern Details"):
        # Display specific pattern properties
        for key, value in primary_pattern.items():
            if key not in ['pattern', 'confidence']:
                # Format the key
                formatted_key = key.replace('_', ' ').title()
                
                # Format the value
                if isinstance(value, (int, float)):
                    if 'price' in key or 'level' in key:
                        st.write(f"**{formatted_key}:** ${value:.2f}")
                    elif 'idx' in key:
                        st.write(f"**{formatted_key}:** {value}")
                    else:
                        st.write(f"**{formatted_key}:** {value:.4f}")
                else:
                    st.write(f"**{formatted_key}:** {value}")
    
    # Display pattern interpretation
    with st.expander("Pattern Interpretation"):
        # Provide interpretation based on the pattern type
        if 'head_and_shoulders' in primary_pattern.get('pattern', ''):
            is_inverse = 'inverse' in primary_pattern.get('pattern', '')
            
            if is_inverse:
                st.write("""
                **Inverse Head and Shoulders**
                
                This bullish reversal pattern typically forms at the end of a downtrend and signals a potential trend change from bearish to bullish. It consists of:
                
                - A left shoulder (first trough)
                - A head (lower/deeper trough)
                - A right shoulder (third trough at similar level to the first)
                - A neckline connecting the peaks between these troughs
                
                **Trading Implication:** The pattern is considered complete when price breaks above the neckline. The projected upside target is typically the distance from the head to the neckline, added to the breakout point.
                
                **Success Rate:** Inverse head and shoulders patterns have a generally high reliability, especially with higher volume on the breakout.
                """)
            else:
                st.write("""
                **Head and Shoulders**
                
                This bearish reversal pattern typically forms at the end of an uptrend and signals a potential trend change from bullish to bearish. It consists of:
                
                - A left shoulder (first peak)
                - A head (higher peak)
                - A right shoulder (third peak at similar level to the first)
                - A neckline connecting the troughs between these peaks
                
                **Trading Implication:** The pattern is considered complete when price breaks below the neckline. The projected downside target is typically the distance from the head to the neckline, projected downward from the breakout point.
                
                **Success Rate:** Head and shoulders patterns have a generally high reliability, especially with higher volume on the breakdown.
                """)
        
        elif 'double_top' in primary_pattern.get('pattern', ''):
            st.write("""
            **Double Top**
            
            This bearish reversal pattern occurs after an uptrend and signals a potential trend change from bullish to bearish. It consists of:
            
            - Two peaks at approximately the same price level
            - A moderate trough between the peaks
            
            **Trading Implication:** The pattern is confirmed when price breaks below the trough between the peaks. The projected downside target is typically the height of the pattern (peak to trough) projected down from the breakdown point.
            
            **Success Rate:** Double tops have moderate reliability, with stronger signals when the second peak has lower volume than the first.
            """)
        
        elif 'double_bottom' in primary_pattern.get('pattern', ''):
            st.write("""
            **Double Bottom**
            
            This bullish reversal pattern occurs after a downtrend and signals a potential trend change from bearish to bullish. It consists of:
            
            - Two troughs at approximately the same price level
            - A moderate peak between the troughs
            
            **Trading Implication:** The pattern is confirmed when price breaks above the peak between the troughs. The projected upside target is typically the height of the pattern (peak to trough) projected up from the breakout point.
            
            **Success Rate:** Double bottoms have moderate reliability, with stronger signals when the second trough has higher volume than the first.
            """)
        
        elif 'triangle' in primary_pattern.get('pattern', ''):
            if 'ascending' in primary_pattern.get('pattern', ''):
                st.write("""
                **Ascending Triangle**
                
                This bullish continuation pattern typically forms during an uptrend and signals a potential continuation of the bullish trend. It consists of:
                
                - A horizontal resistance line (flat top)
                - An upward-sloping support line (rising bottoms)
                
                **Trading Implication:** The pattern is confirmed when price breaks above the horizontal resistance line. The projected upside target is typically the height of the triangle at its widest point, projected up from the breakout point.
                
                **Success Rate:** Ascending triangles have good reliability when they form in the direction of the overall trend.
                """)
            elif 'descending' in primary_pattern.get('pattern', ''):
                st.write("""
                **Descending Triangle**
                
                This bearish continuation pattern typically forms during a downtrend and signals a potential continuation of the bearish trend. It consists of:
                
                - A horizontal support line (flat bottom)
                - A downward-sloping resistance line (lower tops)
                
                **Trading Implication:** The pattern is confirmed when price breaks below the horizontal support line. The projected downside target is typically the height of the triangle at its widest point, projected down from the breakdown point.
                
                **Success Rate:** Descending triangles have good reliability when they form in the direction of the overall trend.
                """)
            else:
                st.write("""
                **Symmetrical Triangle**
                
                This neutral continuation pattern can form during either uptrends or downtrends and signals a consolidation before a potential continuation of the existing trend. It consists of:
                
                - A downward-sloping resistance line (lower tops)
                - An upward-sloping support line (higher bottoms)
                
                **Trading Implication:** The pattern is confirmed when price breaks out of the triangle in either direction. The projected target is typically the height of the triangle at its widest point, projected from the breakout point in the direction of the breakout.
                
                **Success Rate:** Symmetrical triangles tend to continue in the direction of the prevailing trend, but the direction of the breakout should be monitored to confirm.
                """)
        
        elif 'cup_and_handle' in primary_pattern.get('pattern', ''):
            st.write("""
            **Cup and Handle**
            
            This bullish continuation pattern typically forms during an uptrend and signals a potential continuation of the bullish trend. It consists of:
            
            - A rounded bottom formation (the cup)
            - A slight downward drift following the cup (the handle)
            
            **Trading Implication:** The pattern is confirmed when price breaks above the resistance level formed by the cup rim. The projected upside target is typically the depth of the cup added to the breakout point.
            
            **Success Rate:** Cup and handle patterns have good reliability, especially when they form after a strong prior uptrend.
            """)
        
        elif 'flag' in primary_pattern.get('pattern', ''):
            if 'bullish' in primary_pattern.get('pattern', ''):
                st.write("""
                **Bullish Flag**
                
                This bullish continuation pattern typically forms during an uptrend after a strong price advance (the flagpole) and signals a potential continuation of the bullish trend. It consists of:
                
                - A sharp upward move (the flagpole)
                - A rectangular pattern or channel tilting against the prior trend (the flag)
                
                **Trading Implication:** The pattern is confirmed when price breaks above the upper trendline of the flag. The projected upside target is typically the length of the flagpole added to the breakout point.
                
                **Success Rate:** Bullish flags have high reliability, especially when they form after a strong prior uptrend and the flag portion consolidates on decreasing volume.
                """)
            else:
                st.write("""
                **Bearish Flag**
                
                This bearish continuation pattern typically forms during a downtrend after a strong price decline (the flagpole) and signals a potential continuation of the bearish trend. It consists of:
                
                - A sharp downward move (the flagpole)
                - A rectangular pattern or channel tilting against the prior trend (the flag)
                
                **Trading Implication:** The pattern is confirmed when price breaks below the lower trendline of the flag. The projected downside target is typically the length of the flagpole projected down from the breakdown point.
                
                **Success Rate:** Bearish flags have high reliability, especially when they form after a strong prior downtrend and the flag portion consolidates on decreasing volume.
                """)
        
        elif 'pennant' in primary_pattern.get('pattern', ''):
            if 'bullish' in primary_pattern.get('pattern', ''):
                st.write("""
                **Bullish Pennant**
                
                This bullish continuation pattern is similar to a bullish flag but with converging trendlines creating a small symmetrical triangle. It typically forms during an uptrend after a strong price advance (the flagpole) and signals a potential continuation of the bullish trend. It consists of:
                
                - A sharp upward move (the flagpole)
                - A small symmetrical triangle pattern with converging trendlines (the pennant)
                
                **Trading Implication:** The pattern is confirmed when price breaks above the upper trendline of the pennant. The projected upside target is typically the length of the flagpole added to the breakout point.
                
                **Success Rate:** Bullish pennants have high reliability, especially when they form after a strong prior uptrend and the pennant portion consolidates on decreasing volume.
                """)
            else:
                st.write("""
                **Bearish Pennant**
                
                This bearish continuation pattern is similar to a bearish flag but with converging trendlines creating a small symmetrical triangle. It typically forms during a downtrend after a strong price decline (the flagpole) and signals a potential continuation of the bearish trend. It consists of:
                
                - A sharp downward move (the flagpole)
                - A small symmetrical triangle pattern with converging trendlines (the pennant)
                
                **Trading Implication:** The pattern is confirmed when price breaks below the lower trendline of the pennant. The projected downside target is typically the length of the flagpole projected down from the breakdown point.
                
                **Success Rate:** Bearish pennants have high reliability, especially when they form after a strong prior downtrend and the pennant portion consolidates on decreasing volume.
                """)
        
        elif 'wedge' in primary_pattern.get('pattern', ''):
            if 'rising' in primary_pattern.get('pattern', ''):
                st.write("""
                **Rising Wedge**
                
                This bearish pattern can be either a reversal pattern at the end of an uptrend or a continuation pattern during a downtrend. It consists of:
                
                - Converging trendlines where both are sloping upward
                - The upper trendline has a shallower slope than the lower trendline
                
                **Trading Implication:** The pattern is confirmed when price breaks below the lower trendline. The projected downside target is typically the height of the wedge at its widest point, projected down from the breakdown point.
                
                **Success Rate:** Rising wedges have moderate reliability, with stronger signals when they form after extended uptrends and when volume decreases during the formation.
                """)
            else:
                st.write("""
                **Falling Wedge**
                
                This bullish pattern can be either a reversal pattern at the end of a downtrend or a continuation pattern during an uptrend. It consists of:
                
                - Converging trendlines where both are sloping downward
                - The lower trendline has a shallower slope than the upper trendline
                
                **Trading Implication:** The pattern is confirmed when price breaks above the upper trendline. The projected upside target is typically the height of the wedge at its widest point, projected up from the breakout point.
                
                **Success Rate:** Falling wedges have moderate reliability, with stronger signals when they form after extended downtrends and when volume decreases during the formation.
                """)
    
    # Display all detected patterns
    if len(all_patterns) > 1:
        st.subheader("All Detected Patterns")
        
        # Create columns for patterns
        cols = st.columns(min(3, len(all_patterns)))
        
        for i, pattern in enumerate(all_patterns):
            col = cols[i % len(cols)]
            
            with col:
                pattern_name = pattern.get('pattern', '').replace('_', ' ').title()
                confidence = pattern.get('confidence', 0)
                
                st.metric(
                    label=pattern_name,
                    value=f"{confidence:.2f}",
                    delta=None
                )
        
        # Option to show chart with all patterns
        if st.checkbox("Show Chart with All Patterns"):
            fig = candlestick_patterns.create_multiple_pattern_chart(
                data, 
                all_patterns,
                title=f"{ticker} - Multiple Patterns Detected",
                show_volume=True
            )
            st.plotly_chart(fig, use_container_width=True)

def run_advanced_visualizations():
    """Run the advanced visualizations analysis"""
    
    st.header("Advanced Visualizations")
    st.write("""
    This feature provides enhanced visualization capabilities for stock analysis, including
    heatmaps, network graphs, advanced candlestick visualizations, and prediction bands.
    """)
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Correlation Heatmap",
            "Sector Performance Heatmap",
            "Stock Relationship Network",
            "Pattern Visualization",
            "Prediction Bands"
        ]
    )
    
    # Run the selected visualization
    if viz_type == "Correlation Heatmap":
        run_correlation_heatmap()
    elif viz_type == "Sector Performance Heatmap":
        run_sector_heatmap()
    elif viz_type == "Stock Relationship Network":
        run_stock_network()
    elif viz_type == "Pattern Visualization":
        run_pattern_visualization()
    elif viz_type == "Prediction Bands":
        run_prediction_bands()

def run_correlation_heatmap():
    """Run the correlation heatmap visualization"""
    
    st.subheader("Stock Correlation Heatmap")
    st.write("""
    This visualization shows the correlation between multiple stocks over a selected time period.
    Higher correlation (closer to 1.0) indicates that stocks tend to move together.
    """)
    
    # Select multiple stocks
    tickers = stock_selector.display_multi_stock_selector(max_stocks=10)
    
    if not tickers:
        st.warning("Please select at least two stocks for correlation analysis.")
        return
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Correlation parameters
    st.subheader("Correlation Parameters")
    
    price_col = st.selectbox(
        "Select Price Column",
        ["Close", "Open", "High", "Low", "Adj Close"],
        index=0
    )
    
    normalize = st.checkbox(
        "Normalize Prices",
        value=True,
        help="Normalize prices to percentage change from first value"
    )
    
    # Visualization options
    viz_style = st.radio(
        "Visualization Style",
        ["Single Heatmap", "Multi-Timeframe Analysis"],
        index=0
    )
    
    # Run visualization button
    if st.button("Generate Correlation Heatmap"):
        with st.spinner("Fetching data and generating heatmap..."):
            # Fetch data for all tickers
            data_dict = data_fetcher.get_multiple_stocks_data(tickers, period=period, interval=interval)
            
            # Check if we have valid data
            valid_tickers = [t for t, df in data_dict.items() if not df.empty and price_col in df.columns]
            
            if len(valid_tickers) < 2:
                st.error("Could not fetch valid data for at least two tickers")
                return
            
            # Filter data_dict to only include valid tickers
            data_dict = {t: df for t, df in data_dict.items() if t in valid_tickers}
            
            # Generate the appropriate visualization
            if viz_style == "Single Heatmap":
                fig = heatmap.create_correlation_heatmap(
                    data_dict,
                    column=price_col,
                    period=period,
                    normalize=normalize,
                    title=f"Stock Correlation Heatmap ({price_col})"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Multi-timeframe analysis
                timeframes = [30, 90, 180, 365]  # 1mo, 3mo, 6mo, 1yr
                
                # Ensure we have enough data for the longest timeframe
                for ticker, df in data_dict.items():
                    if len(df) < max(timeframes):
                        st.warning(f"Not enough data for {ticker} to show all timeframes. Using available data.")
                        timeframes = [tf for tf in timeframes if tf <= len(df)]
                        break
                
                if not timeframes:
                    st.error("Not enough data to generate multi-timeframe analysis")
                    return
                
                fig = heatmap.create_multi_timeframe_correlation_heatmap(
                    data_dict,
                    column=price_col,
                    timeframes=timeframes,
                    title=f"Multi-Timeframe Correlation Analysis ({price_col})"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation interpretation
            with st.expander("Correlation Interpretation Guide"):
                st.write("""
                **Correlation Values:**
                
                - **1.0**: Perfect positive correlation. Stocks move in perfect tandem.
                - **0.7 to 0.9**: Strong positive correlation. Stocks generally move in the same direction.
                - **0.4 to 0.6**: Moderate positive correlation. Stocks often move in the same direction but with notable differences.
                - **0.1 to 0.3**: Weak positive correlation. Stocks sometimes move in the same direction.
                - **0.0**: No correlation. Stocks move independently of each other.
                - **-0.1 to -0.3**: Weak negative correlation. Stocks sometimes move in opposite directions.
                - **-0.4 to -0.6**: Moderate negative correlation. Stocks often move in opposite directions.
                - **-0.7 to -0.9**: Strong negative correlation. Stocks generally move in opposite directions.
                - **-1.0**: Perfect negative correlation. Stocks move in perfectly opposite directions.
                
                **Trading Implications:**
                
                - **Diversification**: Stocks with low correlation provide better portfolio diversification.
                - **Hedging**: Negatively correlated stocks can be used to hedge against market movements.
                - **Sector Insights**: Stocks within the same sector often have higher correlation.
                - **Rotation Strategies**: When correlation patterns change, it may indicate sector rotation.
                
                **Time Sensitivity:**
                
                Correlations can change over time. A multi-timeframe analysis provides insights into how relationships evolve across different time horizons.
                """)

def run_sector_heatmap():
    """Run the sector performance heatmap visualization"""
    
    st.subheader("Sector Performance Heatmap")
    st.write("""
    This visualization shows the relative performance of different market sectors.
    It helps identify which sectors are outperforming or underperforming.
    """)
    
    # Input file selection for snapshot data
    st.info("This visualization requires snapshot data (CSV or Excel file) with sector information.")
    
    uploaded_file = st.file_uploader("Upload snapshot file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.warning("Please upload a snapshot file to generate the sector heatmap.")
        return
    
    # Process the file
    try:
        # Determine file type
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        if file_ext == "csv":
            snapshot_data = pd.read_csv(uploaded_file)
        else:
            snapshot_data = pd.read_excel(uploaded_file)
        
        # Check for required columns
        required_columns = ['Sector', 'Price_Change_Pct']
        alternative_columns = {
            'Price_Change_Pct': ['% Change', 'Price Change %', 'PercentChange', 'Change_Pct']
        }
        
        # Handle different column names
        for col in required_columns:
            if col not in snapshot_data.columns:
                alternatives = alternative_columns.get(col, [])
                found = False
                
                for alt in alternatives:
                    if alt in snapshot_data.columns:
                        snapshot_data[col] = snapshot_data[alt]
                        found = True
                        break
                
                if not found and col == 'Price_Change_Pct':
                    # Try to calculate from other columns
                    if 'Close' in snapshot_data.columns and 'Previous_Close' in snapshot_data.columns:
                        snapshot_data['Price_Change_Pct'] = (snapshot_data['Close'] - snapshot_data['Previous_Close']) / snapshot_data['Previous_Close'] * 100
                    elif 'Last Sale' in snapshot_data.columns and 'Net Change' in snapshot_data.columns:
                        snapshot_data['Price_Change_Pct'] = (snapshot_data['Net Change'] / (snapshot_data['Last Sale'] - snapshot_data['Net Change'])) * 100
                
                if col not in snapshot_data.columns:
                    st.error(f"Required column '{col}' not found in the snapshot file. Please ensure the file contains the necessary data.")
                    return
        
        # Generate the heatmap
        fig = heatmap.create_sector_performance_heatmap(
            snapshot_data,
            sector_column='Sector',
            performance_column='Price_Change_Pct',
            title="Sector Performance Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add sector breakdown
        sector_counts = snapshot_data['Sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Count']
        
        sector_performance = snapshot_data.groupby('Sector')['Price_Change_Pct'].agg(['mean', 'min', 'max']).reset_index()
        sector_performance['mean'] = sector_performance['mean'].round(2)
        sector_performance['min'] = sector_performance['min'].round(2)
        sector_performance['max'] = sector_performance['max'].round(2)
        
        # Merge counts and performance
        sector_stats = pd.merge(sector_counts, sector_performance, on='Sector')
        sector_stats.columns = ['Sector', 'Stock Count', 'Avg Change %', 'Min Change %', 'Max Change %']
        
        # Sort by performance
        sector_stats = sector_stats.sort_values('Avg Change %', ascending=False)
        
        st.subheader("Sector Breakdown")
        st.dataframe(sector_stats, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing the snapshot file: {str(e)}")

def run_stock_network():
    """Run the stock relationship network visualization"""
    
    st.subheader("Stock Relationship Network")
    st.write("""
    This visualization shows the relationships between stocks as a network graph.
    Connections represent correlations, and node sizes represent the number of significant connections.
    """)
    
    # Select network type
    network_type = st.radio(
        "Select Network Type",
        ["Correlation Network", "Correlation Clusters", "Sector Network"],
        index=0
    )
    
    if network_type in ["Correlation Network", "Correlation Clusters"]:
        # Select multiple stocks
        tickers = stock_selector.display_multi_stock_selector(max_stocks=20)
        
        if len(tickers) < 5:
            st.warning("Please select at least 5 stocks for a meaningful network visualization.")
            return
        
        # Select timeframe
        period, interval = timeframe_selector.display_timeframe_selector()
        
        # Network parameters
        st.subheader("Network Parameters")
        
        price_col = st.selectbox(
            "Select Price Column",
            ["Close", "Open", "High", "Low", "Adj Close"],
            index=0
        )
        
        min_correlation = st.slider(
            "Minimum Correlation",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Minimum correlation value to show as an edge"
        )
        
        max_edges = st.slider(
            "Maximum Edges",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of edges to display"
        )
        
        # Run visualization button
        if st.button("Generate Network Graph"):
            with st.spinner("Fetching data and generating network..."):
                # Fetch data for all tickers
                data_dict = data_fetcher.get_multiple_stocks_data(tickers, period=period, interval=interval)
                
                # Check if we have valid data
                valid_tickers = [t for t, df in data_dict.items() if not df.empty and price_col in df.columns]
                
                if len(valid_tickers) < 5:
                    st.error("Could not fetch valid data for at least 5 tickers")
                    return
                
                # Filter data_dict to only include valid tickers
                data_dict = {t: df for t, df in data_dict.items() if t in valid_tickers}
                
                # Generate the appropriate visualization
                if network_type == "Correlation Network":
                    fig = network_graph.create_correlation_network(
                        data_dict,
                        column=price_col,
                        min_correlation=min_correlation,
                        max_edges=max_edges,
                        period=period,
                        normalize=True,
                        title="Stock Correlation Network"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Check if scikit-learn is available for clustering
                    try:
                        from sklearn.cluster import AgglomerativeClustering
                        
                        # Set number of clusters
                        n_clusters = st.slider(
                            "Number of Clusters",
                            min_value=2,
                            max_value=10,
                            value=5,
                            step=1
                        )
                        
                        fig = network_graph.create_correlation_cluster_map(
                            data_dict,
                            column=price_col,
                            period=period,
                            n_clusters=n_clusters,
                            title="Stock Correlation Clusters"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.error("Scikit-learn is required for correlation clustering. Please install it with 'pip install scikit-learn'.")
    else:
        # Sector Network
        st.info("This visualization requires snapshot data (CSV or Excel file) with sector information.")
        
        uploaded_file = st.file_uploader("Upload snapshot file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is None:
            st.warning("Please upload a snapshot file to generate the sector network.")
            return
        
        # Process the file
        try:
            # Determine file type
            file_ext = uploaded_file.name.split(".")[-1].lower()
            
            if file_ext == "csv":
                snapshot_data = pd.read_csv(uploaded_file)
            else:
                snapshot_data = pd.read_excel(uploaded_file)
            
            # Check for required columns
            required_columns = ['Symbol', 'Sector', 'Price_Change_Pct', 'Market_Cap']
            alternative_columns = {
                'Symbol': ['Ticker', 'TickerSymbol'],
                'Price_Change_Pct': ['% Change', 'Price Change %', 'PercentChange', 'Change_Pct'],
                'Market_Cap': ['MarketCap', 'Market Cap', 'Mkt Cap']
            }
            
            # Handle different column names
            for col in required_columns:
                if col not in snapshot_data.columns:
                    alternatives = alternative_columns.get(col, [])
                    found = False
                    
                    for alt in alternatives:
                        if alt in snapshot_data.columns:
                            snapshot_data[col] = snapshot_data[alt]
                            found = True
                            break
                    
                    if not found and col == 'Price_Change_Pct':
                        # Try to calculate from other columns
                        if 'Close' in snapshot_data.columns and 'Previous_Close' in snapshot_data.columns:
                            snapshot_data['Price_Change_Pct'] = (snapshot_data['Close'] - snapshot_data['Previous_Close']) / snapshot_data['Previous_Close'] * 100
                        elif 'Last Sale' in snapshot_data.columns and 'Net Change' in snapshot_data.columns:
                            snapshot_data['Price_Change_Pct'] = (snapshot_data['Net Change'] / (snapshot_data['Last Sale'] - snapshot_data['Net Change'])) * 100
                    
                    if not found and col == 'Market_Cap':
                        # Use a default value
                        snapshot_data['Market_Cap'] = 1.0
                    
                    if not found and col == 'Symbol':
                        # Try to use the index
                        snapshot_data['Symbol'] = snapshot_data.index
                
                if col not in snapshot_data.columns:
                    st.error(f"Required column '{col}' not found in the snapshot file. Please ensure the file contains the necessary data.")
                    return
            
            # Network parameters
            min_stocks = st.slider(
                "Minimum Stocks per Sector",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Minimum number of stocks in a sector to include"
            )
            
            # Generate the network
            fig = network_graph.create_sector_network(
                snapshot_data,
                sector_column='Sector',
                performance_column='Price_Change_Pct',
                size_column='Market_Cap',
                min_stocks=min_stocks,
                title="Sector Performance Network"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add market map visualization
            st.subheader("Market Map Visualization")
            st.write("This treemap shows the market landscape with size representing market cap and color representing performance.")
            
            # Generate the market map
            fig = network_graph.create_market_map(
                snapshot_data,
                size_column='Market_Cap',
                performance_column='Price_Change_Pct',
                sector_column='Sector',
                title="Market Map"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing the snapshot file: {str(e)}")

def run_pattern_visualization():
    """Run the advanced candlestick pattern visualization"""
    
    st.subheader("Advanced Candlestick Pattern Visualization")
    st.write("""
    This visualization provides enhanced candlestick charts with pattern annotations
    and interpretations.
    """)
    
    # Select stock
    ticker = stock_selector.display_stock_selector()
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Visualization options
    st.subheader("Visualization Options")
    
    viz_style = st.radio(
        "Select Visualization Style",
        ["Simple Candlestick", "Pattern Detection", "Multiple Pattern Comparison"],
        index=0
    )
    
    show_volume = st.checkbox("Show Volume", value=True)
    
    # If pattern detection is selected, show pattern recognition parameters
    pattern_results = None
    
    if viz_style in ["Pattern Detection", "Multiple Pattern Comparison"]:
        # Pattern recognition parameters
        st.subheader("Pattern Recognition Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smoothing_period = st.slider(
                "Smoothing Period",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Window size for smoothing price data"
            )
            
            threshold_pct = st.slider(
                "Threshold Percentage",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Percentage threshold for pattern recognition"
            )
        
        with col2:
            peak_distance = st.slider(
                "Minimum Peak Distance",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                help="Minimum distance between peaks/troughs"
            )
            
            pattern_window = st.slider(
                "Analysis Window Size",
                min_value=60,
                max_value=252,
                value=120,
                step=10,
                help="Window size for pattern detection (number of days)"
            )
        
        # Create pattern recognizer configuration
        config = {
            'smoothing_period': smoothing_period,
            'peak_distance': peak_distance,
            'threshold_pct': threshold_pct,
            'pattern_window': pattern_window
        }
    
    # For technical indicators visualization, add options
    if viz_style == "Simple Candlestick":
        indicator_type = st.selectbox(
            "Select Technical Indicator",
            ["None", "Bollinger Bands", "Keltner Channels", "Donchian Channels", "Parabolic SAR", "Support/Resistance Levels"],
            index=0
        )
    
    # Run visualization button
    if st.button("Generate Visualization"):
        with st.spinner("Fetching data and generating visualization..."):
            # Fetch data
            data = data_fetcher.get_stock_data(ticker, period=period, interval=interval)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Generate the appropriate visualization
            if viz_style == "Simple Candlestick":
                if indicator_type == "None":
                    fig = candlestick_patterns.create_candlestick_chart(
                        data, 
                        title=f"{ticker} - Candlestick Chart",
                        show_volume=show_volume
                    )
                elif indicator_type == "Bollinger Bands":
                    fig = prediction_bands.create_bollinger_bands_chart(
                        data,
                        window=20,
                        num_std=2.0,
                        column='Close',
                        title=f"{ticker} - Bollinger Bands",
                        show_volume=show_volume
                    )
                elif indicator_type in ["Keltner Channels", "Donchian Channels", "Parabolic SAR"]:
                    bands_type = indicator_type.split()[0].lower()
                    fig = prediction_bands.create_technical_bands_chart(
                        data,
                        bands_type=bands_type,
                        column='Close',
                        title=f"{ticker} - {indicator_type}",
                        show_volume=show_volume
                    )
                elif indicator_type == "Support/Resistance Levels":
                    fig = prediction_bands.create_support_resistance_chart(
                        data,
                        auto_detect=True,
                        n_levels=5,
                        title=f"{ticker} - Support and Resistance Levels",
                        show_volume=show_volume
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_style == "Pattern Detection":
                # Run pattern recognition
                results = pattern_recognition.detect_patterns(ticker, data, config)
                
                if 'error' in results:
                    st.error(f"Error detecting patterns: {results['error']}")
                    return
                
                # Get primary pattern
                primary_pattern = results.get('primary_pattern', {})
                
                # Create pattern visualization
                fig = candlestick_patterns.create_pattern_candlestick_chart(
                    data, 
                    primary_pattern,
                    title=f"{ticker} - Pattern Detection",
                    show_volume=show_volume
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store pattern results for interpretation
                pattern_results = results
                
                # Display pattern interpretation
                display_pattern_interpretation(primary_pattern)
            
            elif viz_style == "Multiple Pattern Comparison":
                # Run pattern recognition
                results = pattern_recognition.detect_patterns(ticker, data, config)
                
                if 'error' in results:
                    st.error(f"Error detecting patterns: {results['error']}")
                    return
                
                # Get all patterns
                all_patterns = results.get('all_patterns', [])
                
                # Create multiple pattern visualization
                fig = candlestick_patterns.create_multiple_pattern_chart(
                    data, 
                    all_patterns,
                    title=f"{ticker} - Multiple Pattern Detection",
                    show_volume=show_volume
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store pattern results for interpretation
                pattern_results = results
                
                # Display pattern summary
                if all_patterns:
                    st.subheader("Detected Patterns Summary")
                    
                    pattern_df = pd.DataFrame([
                        {
                            'Pattern': p.get('pattern', '').replace('_', ' ').title(),
                            'Confidence': p.get('confidence', 0),
                            'Type': 'Bullish' if any(t in p.get('pattern', '') for t in ['bullish', 'inverse', 'bottom', 'falling_wedge']) else
                                  'Bearish' if any(t in p.get('pattern', '') for t in ['bearish', 'head_and_shoulders', 'top', 'rising_wedge']) else
                                  'Neutral'
                        }
                        for p in all_patterns
                    ])
                    
                    st.dataframe(pattern_df, use_container_width=True)
                else:
                    st.info("No patterns detected in the selected time period.")

def run_prediction_bands():
    """Run the prediction bands visualization"""
    
    st.subheader("Price Prediction Bands")
    st.write("""
    This visualization shows price prediction bands and confidence intervals
    for potential future price movements.
    """)
    
    # Select stock
    ticker = stock_selector.display_stock_selector()
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Prediction type
    pred_type = st.radio(
        "Select Prediction Type",
        ["Bollinger Bands", "Monte Carlo Simulation", "Support/Resistance Levels"],
        index=0
    )
    
    # Prediction parameters
    st.subheader("Prediction Parameters")
    
    if pred_type == "Bollinger Bands":
        col1, col2 = st.columns(2)
        
        with col1:
            window = st.slider(
                "Window Size",
                min_value=5,
                max_value=50,
                value=20,
                step=1,
                help="Window size for moving average"
            )
        
        with col2:
            num_std = st.slider(
                "Standard Deviations",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.1,
                help="Number of standard deviations for bands"
            )
        
        show_volume = st.checkbox("Show Volume", value=True)
    
    elif pred_type == "Monte Carlo Simulation":
        col1, col2 = st.columns(2)
        
        with col1:
            num_simulations = st.slider(
                "Number of Simulations",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of Monte Carlo simulations to run"
            )
        
        with col2:
            days_to_forecast = st.slider(
                "Days to Forecast",
                min_value=5,
                max_value=252,
                value=30,
                step=5,
                help="Number of days to forecast"
            )
    
    elif pred_type == "Support/Resistance Levels":
        auto_detect = st.checkbox("Auto-detect Levels", value=True)
        
        n_levels = st.slider(
            "Number of Levels",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of support/resistance levels to detect"
        )
        
        show_volume = st.checkbox("Show Volume", value=True)
    
    # Run visualization button
    if st.button("Generate Prediction Bands"):
        with st.spinner("Fetching data and generating prediction bands..."):
            # Fetch data
            data = data_fetcher.get_stock_data(ticker, period=period, interval=interval)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Generate the appropriate visualization
            if pred_type == "Bollinger Bands":
                fig = prediction_bands.create_bollinger_bands_chart(
                    data,
                    window=window,
                    num_std=num_std,
                    column='Close',
                    title=f"{ticker} - Bollinger Bands",
                    show_volume=show_volume
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                with st.expander("Bollinger Bands Interpretation"):
                    st.write("""
                    **Bollinger Bands Interpretation:**
                    
                    Bollinger Bands consist of a middle band (simple moving average) and two outer bands at a specified number of standard deviations away from the middle band.
                    
                    **Key Signals:**
                    
                    1. **Squeeze**: When the bands narrow (low volatility), it often precedes a significant price move.
                    
                    2. **Expansion**: When the bands widen (high volatility), it suggests a continuation of the current trend.
                    
                    3. **Price at Upper Band**: When price touches or exceeds the upper band, it might indicate:
                       - In an uptrend: Strong trend continuation
                       - In a ranging market: Potential reversal or pullback
                    
                    4. **Price at Lower Band**: When price touches or drops below the lower band, it might indicate:
                       - In a downtrend: Strong trend continuation
                       - In a ranging market: Potential reversal or bounce
                    
                    5. **W-Bottom**: A W pattern where the second low is higher than the first and stays above the lower band, often signals a strong bullish reversal.
                    
                    6. **M-Top**: An M pattern where the second high is lower than the first and stays below the upper band, often signals a bearish reversal.
                    
                    **Statistical Context:**
                    
                    The bands encompass a specific percentage of price action based on standard deviations:
                    - 1σ bands: ~68% of price action
                    - 2σ bands: ~95% of price action
                    - 3σ bands: ~99.7% of price action
                    
                    Prices moving outside the bands are increasingly rare statistical events that may signal unsustainable extremes.
                    """)
            
            elif pred_type == "Monte Carlo Simulation":
                fig = prediction_bands.create_monte_carlo_simulation_chart(
                    data,
                    num_simulations=num_simulations,
                    days_to_forecast=days_to_forecast,
                    percentiles=[5, 25, 50, 75, 95],
                    column='Close',
                    title=f"{ticker} - Monte Carlo Price Simulation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                with st.expander("Monte Carlo Simulation Interpretation"):
                    st.write("""
                    **Monte Carlo Simulation Interpretation:**
                    
                    Monte Carlo simulation generates multiple possible future price paths based on the historical volatility and returns of the stock.
                    
                    **Key Components:**
                    
                    1. **Percentile Lines**: 
                       - The 50th percentile represents the median forecast
                       - The 25th/75th percentiles represent a likely range (50% of scenarios)
                       - The 5th/95th percentiles represent a wider range (90% of scenarios)
                    
                    2. **Simulation Paths**: The light gray lines show individual simulation paths, providing a visual sense of possible scenarios.
                    
                    **How to Interpret:**
                    
                    - **Wide Dispersion**: Greater uncertainty in future price movements, often seen in highly volatile stocks
                    - **Narrow Dispersion**: Lower uncertainty, typical of more stable stocks
                    - **Upward Median Trend**: Overall bullish outlook based on historical patterns
                    - **Downward Median Trend**: Overall bearish outlook based on historical patterns
                    
                    **Important Caveats:**
                    
                    - Simulations are based solely on historical price behavior and volatility
                    - They don't account for fundamental changes, news events, or market regime shifts
                    - The further into the future, the less reliable the projections
                    - Actual prices frequently move outside even the 95% confidence bands due to unforeseen events
                    
                    These simulations are best used for risk assessment rather than making specific price predictions.
                    """)
                
                # Calculate key metrics
                latest_price = data['Close'].iloc[-1]
                
                # Create a forecast summary
                st.subheader("Forecast Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${latest_price:.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="Forecast Horizon",
                        value=f"{days_to_forecast} days"
                    )
                
                with col3:
                    st.metric(
                        label="Simulations",
                        value=f"{num_simulations}"
                    )
            
            elif pred_type == "Support/Resistance Levels":
                fig = prediction_bands.create_support_resistance_chart(
                    data,
                    auto_detect=auto_detect,
                    n_levels=n_levels,
                    title=f"{ticker} - Support and Resistance Levels",
                    show_volume=show_volume
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                with st.expander("Support and Resistance Interpretation"):
                    st.write("""
                    **Support and Resistance Levels Interpretation:**
                    
                    Support and resistance levels represent price areas where a stock has historically had difficulty moving beyond, either to the downside (support) or upside (resistance).
                    
                    **Key Concepts:**
                    
                    1. **Support Levels** (Green):
                       - Price areas where buying interest has previously overcome selling pressure
                       - Often form at prior lows or round numbers
                       - When price approaches support, it may bounce upward
                       - If support breaks, it often becomes resistance
                    
                    2. **Resistance Levels** (Red):
                       - Price areas where selling pressure has previously overcome buying interest
                       - Often form at prior highs or round numbers
                       - When price approaches resistance, it may reverse downward
                       - If resistance breaks, it often becomes support
                    
                    **Trading Applications:**
                    
                    - **Range Trading**: Buy near support, sell near resistance
                    - **Breakout Trading**: Enter when price convincingly breaks through resistance
                    - **Breakdown Trading**: Enter when price convincingly breaks through support
                    - **Stop Loss Placement**: Place stops just beyond key support or resistance levels
                    
                    **Strength Factors:**
                    
                    - **Multiple Tests**: The more times a level has been tested, the stronger it becomes
                    - **Time Frame**: Levels from higher time frames tend to be stronger
                    - **Volume**: Higher volume at a level indicates stronger support/resistance
                    - **Recency**: More recent levels tend to be more relevant than older ones
                    
                    Levels shown in the chart are auto-detected based on price clusters and are sorted by proximity to the current price.
                    """)

def run_enhanced_reporting():
    """Run the enhanced report generation"""
    
    st.header("Enhanced Report Generation")
    st.write("""
    This feature provides enhanced report generation capabilities for stock analysis.
    """)
    
    # Add help expander
    with st.expander("ℹ️ How to use Enhanced Report Generation"):
        st.markdown("""
        ### Enhanced Report Generation Help
        
        This feature provides advanced reporting capabilities for stock analysis:
        
        1. **Interactive Analysis Report**: Generate a comprehensive interactive dashboard for a single stock with technical indicators and chart pattern analysis.
        
        2. **Multi-Stock Comparison Report**: Compare performance and correlations between multiple stocks.
        
        3. **Snapshot Comparison Report**: Analyze differences between market snapshots from different time periods.
        
        **Tips:**
        - For Interactive Analysis, select technical indicators that complement each other
        - For Multi-Stock Comparison, include benchmarks like SPY for better context
        - For Snapshot Analysis, ensure your snapshot files have consistent formatting
        
        [View Full User Guide](docs/user_guide.md)
        """)
    
    # Select report type
    report_type = st.radio(
        "Select Report Type",
        ["Interactive Analysis Report", "Multi-Stock Comparison Report", "Snapshot Comparison Report"],
        index=0
    )
    
    if report_type == "Interactive Analysis Report":
        run_interactive_analysis()
    elif report_type == "Multi-Stock Comparison Report":
        run_comparison_report()
    elif report_type == "Snapshot Comparison Report":
        run_snapshot_report()

def run_interactive_analysis():
    """Run the interactive analysis dashboard"""
    
    st.subheader("Interactive Analysis Dashboard")
    st.write("""
    Generate a comprehensive interactive analysis dashboard for a single stock
    including price analysis, technical indicators, and pattern recognition.
    """)
    
    # Select stock
    ticker = stock_selector.display_stock_selector()
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Technical indicators selection
    st.subheader("Technical Indicators to Include")
    
    # Create columns for indicator selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_ma = st.checkbox("Moving Averages", value=True)
        include_bolling = st.checkbox("Bollinger Bands", value=True)
        include_macd = st.checkbox("MACD", value=True)
    
    with col2:
        include_rsi = st.checkbox("RSI", value=True)
        include_stoch = st.checkbox("Stochastic", value=False)
        include_atr = st.checkbox("ATR", value=False)
    
    with col3:
        include_patterns = st.checkbox("Pattern Detection", value=True)
        include_adx = st.checkbox("ADX", value=False)
        include_volume = st.checkbox("Volume Analysis", value=True)
    
    # Run analysis generation button
    if st.button("Generate Interactive Analysis"):
        with st.spinner("Fetching data and generating analysis..."):
            # Fetch data
            data = data_fetcher.get_stock_data(ticker, period=period, interval=interval)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Display price chart
            st.subheader(f"{ticker} Price Chart")
            
            # Create a candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            )])
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} Price",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prepare indicators
            indicators = {}
            
            # Moving Averages
            if include_ma:
                ma_periods = [10, 20, 50, 200]
                for period in ma_periods:
                    data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
                
                ma_data = data[[f'MA_{p}' for p in ma_periods]].dropna().iloc[-5:]
                
                indicators['Moving Averages'] = {
                    'description': f"Moving averages over {ma_periods} periods.",
                    'values': ma_data,
                    'interpretation': interpret_moving_averages(data, ma_periods)
                }
                
                # Display moving averages chart
                st.subheader("Moving Averages")
                
                ma_fig = go.Figure()
                
                # Add price
                ma_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=1)
                ))
                
                # Add MAs
                colors = ['blue', 'green', 'orange', 'red']
                for i, period in enumerate(ma_periods):
                    ma_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[f'MA_{period}'],
                        mode='lines',
                        name=f'{period}-day MA',
                        line=dict(color=colors[i % len(colors)], width=1)
                    ))
                
                ma_fig.update_layout(
                    title="Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=400
                )
                
                st.plotly_chart(ma_fig, use_container_width=True)
                st.write(f"**Interpretation:** {indicators['Moving Averages']['interpretation']}")
            
            # Bollinger Bands
            if include_bolling:
                window = 20
                num_std = 2
                
                data['MA'] = data['Close'].rolling(window=window).mean()
                data['BB_Upper'] = data['MA'] + (data['Close'].rolling(window=window).std() * num_std)
                data['BB_Lower'] = data['MA'] - (data['Close'].rolling(window=window).std() * num_std)
                
                bb_data = data[['MA', 'BB_Upper', 'BB_Lower']].dropna().iloc[-5:]
                
                indicators['Bollinger Bands'] = {
                    'description': f"{window}-day Bollinger Bands with {num_std} standard deviations.",
                    'values': bb_data,
                    'interpretation': interpret_bollinger_bands(data)
                }
                
                # Display Bollinger Bands chart
                st.subheader("Bollinger Bands")
                
                bb_fig = prediction_bands.create_bollinger_bands_chart(
                    data,
                    window=window,
                    num_std=num_std,
                    column='Close',
                    title=f"{ticker} - Bollinger Bands",
                    show_volume=True
                )
                
                st.plotly_chart(bb_fig, use_container_width=True)
                st.write(f"**Interpretation:** {indicators['Bollinger Bands']['interpretation']}")
            
            # MACD
            if include_macd:
                # Calculate MACD
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
                
                macd_data = data[['MACD', 'Signal_Line', 'MACD_Histogram']].dropna().iloc[-5:]
                
                indicators['MACD'] = {
                    'description': "Moving Average Convergence Divergence (12, 26, 9)",
                    'values': macd_data,
                    'interpretation': interpret_macd(data)
                }
                
                # Display MACD chart
                st.subheader("MACD")
                
                macd_fig = go.Figure()
                
                # Add MACD line
                macd_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                ))
                
                # Add Signal line
                macd_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Signal_Line'],
                    mode='lines',
                    name='Signal Line',
                    line=dict(color='red', width=1)
                ))
                
                # Add Histogram
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                
                macd_fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors
                ))
                
                macd_fig.update_layout(
                    title="MACD (12, 26, 9)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=400
                )
                
                st.plotly_chart(macd_fig, use_container_width=True)
                st.write(f"**Interpretation:** {indicators['MACD']['interpretation']}")
            
            # RSI
            if include_rsi:
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                rsi_data = data[['RSI']].dropna().iloc[-5:]
                
                indicators['RSI'] = {
                    'description': "Relative Strength Index (14)",
                    'values': rsi_data,
                    'interpretation': interpret_rsi(data)
                }
                
                # Display RSI chart
                st.subheader("RSI")
                
                rsi_fig = go.Figure()
                
                # Add RSI line
                rsi_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1.5)
                ))
                
                # Add threshold lines
                rsi_fig.add_shape(
                    type='line',
                    x0=data.index[0],
                    x1=data.index[-1],
                    y0=70,
                    y1=70,
                    line=dict(color='red', width=1, dash='dash')
                )
                
                rsi_fig.add_shape(
                    type='line',
                    x0=data.index[0],
                    x1=data.index[-1],
                    y0=30,
                    y1=30,
                    line=dict(color='green', width=1, dash='dash')
                )
                
                rsi_fig.add_shape(
                    type='line',
                    x0=data.index[0],
                    x1=data.index[-1],
                    y0=50,
                    y1=50,
                    line=dict(color='gray', width=1, dash='dot')
                )
                
                rsi_fig.update_layout(
                    title="RSI (14)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    height=400
                )
                
                st.plotly_chart(rsi_fig, use_container_width=True)
                st.write(f"**Interpretation:** {indicators['RSI']['interpretation']}")
            
            # Pattern Detection
            if include_patterns:
                st.subheader("Chart Pattern Analysis")
                
                # Create pattern recognizer configuration
                config = {
                    'smoothing_period': 5,
                    'peak_distance': 15,
                    'threshold_pct': 2.0,
                    'pattern_window': 120
                }
                
                # Run pattern recognition
                with st.spinner("Detecting patterns..."):
                    pattern_results = pattern_recognition.detect_patterns(ticker, data, config)
                
                if 'error' not in pattern_results:
                    primary_pattern = pattern_results.get('primary_pattern', {})
                    pattern_name = primary_pattern.get('pattern', 'none')
                    
                    if pattern_name != 'none':
                        # Display pattern information
                        st.markdown(f"### Detected Pattern: {pattern_name.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {primary_pattern.get('confidence', 0):.2f}")
                        
                        # Display pattern interpretation
                        pattern_interp = interpret_pattern(primary_pattern)
                        st.markdown(f"**Interpretation:** {pattern_interp}")
                        
                        # Display pattern signal
                        signal_interp = interpret_pattern_signal(primary_pattern)
                        signal_color = "green" if "bullish" in signal_interp.lower() else "red" if "bearish" in signal_interp.lower() else "gray"
                        st.markdown(f"**Signal:** <span style='color:{signal_color}'>{signal_interp}</span>", unsafe_allow_html=True)
                        
                        # Display pattern chart
                        start_idx = primary_pattern.get('start_idx', 0)
                        end_idx = primary_pattern.get('end_idx', len(data) - 1)
                        
                        # Show pattern visualization if available
                        pattern_visualization = candlestick_patterns.visualize_pattern(
                            data.iloc[max(0, start_idx-20):min(len(data), end_idx+20)], 
                            pattern_results
                        )
                        
                        if pattern_visualization:
                            st.plotly_chart(pattern_visualization, use_container_width=True)
                            
                        # Trading considerations
                        st.subheader("Trading Considerations")
                        considerations = get_pattern_trading_considerations(primary_pattern)
                        
                        for i, consideration in enumerate(considerations):
                            st.write(f"{i+1}. {consideration}")
                    else:
                        st.info("No significant chart patterns detected in the current timeframe.")
                else:
                    st.error(f"Error in pattern detection: {pattern_results.get('error', 'Unknown error')}")
            
            # Summary
            st.header("Analysis Summary")
            
            # Create summary tables
            for indicator_name, indicator_data in indicators.items():
                with st.expander(f"{indicator_name} Summary"):
                    # Description
                    st.write(indicator_data['description'])
                    
                    # Values table if not empty
                    if not indicator_data['values'].empty:
                        st.dataframe(indicator_data['values'])
                    
                    # Interpretation
                    st.write(f"**Interpretation:** {indicator_data['interpretation']}")
                    
                    # Signal if available
                    if 'signal' in indicator_data:
                        st.write(f"**Signal:** {indicator_data['signal']}")
            
            st.success(f"Interactive analysis for {ticker} generated successfully!")

def run_comparison_report():
    """Run the multi-stock comparison report generation"""
    
    st.subheader("Multi-Stock Comparison Report")
    st.write("""
    Generate a comprehensive comparison report for multiple stocks including
    correlation analysis, relative performance, and risk metrics.
    """)
    
    # Select multiple stocks
    tickers = stock_selector.display_multi_stock_selector(max_stocks=5)
    
    if len(tickers) < 2:
        st.warning("Please select at least 2 stocks for comparison.")
        return
    
    # Select timeframe
    period, interval = timeframe_selector.display_timeframe_selector()
    
    # Analysis options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_correlation = st.checkbox("Correlation Analysis", value=True)
        include_performance = st.checkbox("Performance Metrics", value=True)
    
    with col2:
        include_volatility = st.checkbox("Volatility Analysis", value=True)
        include_beta = st.checkbox("Beta Analysis", value=True)
    
    # Run report generation button
    if st.button("Generate Comparison Report"):
        with st.spinner("Fetching data and generating report..."):
            # Fetch data for all tickers
            data_dict = data_fetcher.get_multiple_stocks_data(tickers, period=period, interval=interval)
            
            # Check if we have valid data
            valid_tickers = [t for t, df in data_dict.items() if not df.empty and 'Close' in df.columns]
            
            if len(valid_tickers) < 2:
                st.error("Could not fetch valid data for at least two tickers")
                return
            
            # Filter data_dict to only include valid tickers
            data_dict = {t: df for t, df in data_dict.items() if t in valid_tickers}
            
            # Prepare analysis results
            analysis_results = {}
            
            # Correlation Analysis
            if include_correlation:
                # Create a new DataFrame with closing prices
                close_df = pd.DataFrame()
                
                for ticker, df in data_dict.items():
                    close_df[ticker] = df['Close']
                
                # Calculate correlation matrix
                correlation_matrix = close_df.corr()
                
                analysis_results['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Performance Metrics
            if include_performance:
                # Calculate performance metrics for each stock
                performance_metrics = []
                
                for ticker, df in data_dict.items():
                    # Calculate metrics
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    price_change = end_price - start_price
                    price_change_pct = (price_change / start_price) * 100
                    
                    # Calculate max drawdown
                    rolling_max = df['Close'].cummax()
                    drawdown = (df['Close'] - rolling_max) / rolling_max
                    max_drawdown = drawdown.min() * 100
                    
                    # Calculate average daily return
                    daily_returns = df['Close'].pct_change().dropna()
                    avg_daily_return = daily_returns.mean() * 100
                    
                    # Calculate volatility
                    volatility = daily_returns.std() * 100 * np.sqrt(252)  # Annualized
                    
                    # Calculate Sharpe ratio (assuming risk-free rate of 1%)
                    risk_free_rate = 0.01
                    daily_risk_free = risk_free_rate / 252
                    excess_return = daily_returns.mean() - daily_risk_free
                    sharpe_ratio = excess_return / daily_returns.std() * np.sqrt(252)
                    
                    # Add to list
                    performance_metrics.append({
                        'Ticker': ticker,
                        'Start Price': start_price,
                        'End Price': end_price,
                        'Price Change': price_change,
                        'Price Change %': price_change_pct,
                        'Max Drawdown %': max_drawdown,
                        'Avg Daily Return %': avg_daily_return,
                        'Volatility %': volatility,
                        'Sharpe Ratio': sharpe_ratio
                    })
                
                analysis_results['performance_metrics'] = performance_metrics
            
            # Risk Metrics
            if include_volatility or include_beta:
                # Calculate risk metrics
                risk_metrics = []
                
                # Try to get market data (S&P 500) for beta calculation
                market_data = None
                if include_beta:
                    try:
                        market_data = data_fetcher.get_stock_data('^GSPC', period=period, interval=interval)
                    except Exception as e:
                        st.warning(f"Could not fetch market data for beta calculation: {str(e)}")
                
                for ticker, df in data_dict.items():
                    # Calculate daily returns
                    daily_returns = df['Close'].pct_change().dropna()
                    
                    # Calculate various volatility measures
                    daily_vol = daily_returns.std() * 100
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    # Calculate upside and downside volatility
                    up_returns = daily_returns[daily_returns > 0]
                    down_returns = daily_returns[daily_returns < 0]
                    
                    up_vol = up_returns.std() * 100 * np.sqrt(252) if not up_returns.empty else 0
                    down_vol = down_returns.std() * 100 * np.sqrt(252) if not down_returns.empty else 0
                    
                    # Calculate beta if market data is available
                    beta = None
                    r_squared = None
                    
                    if include_beta and market_data is not None and not market_data.empty:
                        market_returns = market_data['Close'].pct_change().dropna()
                        
                        # Get overlapping dates
                        common_idx = daily_returns.index.intersection(market_returns.index)
                        
                        if len(common_idx) > 0:
                            stock_returns_aligned = daily_returns.loc[common_idx]
                            market_returns_aligned = market_returns.loc[common_idx]
                            
                            # Calculate beta through linear regression
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                market_returns_aligned, stock_returns_aligned
                            )
                            
                            beta = slope
                            r_squared = r_value ** 2
                    
                    # Add to list
                    risk_metrics.append({
                        'Ticker': ticker,
                        'Daily Volatility %': daily_vol,
                        'Annual Volatility %': annual_vol,
                        'Upside Volatility %': up_vol,
                        'Downside Volatility %': down_vol,
                        'Beta': beta if beta is not None else 'N/A',
                        'R-Squared': r_squared if r_squared is not None else 'N/A'
                    })
                
                analysis_results['risk_metrics'] = risk_metrics
            
            # Generate and save the report
            output_file = os.path.join(OUTPUT_DIR, f"stock_comparison_report.pdf")
            
            report_path = report_generation.generate_comparison_report(
                tickers, data_dict, analysis_results, output_file
            )
            
            if report_path:
                # Create download button
                with open(report_path, "rb") as f:
                    report_bytes = f.read()
                    
                st.download_button(
                    label="Download Stock Comparison Report",
                    data=report_bytes,
                    file_name="stock_comparison_report.pdf",
                    mime="application/pdf"
                )
                
                st.success("Stock comparison report generated successfully!")
            else:
                st.error("Error generating the report.")

def run_snapshot_report():
    """Run the snapshot comparison report generation"""
    
    st.subheader("Snapshot Comparison Report")
    st.write("""
    Generate a comprehensive report comparing two stock market snapshots from different time periods.
    This report analyzes price movements, volume changes, sector performance, and more.
    """)
    
    # Input file selection
    st.info("This report requires two snapshot files (CSV or Excel) from different time periods.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        previous_file = st.file_uploader("Upload previous snapshot file", type=["csv", "xlsx", "xls"])
    
    with col2:
        current_file = st.file_uploader("Upload current snapshot file", type=["csv", "xlsx", "xls"])
    
    if previous_file is None or current_file is None:
        st.warning("Please upload both previous and current snapshot files.")
        return
    
    # Save uploaded files to disk
    previous_path = os.path.join(OUTPUT_DIR, "previous_snapshot." + previous_file.name.split(".")[-1])
    current_path = os.path.join(OUTPUT_DIR, "current_snapshot." + current_file.name.split(".")[-1])
    
    with open(previous_path, "wb") as f:
        f.write(previous_file.getbuffer())
    
    with open(current_path, "wb") as f:
        f.write(current_file.getbuffer())
    
    # Analysis options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_price = st.checkbox("Price Movement Analysis", value=True)
        include_volume = st.checkbox("Volume Analysis", value=True)
    
    with col2:
        include_sector = st.checkbox("Sector Analysis", value=True)
        include_relative = st.checkbox("Relative Strength Analysis", value=True)
    
    # Run report generation button
    if st.button("Generate Snapshot Report"):
        with st.spinner("Analyzing snapshots and generating report..."):
            try:
                # Import the snapshot analyzer
                from compare_snapshots import SnapshotAnalyzer
                
                # Create snapshot analyzer
                analyzer = SnapshotAnalyzer(previous_path, current_path, output_dir=OUTPUT_DIR)
                
                # Run analyses
                if include_price:
                    analyzer.analyze_price_movement()
                
                if include_volume:
                    analyzer.analyze_volume_spikes()
                
                if include_sector:
                    analyzer.analyze_sectors()
                
                if include_relative:
                    analyzer.analyze_relative_strength()
                
                # Generate summary report
                result = analyzer.generate_summary_report()
                
                # Generate and save HTML report only
                html_report = os.path.join(OUTPUT_DIR, "summary_report.html")
                
                if os.path.exists(html_report):
                    # Create download button for HTML report
                    with open(html_report, "rb") as f:
                        html_bytes = f.read()
                        
                    st.download_button(
                        label="Download Snapshot Comparison HTML Report",
                        data=html_bytes,
                        file_name="snapshot_comparison_report.html",
                        mime="text/html"
                    )
                    
                    st.success("Snapshot comparison report generated successfully!")
                    
                    # Also provide Excel report option
                    st.write("### Excel Report")
                    st.write("Download an Excel report with all analysis data and charts.")
                    
                    # Create Excel report
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Write filtered dataset
                        analyzer.filtered_df.to_excel(writer, sheet_name="Filtered Data")
                        
                        # Write summary statistics
                        summary_df = pd.DataFrame({
                            "Metric": ["Previous Snapshot Date", "Current Snapshot Date", 
                                      "Total Stocks in Previous", "Total Stocks in Current",
                                      "Stocks in Both Snapshots", "Filtered Stocks"],
                            "Value": [
                                analyzer.previous_df['Date'].iloc[0] if 'Date' in analyzer.previous_df.columns else "N/A",
                                analyzer.current_df['Date'].iloc[0] if 'Date' in analyzer.current_df.columns else "N/A",
                                len(analyzer.previous_df),
                                len(analyzer.current_df),
                                len(analyzer.merged_df),
                                len(analyzer.filtered_df)
                            ]
                        })
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Download button for Excel report
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name="snapshot_comparison_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Error generating the report: {str(e)}")

def interpret_moving_averages(data, periods):
    """Interpret moving average signals"""
    # Get the latest prices and MAs
    latest = data.iloc[-1]
    close = latest['Close']
    
    # Get the MAs
    mas = [latest.get(f'MA_{p}', None) for p in periods]
    mas = [ma for ma in mas if ma is not None]
    
    if not mas:
        return "No moving average data available."
    
    # Check if price is above or below MAs
    above_count = sum(1 for ma in mas if close > ma)
    below_count = sum(1 for ma in mas if close < ma)
    
    if above_count == len(mas):
        return "Price is above all moving averages, suggesting a strong bullish trend."
    elif below_count == len(mas):
        return "Price is below all moving averages, suggesting a strong bearish trend."
    else:
        # Check for crossovers
        short_periods = [p for p in periods if p <= 50]
        long_periods = [p for p in periods if p > 50]
        
        if short_periods and long_periods:
            short_mas = [latest.get(f'MA_{p}', None) for p in short_periods]
            short_mas = [ma for ma in short_mas if ma is not None]
            
            long_mas = [latest.get(f'MA_{p}', None) for p in long_periods]
            long_mas = [ma for ma in long_mas if ma is not None]
            
            if short_mas and long_mas:
                avg_short = sum(short_mas) / len(short_mas)
                avg_long = sum(long_mas) / len(long_mas)
                
                if avg_short > avg_long:
                    return "Short-term moving averages above long-term moving averages, suggesting a potential bullish trend."
                else:
                    return "Short-term moving averages below long-term moving averages, suggesting a potential bearish trend."
        
        return "Mixed moving average signals, suggesting a neutral or consolidating market."

def interpret_bollinger_bands(data):
    """Interpret Bollinger Bands signals"""
    if 'MA' not in data.columns or 'BB_Upper' not in data.columns or 'BB_Lower' not in data.columns:
        return "No Bollinger Bands data available."
    
    # Get the last 5 days of data
    recent = data.tail(5).copy()
    
    # Calculate width of the bands
    recent['BB_Width'] = (recent['BB_Upper'] - recent['BB_Lower']) / recent['MA']
    
    # Check for squeeze
    is_squeeze = recent['BB_Width'].iloc[-1] < recent['BB_Width'].iloc[:-1].mean() * 0.9
    
    # Check if price is near upper or lower band
    latest = recent.iloc[-1]
    close = latest['Close']
    upper = latest['BB_Upper']
    lower = latest['BB_Lower']
    middle = latest['MA']
    
    upper_dist = (upper - close) / close * 100
    lower_dist = (close - lower) / close * 100
    
    if is_squeeze:
        return "Bollinger Bands are contracting, indicating decreasing volatility and potential upcoming significant price movement."
    elif upper_dist < 1:
        return "Price is near the upper Bollinger Band, indicating overbought conditions or strong uptrend."
    elif lower_dist < 1:
        return "Price is near the lower Bollinger Band, indicating oversold conditions or strong downtrend."
    elif close > middle:
        return "Price is above the middle band but below the upper band, suggesting a moderate bullish trend."
    elif close < middle:
        return "Price is below the middle band but above the lower band, suggesting a moderate bearish trend."
    else:
        return "Price is near the middle band, suggesting a neutral trend."

def interpret_macd(data):
    """Interpret MACD signals"""
    if 'MACD' not in data.columns or 'Signal_Line' not in data.columns:
        return "No MACD data available."
    
    # Get the latest values
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else None
    
    macd = latest['MACD']
    signal = latest['Signal_Line']
    
    if previous is not None:
        prev_macd = previous['MACD']
        prev_signal = previous['Signal_Line']
        
        # Check for crossovers
        if macd > signal and prev_macd <= prev_signal:
            return "MACD crossed above the signal line, generating a bullish signal."
        elif macd < signal and prev_macd >= prev_signal:
            return "MACD crossed below the signal line, generating a bearish signal."
    
    # Check current relationship
    if macd > signal:
        return "MACD is above the signal line, indicating bullish momentum."
    elif macd < signal:
        return "MACD is below the signal line, indicating bearish momentum."
    else:
        return "MACD and signal line are at similar levels, indicating neutral momentum."

def interpret_rsi(data):
    """Interpret RSI signals"""
    if 'RSI' not in data.columns:
        return "No RSI data available."
    
    # Get the latest value
    latest_rsi = data['RSI'].iloc[-1]
    
    if latest_rsi > 70:
        return "RSI is above 70, indicating overbought conditions and potential reversal or pullback."
    elif latest_rsi < 30:
        return "RSI is below 30, indicating oversold conditions and potential reversal or bounce."
    elif latest_rsi > 50:
        return "RSI is between 50 and 70, suggesting moderate bullish momentum."
    else:
        return "RSI is between 30 and 50, suggesting moderate bearish momentum."

def interpret_pattern(pattern_result):
    """Interpret detected patterns"""
    if not pattern_result or 'pattern' not in pattern_result:
        return "No pattern detected."
    
    pattern_name = pattern_result.get('pattern', '')
    
    if 'head_and_shoulders' in pattern_name:
        is_inverse = 'inverse' in pattern_name
        if is_inverse:
            return "Inverse Head and Shoulders pattern detected, suggesting a potential bullish reversal. Look for confirmation with a break above the neckline on increased volume."
        else:
            return "Head and Shoulders pattern detected, suggesting a potential bearish reversal. Look for confirmation with a break below the neckline on increased volume."
    
    elif 'double_top' in pattern_name:
        return "Double Top pattern detected, suggesting a potential bearish reversal. Look for confirmation with a break below the trough between the tops on increased volume."
    
    elif 'double_bottom' in pattern_name:
        return "Double Bottom pattern detected, suggesting a potential bullish reversal. Look for confirmation with a break above the peak between the bottoms on increased volume."
    
    elif 'triangle' in pattern_name:
        if 'ascending' in pattern_name:
            return "Ascending Triangle pattern detected, suggesting a potential bullish continuation. Look for confirmation with a break above the horizontal resistance on increased volume."
        elif 'descending' in pattern_name:
            return "Descending Triangle pattern detected, suggesting a potential bearish continuation. Look for confirmation with a break below the horizontal support on increased volume."
        else:
            return "Symmetrical Triangle pattern detected, suggesting a potential continuation in the direction of the breakout. Watch for increased volume at the breakout point."
    
    elif 'cup_and_handle' in pattern_name:
        return "Cup and Handle pattern detected, suggesting a potential bullish continuation. Look for confirmation with a break above the resistance level formed by the cup rim."
    
    elif 'flag' in pattern_name:
        if 'bullish' in pattern_name:
            return "Bullish Flag pattern detected, suggesting a potential bullish continuation. Look for confirmation with a break above the upper trendline of the flag on increased volume."
        else:
            return "Bearish Flag pattern detected, suggesting a potential bearish continuation. Look for confirmation with a break below the lower trendline of the flag on increased volume."
    
    elif 'pennant' in pattern_name:
        if 'bullish' in pattern_name:
            return "Bullish Pennant pattern detected, suggesting a potential bullish continuation. Look for confirmation with a break above the upper trendline of the pennant on increased volume."
        else:
            return "Bearish Pennant pattern detected, suggesting a potential bearish continuation. Look for confirmation with a break below the lower trendline of the pennant on increased volume."
    
    elif 'wedge' in pattern_name:
        if 'rising' in pattern_name:
            return "Rising Wedge pattern detected, suggesting a potential bearish reversal or continuation. Look for confirmation with a break below the lower trendline on increased volume."
        else:
            return "Falling Wedge pattern detected, suggesting a potential bullish reversal or continuation. Look for confirmation with a break above the upper trendline on increased volume."
    
    else:
        return f"Pattern detected: {pattern_name.replace('_', ' ').title()}. Refer to technical analysis literature for interpretation."

def interpret_pattern_signal(pattern_result):
    """Generate trading signal based on pattern"""
    if not pattern_result or 'pattern' not in pattern_result:
        return "No signal"
    
    pattern_name = pattern_result.get('pattern', '')
    
    # Bullish patterns
    if any(bullish in pattern_name for bullish in ['inverse_head_and_shoulders', 'double_bottom', 'bullish_flag', 'bullish_pennant', 'cup_and_handle', 'falling_wedge']):
        return "Bullish"
    
    # Bearish patterns
    if any(bearish in pattern_name for bearish in ['head_and_shoulders', 'double_top', 'bearish_flag', 'bearish_pennant', 'rising_wedge']):
        return "Bearish"
    
    # Patterns that depend on trend direction
    if 'ascending_triangle' in pattern_name:
        return "Likely Bullish"
    
    if 'descending_triangle' in pattern_name:
        return "Likely Bearish"
    
    if 'symmetrical_triangle' in pattern_name:
        return "Neutral - Watch Breakout Direction"
    
    return "Neutral"

def check_api_keys():
    """Check if API keys are available for AI features"""
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if anthropic_key or openai_key:
        return True
    
    # If no keys are present, ask for them
    st.warning("AI features require API keys. You can add them as environment variables.")
    
    # Add input fields for API keys
    with st.expander("Add API Keys"):
        anthropic_input = st.text_input("Anthropic API Key", type="password")
        openai_input = st.text_input("OpenAI API Key", type="password")
        
        if st.button("Save API Keys"):
            if anthropic_input:
                os.environ['ANTHROPIC_API_KEY'] = anthropic_input
            
            if openai_input:
                os.environ['OPENAI_API_KEY'] = openai_input
            
            if anthropic_input or openai_input:
                st.success("API keys saved for this session.")
                return True
    
    return False

def display_pattern_interpretation(pattern_result):
    """Display pattern interpretation"""
    if not pattern_result or pattern_result.get('pattern', 'none') == 'none':
        return
    
    pattern_name = pattern_result.get('pattern', '').replace('_', ' ').title()
    
    st.subheader("Pattern Interpretation")
    
    # Display pattern image if available
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display a generic pattern image
        pattern_image = f"pattern_{pattern_result.get('pattern', 'generic')}.png"
        
        # Use a placeholder text when image is not available
        st.write(f"**{pattern_name}**")
        
        # Add direction indicator
        if any(bullish in pattern_result.get('pattern', '') for bullish in 
               ['inverse_head_and_shoulders', 'double_bottom', 'bullish', 'cup_and_handle', 'falling_wedge']):
            st.markdown("📈 **Bullish Pattern**")
        elif any(bearish in pattern_result.get('pattern', '') for bearish in 
                ['head_and_shoulders', 'double_top', 'bearish', 'rising_wedge']):
            st.markdown("📉 **Bearish Pattern**")
        elif 'ascending_triangle' in pattern_result.get('pattern', ''):
            st.markdown("📈 **Likely Bullish Pattern**")
        elif 'descending_triangle' in pattern_result.get('pattern', ''):
            st.markdown("📉 **Likely Bearish Pattern**")
        else:
            st.markdown("↔️ **Neutral Pattern**")
    
    with col2:
        st.write(interpret_pattern(pattern_result))
        
        # Add trading considerations
        st.subheader("Trading Considerations")
        
        considerations = get_pattern_trading_considerations(pattern_result)
        
        for i, consideration in enumerate(considerations):
            st.write(f"{i+1}. {consideration}")

def get_pattern_trading_considerations(pattern_result):
    """Get trading considerations for a pattern"""
    if not pattern_result or 'pattern' not in pattern_result:
        return []
    
    pattern_name = pattern_result.get('pattern', '')
    
    # Common considerations
    common = [
        "Confirm the pattern with additional indicators or signals.",
        "Consider the overall market trend when interpreting this pattern.",
        "Check volume for confirmation of pattern validity."
    ]
    
    # Pattern-specific considerations
    specific = []
    
    if 'head_and_shoulders' in pattern_name:
        is_inverse = 'inverse' in pattern_name
        if is_inverse:
            specific = [
                "Look for a breakout above the neckline for trade entry.",
                "Set targets based on the pattern height (head to neckline).",
                "Place a stop loss below the right shoulder."
            ]
        else:
            specific = [
                "Look for a breakdown below the neckline for trade entry.",
                "Set targets based on the pattern height (head to neckline).",
                "Place a stop loss above the right shoulder."
            ]
    
    elif 'double_top' in pattern_name:
        specific = [
            "Look for a breakdown below the middle trough for trade entry.",
            "Set targets based on the pattern height (top to trough).",
            "Place a stop loss above the second peak."
        ]
    
    elif 'double_bottom' in pattern_name:
        specific = [
            "Look for a breakout above the middle peak for trade entry.",
            "Set targets based on the pattern height (bottom to peak).",
            "Place a stop loss below the second trough."
        ]
    
    elif 'triangle' in pattern_name:
        if 'ascending' in pattern_name:
            specific = [
                "Look for a breakout above the horizontal resistance line.",
                "Set targets based on the pattern height at the widest point.",
                "Place a stop loss below the most recent higher low."
            ]
        elif 'descending' in pattern_name:
            specific = [
                "Look for a breakdown below the horizontal support line.",
                "Set targets based on the pattern height at the widest point.",
                "Place a stop loss above the most recent lower high."
            ]
        else:
            specific = [
                "Wait for a breakout from either the upper or lower trendline.",
                "Trade in the direction of the breakout.",
                "Set targets based on the pattern height at the widest point."
            ]
    
    elif 'cup_and_handle' in pattern_name:
        specific = [
            "Look for a breakout above the cup rim resistance.",
            "Set targets based on the cup depth.",
            "Place a stop loss below the handle low."
        ]
    
    elif 'flag' in pattern_name or 'pennant' in pattern_name:
        if 'bullish' in pattern_name:
            specific = [
                "Look for a breakout above the upper trendline of the flag/pennant.",
                "Set targets based on the flagpole height.",
                "Place a stop loss below the flag/pennant low."
            ]
        else:
            specific = [
                "Look for a breakdown below the lower trendline of the flag/pennant.",
                "Set targets based on the flagpole height.",
                "Place a stop loss above the flag/pennant high."
            ]
    
    elif 'wedge' in pattern_name:
        if 'rising' in pattern_name:
            specific = [
                "Look for a breakdown below the lower trendline of the wedge.",
                "Set targets based on the wedge height at the widest point.",
                "Place a stop loss above the most recent swing high."
            ]
        else:
            specific = [
                "Look for a breakout above the upper trendline of the wedge.",
                "Set targets based on the wedge height at the widest point.",
                "Place a stop loss below the most recent swing low."
            ]
    
    return specific + common

if __name__ == "__main__":
    main()