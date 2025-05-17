import streamlit as st

def display_indicator_selector():
    """
    Display a multi-select widget for technical indicators.
    
    Returns:
    --------
    list
        List of selected indicator names
    """
    st.subheader("Technical Indicators")
    
    # Group indicators by category
    indicator_categories = {
        "Momentum Indicators": [
            "RSI", "MACD", "Stochastic Oscillator", "Williams %R", "ROC", "CCI"
        ],
        "Volatility Indicators": [
            "Bollinger Bands", "ATR", "Standard Deviation"
        ],
        "Volume Indicators": [
            "OBV", "Volume", "Money Flow Index"
        ],
        "Trend Indicators": [
            "Moving Averages", "ADX", "Parabolic SAR", "Ichimoku Cloud"
        ]
    }
    
    # Create expanders for each category
    selected_indicators = []
    
    for category, indicators in indicator_categories.items():
        with st.expander(category, expanded=True if category == "Momentum Indicators" else False):
            # Use checkbox for each indicator
            for indicator in indicators:
                if st.checkbox(indicator, value=indicator in ["RSI", "MACD"]):
                    selected_indicators.append(indicator)
    
    # Information about the selected indicators
    if selected_indicators:
        st.success(f"Selected {len(selected_indicators)} indicators")
    else:
        st.info("No indicators selected. Default charts will be shown.")
    
    return selected_indicators

def display_indicator_info(indicator):
    """
    Display information about a specific technical indicator.
    
    Parameters:
    -----------
    indicator : str
        Name of the indicator to display information for
    """
    indicator_info = {
        "RSI": {
            "full_name": "Relative Strength Index",
            "description": "Measures the magnitude of recent price changes to evaluate overbought or oversold conditions. RSI values range from 0 to 100, with values above 70 indicating overbought conditions and values below 30 indicating oversold conditions.",
            "parameters": ["Period (default: 14)"],
            "interpretation": "RSI values above 70 suggest an asset may be overbought and due for a correction. Values below 30 suggest an asset may be oversold and due for a recovery. Divergences between RSI and price can indicate potential reversals."
        },
        "MACD": {
            "full_name": "Moving Average Convergence Divergence",
            "description": "Shows the relationship between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA. The result is the MACD line. A 9-period EMA of the MACD (the 'signal line') is then plotted on top of the MACD line.",
            "parameters": ["Fast Period (default: 12)", "Slow Period (default: 26)", "Signal Period (default: 9)"],
            "interpretation": "When the MACD crosses above the signal line, it indicates a bullish signal. When it crosses below, it's a bearish signal. The histogram shows the difference between the MACD and the signal line, with increasing bars suggesting strengthening momentum."
        },
        "Bollinger Bands": {
            "full_name": "Bollinger Bands",
            "description": "Consists of a middle band (simple moving average) with upper and lower bands that are standard deviations away from the middle band. Bollinger Bands adjust to volatility by widening during volatile periods and narrowing during less volatile periods.",
            "parameters": ["Period (default: 20)", "Standard Deviation (default: 2)"],
            "interpretation": "Prices tend to stay within the bands. A price touching the upper band may indicate overbought conditions, while touching the lower band may indicate oversold conditions. Band squeezes (narrowing) often precede significant price movements."
        },
        "Stochastic Oscillator": {
            "full_name": "Stochastic Oscillator",
            "description": "Compares a security's closing price to its price range over a specific period. It consists of two lines: %K (fast) and %D (slow). The oscillator ranges from 0 to 100.",
            "parameters": ["K Period (default: 14)", "D Period (default: 3)", "Slowing Period (default: 3)"],
            "interpretation": "Values above 80 indicate overbought conditions, while values below 20 indicate oversold conditions. A bullish signal occurs when %K crosses above %D, and a bearish signal when %K crosses below %D."
        },
        "ATR": {
            "full_name": "Average True Range",
            "description": "Measures market volatility by decomposing the entire range of an asset price for a specific period. ATR doesn't provide directional indication, only volatility.",
            "parameters": ["Period (default: 14)"],
            "interpretation": "Higher ATR values indicate higher volatility, while lower values indicate lower volatility. ATR is often used to set stop-loss levels or to determine position sizing in trading systems."
        }
    }
    
    # Display the information
    if indicator in indicator_info:
        info = indicator_info[indicator]
        
        st.subheader(info["full_name"])
        st.write(info["description"])
        
        st.write("**Parameters:**")
        for param in info["parameters"]:
            st.write(f"- {param}")
        
        st.write("**Interpretation:**")
        st.write(info["interpretation"])
    else:
        st.write("Detailed information for this indicator is not available.")
