import streamlit as st
from datetime import datetime, timedelta

def display_timeframe_selector():
    """
    Display timeframe selection widgets for period and interval.
    
    Returns:
    --------
    tuple
        (period, interval) selected by the user
    """
    # Options for the period and interval
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "Year to Date": "ytd",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y",
        "Max": "max"
    }
    
    interval_options = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Week": "1wk",
        "1 Month": "1mo",
        "3 Months": "3mo"
    }
    
    # Guide for which intervals are valid for which periods
    valid_intervals = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"],
        "1mo": ["2m", "5m", "15m", "30m", "60m", "1h", "1d"],
        "3mo": ["15m", "30m", "60m", "1h", "1d"],
        "6mo": ["30m", "60m", "1h", "1d"],
        "ytd": ["60m", "1h", "1d"],
        "1y": ["1d", "5d", "1wk"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo", "3mo"],
        "10y": ["1d", "5d", "1wk", "1mo", "3mo"],
        "max": ["1d", "5d", "1wk", "1mo", "3mo"]
    }
    
    # Default selections
    default_period = "1 Year"
    
    # Display the selectors in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time Period")
        period_key = st.selectbox(
            "Select period",
            list(period_options.keys()),
            index=list(period_options.keys()).index(default_period)
        )
    
    # Get the selected period value
    period_value = period_options[period_key]
    
    # Determine valid intervals for the selected period
    valid_interval_values = valid_intervals.get(period_value, ["1d"])
    valid_interval_keys = [k for k, v in interval_options.items() if v in valid_interval_values]
    
    # Default to daily if available, otherwise use the first valid option
    default_interval_key = "1 Day" if "1 Day" in valid_interval_keys else valid_interval_keys[0]
    
    with col2:
        st.subheader("Interval")
        interval_key = st.selectbox(
            "Select interval",
            valid_interval_keys,
            index=valid_interval_keys.index(default_interval_key)
        )
    
    # Get the selected interval value
    interval_value = interval_options[interval_key]
    
    # Add an info message about data limitations
    if interval_value in ["1m", "2m", "5m", "15m", "30m"]:
        st.info("Note: Minute-level data is only available for the last 7 days. For longer periods, consider using hourly or daily intervals.")
    
    return period_value, interval_value

def display_custom_date_selector():
    """
    Display a custom date range selector.
    
    Returns:
    --------
    tuple
        (start_date, end_date) selected by the user
    """
    st.subheader("Custom Date Range")
    
    # Default end date is today
    end_date = datetime.now().date()
    
    # Default start date is one year ago
    start_date = end_date - timedelta(days=365)
    
    # Date selectors
    start_date = st.date_input(
        "Start Date",
        value=start_date,
        max_value=end_date
    )
    
    end_date = st.date_input(
        "End Date",
        value=end_date,
        min_value=start_date,
        max_value=datetime.now().date()
    )
    
    # Warning for large date ranges
    date_diff = (end_date - start_date).days
    if date_diff > 365 * 2:
        st.warning("Very large date ranges may result in slower performance or incomplete data.")
    
    return start_date, end_date
