import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

from utils.options_utils import get_options_chain, calculate_greeks, options_strategy_analysis
from components.stock_selector import display_stock_selector

st.set_page_config(
    page_title="Options Analysis - Stock Analysis Platform",
    page_icon="🧮",
    layout="wide"
)

def main():
    st.title("🧮 Options Analysis")
    
    st.markdown("""
    Analyze stock options with detailed options chain data, Greeks calculations, and strategy visualization.
    This tool helps you understand the complex world of options trading and make more informed decisions.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Options Analysis Settings")
        
        # Stock selection
        ticker = display_stock_selector()
        
        if ticker:
            # Get stock info and available expiration dates
            try:
                stock = yf.Ticker(ticker)
                expirations = stock.options
                
                if not expirations:
                    st.error(f"No options data available for {ticker}")
                    expiration_date = None
                else:
                    # Expiration date selection
                    expiration_date = st.selectbox(
                        "Select expiration date",
                        expirations
                    )
            except Exception as e:
                st.error(f"Error retrieving options data: {e}")
                expiration_date = None
        else:
            expiration_date = None
        
        # Analysis type
        analysis_type = st.radio(
            "Select analysis type",
            ["Options Chain", "Greeks Analysis", "Strategy Analysis"],
            index=0
        )
        
        # Analyze button
        analyze_button = st.button("Analyze Options", type="primary")
    
    # Main content
    if ticker and expiration_date and analyze_button:
        st.header(f"Options Analysis for {ticker} - {expiration_date}")
        
        try:
            with st.spinner(f"Fetching options data for {ticker}..."):
                # Get current stock price and options data
                stock = yf.Ticker(ticker)
                current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))
                
                # Display stock info
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.metric(
                        label=f"{ticker} - Current Price",
                        value=f"${current_price:.2f}"
                    )
                
                with info_col2:
                    st.metric(
                        label="Expiration Date",
                        value=expiration_date
                    )
                
                with info_col3:
                    # Calculate days to expiration
                    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
                    today = datetime.now()
                    days_to_exp = (exp_date - today).days
                    
                    st.metric(
                        label="Days to Expiration",
                        value=days_to_exp
                    )
                
                # Get options chain
                calls, puts = get_options_chain(ticker, expiration_date)
                
                if analysis_type == "Options Chain":
                    # Create tabs for calls and puts
                    tab1, tab2 = st.tabs(["Calls", "Puts"])
                    
                    with tab1:
                        st.subheader("Calls Options Chain")
                        
                        if calls is not None and not calls.empty:
                            # Add in-the-money indicator
                            calls['ITM'] = calls['strike'] < current_price
                            
                            # Sort by strike price
                            calls = calls.sort_values('strike')
                            
                            # Display calls table
                            st.dataframe(calls, use_container_width=True)
                            
                            # Volume and open interest visualization
                            st.subheader("Calls - Volume and Open Interest")
                            
                            fig = go.Figure()
                            
                            # Create hover text with detailed information for volume bars
                            volume_hover = [
                                f"Strike: ${strike:.2f}<br>" +
                                f"Volume: {volume:,}<br>" +
                                f"Bid: ${bid:.2f}<br>" +
                                f"Ask: ${ask:.2f}<br>" +
                                f"Last: ${last:.2f}<br>" +
                                f"Implied Vol: {impliedVolatility:.1%}" 
                                for strike, volume, bid, ask, last, impliedVolatility in zip(
                                    calls['strike'],
                                    calls['volume'],
                                    calls['bid'],
                                    calls['ask'],
                                    calls['lastPrice'],
                                    calls['impliedVolatility']
                                )
                            ]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=calls['strike'],
                                    y=calls['volume'],
                                    name="Volume",
                                    marker_color='rgba(50, 171, 96, 0.7)',
                                    hoverinfo='text',
                                    hovertext=volume_hover
                                )
                            )
                            
                            # Check if Greek values are available in the calls data
                            has_greeks = all(column in calls.columns for column in ['delta', 'gamma', 'theta', 'vega'])
                            
                            if has_greeks:
                                # Create hover text with Greeks for open interest line
                                oi_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Open Interest: {oi:,}<br>" +
                                    f"Delta: {delta:.3f}<br>" +
                                    f"Gamma: {gamma:.4f}<br>" +
                                    f"Theta: {theta:.4f}<br>" +
                                    f"Vega: {vega:.4f}"
                                    for strike, oi, delta, gamma, theta, vega in zip(
                                        calls['strike'],
                                        calls['openInterest'],
                                        calls['delta'],
                                        calls['gamma'],
                                        calls['theta'],
                                        calls['vega']
                                    )
                                ]
                            else:
                                # Create simple hover text without Greeks
                                oi_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Open Interest: {oi:,}"
                                    for strike, oi in zip(
                                        calls['strike'],
                                        calls['openInterest']
                                    )
                                ]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['strike'],
                                    y=calls['openInterest'],
                                    name="Open Interest",
                                    marker_color='rgba(250, 128, 114, 1)',
                                    hoverinfo='text',
                                    hovertext=oi_hover
                                )
                            )
                            
                            # Add line for current price
                            fig.add_vline(
                                x=current_price,
                                line_width=2,
                                line_dash="dash",
                                line_color="yellow",
                                annotation_text=f"Current: ${current_price:.2f}",
                                annotation_position="top right"
                            )
                            
                            fig.update_layout(
                                title="Calls - Volume and Open Interest by Strike Price",
                                xaxis_title="Strike Price ($)",
                                yaxis_title="Contracts",
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
                        else:
                            st.info("No calls data available for the selected expiration date.")
                    
                    with tab2:
                        st.subheader("Puts Options Chain")
                        
                        if puts is not None and not puts.empty:
                            # Add in-the-money indicator
                            puts['ITM'] = puts['strike'] > current_price
                            
                            # Sort by strike price
                            puts = puts.sort_values('strike')
                            
                            # Display puts table
                            st.dataframe(puts, use_container_width=True)
                            
                            # Volume and open interest visualization
                            st.subheader("Puts - Volume and Open Interest")
                            
                            fig = go.Figure()
                            
                            # Create hover text with detailed information for volume bars
                            # Make sure all required columns exist
                            volume_columns = ['strike', 'volume', 'bid', 'ask', 'lastPrice', 'impliedVolatility']
                            has_all_columns = all(column in puts.columns for column in volume_columns)
                            
                            if has_all_columns:
                                volume_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Volume: {volume:,}<br>" +
                                    f"Bid: ${bid:.2f}<br>" +
                                    f"Ask: ${ask:.2f}<br>" +
                                    f"Last: ${last:.2f}<br>" +
                                    f"Implied Vol: {impliedVolatility:.1%}" 
                                    for strike, volume, bid, ask, last, impliedVolatility in zip(
                                        puts['strike'],
                                        puts['volume'],
                                        puts['bid'],
                                        puts['ask'],
                                        puts['lastPrice'],
                                        puts['impliedVolatility']
                                    )
                                ]
                            else:
                                # Fallback to simpler hover text if not all columns are available
                                volume_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Volume: {volume:,}"
                                    for strike, volume in zip(
                                        puts['strike'],
                                        puts['volume']
                                    )
                                ]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=puts['strike'],
                                    y=puts['volume'],
                                    name="Volume",
                                    marker_color='rgba(50, 171, 96, 0.7)',
                                    hoverinfo='text',
                                    hovertext=volume_hover
                                )
                            )
                            
                            # Check if Greek values are available in the puts data
                            has_greeks = all(column in puts.columns for column in ['delta', 'gamma', 'theta', 'vega'])
                            
                            if has_greeks:
                                # Create hover text with Greeks for open interest line
                                oi_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Open Interest: {oi:,}<br>" +
                                    f"Delta: {delta:.3f}<br>" +
                                    f"Gamma: {gamma:.4f}<br>" +
                                    f"Theta: {theta:.4f}<br>" +
                                    f"Vega: {vega:.4f}"
                                    for strike, oi, delta, gamma, theta, vega in zip(
                                        puts['strike'],
                                        puts['openInterest'],
                                        puts['delta'],
                                        puts['gamma'],
                                        puts['theta'],
                                        puts['vega']
                                    )
                                ]
                            else:
                                # Create simple hover text without Greeks
                                oi_hover = [
                                    f"Strike: ${strike:.2f}<br>" +
                                    f"Open Interest: {oi:,}"
                                    for strike, oi in zip(
                                        puts['strike'],
                                        puts['openInterest']
                                    )
                                ]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['strike'],
                                    y=puts['openInterest'],
                                    name="Open Interest",
                                    marker_color='rgba(250, 128, 114, 1)',
                                    hoverinfo='text',
                                    hovertext=oi_hover
                                )
                            )
                            
                            # Add line for current price
                            fig.add_vline(
                                x=current_price,
                                line_width=2,
                                line_dash="dash",
                                line_color="yellow",
                                annotation_text=f"Current: ${current_price:.2f}",
                                annotation_position="top right"
                            )
                            
                            fig.update_layout(
                                title="Puts - Volume and Open Interest by Strike Price",
                                xaxis_title="Strike Price ($)",
                                yaxis_title="Contracts",
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
                        else:
                            st.info("No puts data available for the selected expiration date.")
                
                elif analysis_type == "Greeks Analysis":
                    # Calculate Greeks for calls and puts
                    if calls is not None and not calls.empty:
                        calls = calculate_greeks(calls, current_price, days_to_exp / 365, option_type='call')
                    
                    if puts is not None and not puts.empty:
                        puts = calculate_greeks(puts, current_price, days_to_exp / 365, option_type='put')
                    
                    # Create tabs for different Greeks
                    tab1, tab2, tab3, tab4 = st.tabs(["Delta", "Gamma", "Theta", "Vega"])
                    
                    with tab1:
                        st.subheader("Delta Analysis")
                        st.markdown("""
                        **Delta** measures the rate of change in an option's price relative to changes in the underlying asset's price.
                        - Call options have positive delta between 0 and 1
                        - Put options have negative delta between -1 and 0
                        - Delta close to 1 or -1 means the option behaves more like the underlying stock
                        """)
                        
                        fig = go.Figure()
                        
                        if calls is not None and not calls.empty and 'delta' in calls.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['strike'],
                                    y=calls['delta'],
                                    name="Calls Delta",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='green')
                                )
                            )
                        
                        if puts is not None and not puts.empty and 'delta' in puts.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['strike'],
                                    y=puts['delta'],
                                    name="Puts Delta",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='red')
                                )
                            )
                        
                        # Add line for current price
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title="Delta vs Strike Price",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Delta",
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
                    
                    with tab2:
                        st.subheader("Gamma Analysis")
                        st.markdown("""
                        **Gamma** measures the rate of change in delta relative to changes in the underlying asset's price.
                        - Higher gamma means the option's delta will change more rapidly
                        - Gamma is typically highest for at-the-money options and decreases as options move in- or out-of-the-money
                        """)
                        
                        fig = go.Figure()
                        
                        if calls is not None and not calls.empty and 'gamma' in calls.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['strike'],
                                    y=calls['gamma'],
                                    name="Calls Gamma",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='green')
                                )
                            )
                        
                        if puts is not None and not puts.empty and 'gamma' in puts.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['strike'],
                                    y=puts['gamma'],
                                    name="Puts Gamma",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='red')
                                )
                            )
                        
                        # Add line for current price
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title="Gamma vs Strike Price",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Gamma",
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
                    
                    with tab3:
                        st.subheader("Theta Analysis")
                        st.markdown("""
                        **Theta** measures the rate of decline in the value of an option due to the passage of time.
                        - Theta is typically negative for both calls and puts
                        - The value is expressed as the option price decrease per day
                        - Theta increases as the option approaches expiration
                        """)
                        
                        fig = go.Figure()
                        
                        if calls is not None and not calls.empty and 'theta' in calls.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['strike'],
                                    y=calls['theta'],
                                    name="Calls Theta",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='green')
                                )
                            )
                        
                        if puts is not None and not puts.empty and 'theta' in puts.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['strike'],
                                    y=puts['theta'],
                                    name="Puts Theta",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='red')
                                )
                            )
                        
                        # Add line for current price
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title="Theta vs Strike Price",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Theta",
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
                    
                    with tab4:
                        st.subheader("Vega Analysis")
                        st.markdown("""
                        **Vega** measures the rate of change in an option's price relative to changes in the underlying asset's implied volatility.
                        - Higher vega means the option is more sensitive to volatility changes
                        - Vega is typically highest for at-the-money options with more time until expiration
                        """)
                        
                        fig = go.Figure()
                        
                        if calls is not None and not calls.empty and 'vega' in calls.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['strike'],
                                    y=calls['vega'],
                                    name="Calls Vega",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='green')
                                )
                            )
                        
                        if puts is not None and not puts.empty and 'vega' in puts.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['strike'],
                                    y=puts['vega'],
                                    name="Puts Vega",
                                    mode='lines+markers',
                                    marker=dict(size=8),
                                    line=dict(width=2, color='red')
                                )
                            )
                        
                        # Add line for current price
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title="Vega vs Strike Price",
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Vega",
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
                
                elif analysis_type == "Strategy Analysis":
                    st.subheader("Options Strategy Analysis")
                    
                    # Strategy selection
                    strategy = st.selectbox(
                        "Select options strategy",
                        ["Long Call", "Long Put", "Covered Call", "Protective Put", "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"]
                    )
                    
                    # For strategies that need strike selection
                    if strategy in ["Long Call", "Long Put"]:
                        options_df = calls if strategy == "Long Call" else puts
                        
                        if options_df is not None and not options_df.empty:
                            # Sort by strike
                            options_df = options_df.sort_values('strike')
                            
                            # Allow user to select a strike price
                            strikes = list(options_df['strike'].unique())
                            
                            # Find closest strike to current price
                            closest_strike_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price))
                            
                            selected_strike = st.selectbox(
                                "Select strike price",
                                strikes,
                                index=closest_strike_idx
                            )
                            
                            # Analyze the strategy
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price=selected_strike,
                                option_price=options_df[options_df['strike'] == selected_strike]['lastPrice'].iloc[0],
                                days_to_expiration=days_to_exp
                            )
                            
                            # Display the payoff diagram
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=analysis_results['stock_prices'],
                                    y=analysis_results['payoff'],
                                    mode='lines',
                                    name='Payoff',
                                    line=dict(width=3)
                                )
                            )
                            
                            # Add breakeven
                            if 'breakeven' in analysis_results:
                                fig.add_vline(
                                    x=analysis_results['breakeven'],
                                    line_width=2,
                                    line_dash="dash",
                                    line_color="white",
                                    annotation_text=f"Breakeven: ${analysis_results['breakeven']:.2f}",
                                    annotation_position="top left"
                                )
                            
                            # Add current price marker
                            fig.add_vline(
                                x=current_price,
                                line_width=2,
                                line_dash="dash",
                                line_color="yellow",
                                annotation_text=f"Current: ${current_price:.2f}",
                                annotation_position="top right"
                            )
                            
                            # Add strike price marker
                            fig.add_vline(
                                x=selected_strike,
                                line_width=2,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Strike: ${selected_strike:.2f}",
                                annotation_position="bottom right"
                            )
                            
                            fig.update_layout(
                                title=f"{strategy} Payoff Diagram - Strike: ${selected_strike:.2f}",
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display strategy metrics
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric(
                                    label="Max Profit",
                                    value=f"${analysis_results.get('max_profit', 'Unlimited')}" if analysis_results.get('max_profit') != 'Unlimited' else "Unlimited"
                                )
                            
                            with metric_col2:
                                st.metric(
                                    label="Max Loss",
                                    value=f"${analysis_results.get('max_loss', 'Unlimited')}" if analysis_results.get('max_loss') != 'Unlimited' else "Unlimited"
                                )
                            
                            with metric_col3:
                                st.metric(
                                    label="Breakeven",
                                    value=f"${analysis_results.get('breakeven', 'N/A'):.2f}" if analysis_results.get('breakeven') != 'N/A' else "N/A"
                                )
                            
                            with metric_col4:
                                st.metric(
                                    label="Initial Cost",
                                    value=f"${analysis_results.get('initial_cost', 'N/A'):.2f}" if analysis_results.get('initial_cost') != 'N/A' else "N/A"
                                )
                            
                            # Strategy description
                            st.subheader("Strategy Description")
                            st.markdown(analysis_results.get('description', 'No description available.'))
                        else:
                            st.error(f"No options data available for {strategy}.")
                    
                    elif strategy in ["Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"]:
                        st.info(f"For demonstration purposes, we're showing a pre-configured {strategy} strategy.")
                        
                        # Setup some default values for visualization
                        if strategy == "Bull Call Spread":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price_lower=current_price * 0.95,
                                strike_price_higher=current_price * 1.05,
                                option_price_lower=5.0,
                                option_price_higher=2.0,
                                days_to_expiration=days_to_exp
                            )
                        elif strategy == "Bear Put Spread":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price_lower=current_price * 0.95,
                                strike_price_higher=current_price * 1.05,
                                option_price_lower=2.0,
                                option_price_higher=5.0,
                                days_to_expiration=days_to_exp
                            )
                        elif strategy == "Straddle":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price=current_price,
                                call_price=5.0,
                                put_price=5.0,
                                days_to_expiration=days_to_exp
                            )
                        elif strategy == "Strangle":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                call_strike=current_price * 1.05,
                                put_strike=current_price * 0.95,
                                call_price=3.0,
                                put_price=3.0,
                                days_to_expiration=days_to_exp
                            )
                        
                        # Display the payoff diagram
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=analysis_results['stock_prices'],
                                y=analysis_results['payoff'],
                                mode='lines',
                                name='Payoff',
                                line=dict(width=3)
                            )
                        )
                        
                        # Add breakeven points if available
                        if 'breakeven_lower' in analysis_results:
                            fig.add_vline(
                                x=analysis_results['breakeven_lower'],
                                line_width=2,
                                line_dash="dash",
                                line_color="white",
                                annotation_text=f"Lower Breakeven: ${analysis_results['breakeven_lower']:.2f}",
                                annotation_position="top left"
                            )
                        
                        if 'breakeven_upper' in analysis_results:
                            fig.add_vline(
                                x=analysis_results['breakeven_upper'],
                                line_width=2,
                                line_dash="dash",
                                line_color="white",
                                annotation_text=f"Upper Breakeven: ${analysis_results['breakeven_upper']:.2f}",
                                annotation_position="top right"
                            )
                        
                        # Add current price marker
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="bottom left"
                        )
                        
                        fig.update_layout(
                            title=f"{strategy} Payoff Diagram",
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display strategy metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Max Profit",
                                value=f"${analysis_results.get('max_profit', 'Unlimited')}" if analysis_results.get('max_profit') != 'Unlimited' else "Unlimited"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Max Loss",
                                value=f"${analysis_results.get('max_loss', 'Unlimited')}" if analysis_results.get('max_loss') != 'Unlimited' else "Unlimited"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="Initial Cost",
                                value=f"${analysis_results.get('initial_cost', 'N/A'):.2f}" if analysis_results.get('initial_cost') != 'N/A' else "N/A"
                            )
                        
                        # Strategy description
                        st.subheader("Strategy Description")
                        st.markdown(analysis_results.get('description', 'No description available.'))
                    
                    else:  # Covered Call, Protective Put
                        st.info(f"For demonstration purposes, we're showing a pre-configured {strategy} strategy.")
                        
                        # Setup some default values for visualization
                        if strategy == "Covered Call":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price=current_price * 1.05,
                                option_price=3.0,
                                days_to_expiration=days_to_exp
                            )
                        elif strategy == "Protective Put":
                            analysis_results = options_strategy_analysis(
                                strategy=strategy,
                                current_price=current_price,
                                strike_price=current_price * 0.95,
                                option_price=3.0,
                                days_to_expiration=days_to_exp
                            )
                        
                        # Display the payoff diagram
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=analysis_results['stock_prices'],
                                y=analysis_results['payoff'],
                                mode='lines',
                                name='Payoff',
                                line=dict(width=3)
                            )
                        )
                        
                        # Add breakeven if available
                        if 'breakeven' in analysis_results:
                            fig.add_vline(
                                x=analysis_results['breakeven'],
                                line_width=2,
                                line_dash="dash",
                                line_color="white",
                                annotation_text=f"Breakeven: ${analysis_results['breakeven']:.2f}",
                                annotation_position="top left"
                            )
                        
                        # Add current price marker
                        fig.add_vline(
                            x=current_price,
                            line_width=2,
                            line_dash="dash",
                            line_color="yellow",
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="bottom right"
                        )
                        
                        # Add strike price marker
                        strike = analysis_results.get('strike_price', current_price * 1.05)
                        fig.add_vline(
                            x=strike,
                            line_width=2,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"Strike: ${strike:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title=f"{strategy} Payoff Diagram",
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display strategy metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(
                                label="Max Profit",
                                value=f"${analysis_results.get('max_profit', 'Unlimited')}" if analysis_results.get('max_profit') != 'Unlimited' else "Unlimited"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Max Loss",
                                value=f"${analysis_results.get('max_loss', 'Unlimited')}" if analysis_results.get('max_loss') != 'Unlimited' else "Unlimited"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="Breakeven",
                                value=f"${analysis_results.get('breakeven', 'N/A'):.2f}" if analysis_results.get('breakeven') != 'N/A' else "N/A"
                            )
                        
                        with metric_col4:
                            st.metric(
                                label="Initial Investment",
                                value=f"${analysis_results.get('initial_investment', 'N/A'):.2f}" if analysis_results.get('initial_investment') != 'N/A' else "N/A"
                            )
                        
                        # Strategy description
                        st.subheader("Strategy Description")
                        st.markdown(analysis_results.get('description', 'No description available.'))
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    elif not ticker:
        st.info("Please select a stock ticker to begin options analysis.")
    elif not expiration_date:
        st.info("No options data available for the selected stock.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** Options trading involves significant risk and is not suitable for all investors. The examples shown are for 
    educational purposes only and do not constitute investment advice. Always consult with a financial advisor before 
    entering into options trades.
    """)

if __name__ == "__main__":
    main()
