import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
from scipy.stats import norm

def get_options_chain(ticker, expiration_date):
    """
    Fetch options chain data for a given ticker and expiration date.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str
        Option expiration date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    tuple
        (calls_dataframe, puts_dataframe) containing options data
    """
    try:
        # Get options data
        stock = yf.Ticker(ticker)
        
        # Check if expiration date is available
        if expiration_date not in stock.options:
            return None, None
        
        # Get options chain
        options = stock.option_chain(expiration_date)
        calls = options.calls
        puts = options.puts
        
        return calls, puts
    
    except Exception as e:
        print(f"Error fetching options chain for {ticker}: {e}")
        return None, None

def calculate_greeks(options_df, current_price, time_to_expiry, option_type='call', risk_free_rate=0.03):
    """
    Calculate option Greeks for a given options chain.
    
    Parameters:
    -----------
    options_df : pandas.DataFrame
        DataFrame containing options data
    current_price : float
        Current price of the underlying stock
    time_to_expiry : float
        Time to expiration in years
    option_type : str, default 'call'
        Type of option ('call' or 'put')
    risk_free_rate : float, default 0.03
        Risk-free interest rate
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Greeks columns
    """
    if options_df is None or options_df.empty:
        return options_df
    
    # Make a copy to avoid modifying the original
    df = options_df.copy()
    
    # Calculate implied volatility
    if 'impliedVolatility' not in df.columns:
        # Default to a reasonable volatility if not available
        df['impliedVolatility'] = 0.3
    
    # Function to calculate d1 and d2 for Black-Scholes model
    def calculate_d1d2(S, K, T, r, sigma):
        if sigma <= 0 or T <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    # Calculate Delta
    def calculate_delta(S, K, T, r, sigma, option_type):
        d1, _ = calculate_d1d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # Put
            return norm.cdf(d1) - 1
    
    # Calculate Gamma
    def calculate_gamma(S, K, T, r, sigma):
        d1, _ = calculate_d1d2(S, K, T, r, sigma)
        
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Calculate Theta
    def calculate_theta(S, K, T, r, sigma, option_type):
        d1, d2 = calculate_d1d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # Put
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to daily theta
        return theta / 365
    
    # Calculate Vega
    def calculate_vega(S, K, T, r, sigma):
        d1, _ = calculate_d1d2(S, K, T, r, sigma)
        
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
        return vega
    
    # Apply Greek calculations
    df['delta'] = df.apply(
        lambda row: calculate_delta(
            current_price, row['strike'], time_to_expiry, 
            risk_free_rate, row['impliedVolatility'], option_type
        ), 
        axis=1
    )
    
    df['gamma'] = df.apply(
        lambda row: calculate_gamma(
            current_price, row['strike'], time_to_expiry, 
            risk_free_rate, row['impliedVolatility']
        ), 
        axis=1
    )
    
    df['theta'] = df.apply(
        lambda row: calculate_theta(
            current_price, row['strike'], time_to_expiry, 
            risk_free_rate, row['impliedVolatility'], option_type
        ), 
        axis=1
    )
    
    df['vega'] = df.apply(
        lambda row: calculate_vega(
            current_price, row['strike'], time_to_expiry, 
            risk_free_rate, row['impliedVolatility']
        ), 
        axis=1
    )
    
    return df

def options_strategy_analysis(strategy, **kwargs):
    """
    Analyze different options strategies and generate payoff diagrams.
    
    Parameters:
    -----------
    strategy : str
        Name of the options strategy to analyze
    **kwargs : dict
        Additional parameters specific to each strategy
    
    Returns:
    --------
    dict
        Dictionary containing analysis results including payoff data and metadata
    """
    # Create a range of stock prices for payoff diagram
    current_price = kwargs.get('current_price', 100)
    
    # Generate price range from 0.7 to 1.3 times the current price
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
    
    results = {
        'stock_prices': price_range,
        'payoff': []
    }
    
    # Long Call Strategy
    if strategy == 'Long Call':
        strike_price = kwargs.get('strike_price', current_price)
        option_price = kwargs.get('option_price', 5.0)
        
        # Calculate payoff at each potential stock price
        payoff = [max(0, price - strike_price) - option_price for price in price_range]
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven'] = strike_price + option_price
        
        # Calculate max profit and max loss
        results['max_loss'] = option_price
        results['max_profit'] = 'Unlimited'
        
        # Initial cost
        results['initial_cost'] = option_price
        
        # Description
        results['description'] = """
        **Long Call Strategy:**
        
        A long call gives you the right to buy the underlying stock at the strike price before expiration.
        
        - **Maximum Loss:** Limited to the premium paid for the call option.
        - **Maximum Profit:** Unlimited as the stock price can theoretically rise infinitely.
        - **Breakeven Point:** Strike Price + Premium Paid
        - **When to Use:** When you expect the underlying stock price to rise significantly.
        """
    
    # Long Put Strategy
    elif strategy == 'Long Put':
        strike_price = kwargs.get('strike_price', current_price)
        option_price = kwargs.get('option_price', 5.0)
        
        # Calculate payoff at each potential stock price
        payoff = [max(0, strike_price - price) - option_price for price in price_range]
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven'] = strike_price - option_price
        
        # Calculate max profit and max loss
        results['max_loss'] = option_price
        results['max_profit'] = strike_price - option_price
        
        # Initial cost
        results['initial_cost'] = option_price
        
        # Description
        results['description'] = """
        **Long Put Strategy:**
        
        A long put gives you the right to sell the underlying stock at the strike price before expiration.
        
        - **Maximum Loss:** Limited to the premium paid for the put option.
        - **Maximum Profit:** Limited to the strike price minus the premium paid (occurs if the stock price goes to zero).
        - **Breakeven Point:** Strike Price - Premium Paid
        - **When to Use:** When you expect the underlying stock price to fall significantly.
        """
    
    # Covered Call Strategy
    elif strategy == 'Covered Call':
        strike_price = kwargs.get('strike_price', current_price * 1.1)
        option_price = kwargs.get('option_price', 3.0)
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            if price <= strike_price:
                # Stock price gain/loss + premium received
                p = price - current_price + option_price
            else:
                # Maximum profit is strike_price - current_price + premium
                p = strike_price - current_price + option_price
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven'] = current_price - option_price
        
        # Calculate max profit and max loss
        results['max_profit'] = strike_price - current_price + option_price
        results['max_loss'] = current_price - option_price
        
        # Strike price reference
        results['strike_price'] = strike_price
        
        # Initial investment
        results['initial_investment'] = current_price - option_price
        
        # Description
        results['description'] = """
        **Covered Call Strategy:**
        
        A covered call involves owning the underlying stock and selling a call option against it.
        
        - **Maximum Loss:** The value of the underlying shares minus the premium received (if the stock price goes to zero).
        - **Maximum Profit:** Limited to the strike price minus the purchase price of the stock plus the premium received.
        - **Breakeven Point:** Stock Purchase Price - Premium Received
        - **When to Use:** When you expect the stock to remain flat or rise slightly.
        """
    
    # Protective Put Strategy
    elif strategy == 'Protective Put':
        strike_price = kwargs.get('strike_price', current_price * 0.9)
        option_price = kwargs.get('option_price', 3.0)
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            if price >= strike_price:
                # Stock price gain/loss - premium paid
                p = price - current_price - option_price
            else:
                # Minimum loss is strike_price - current_price - premium
                p = strike_price - current_price - option_price
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven'] = current_price + option_price
        
        # Calculate max profit and max loss
        results['max_profit'] = 'Unlimited'
        results['max_loss'] = current_price - strike_price + option_price
        
        # Strike price reference
        results['strike_price'] = strike_price
        
        # Initial investment
        results['initial_investment'] = current_price + option_price
        
        # Description
        results['description'] = """
        **Protective Put Strategy:**
        
        A protective put involves owning the underlying stock and buying a put option to protect against downside risk.
        
        - **Maximum Loss:** Limited to the purchase price of the stock minus the strike price plus the premium paid.
        - **Maximum Profit:** Unlimited as the stock price can theoretically rise infinitely (minus the premium paid).
        - **Breakeven Point:** Stock Purchase Price + Premium Paid
        - **When to Use:** When you want to protect a long stock position against a significant decline.
        """
    
    # Bull Call Spread Strategy
    elif strategy == 'Bull Call Spread':
        strike_price_lower = kwargs.get('strike_price_lower', current_price * 0.95)
        strike_price_higher = kwargs.get('strike_price_higher', current_price * 1.05)
        option_price_lower = kwargs.get('option_price_lower', 5.0)
        option_price_higher = kwargs.get('option_price_higher', 2.0)
        
        # Calculate net premium paid
        net_premium = option_price_lower - option_price_higher
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            # Long call payoff
            long_call = max(0, price - strike_price_lower)
            
            # Short call payoff
            short_call = -max(0, price - strike_price_higher)
            
            p = long_call + short_call - net_premium
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven_lower'] = strike_price_lower + net_premium
        
        # Calculate max profit and max loss
        results['max_profit'] = strike_price_higher - strike_price_lower - net_premium
        results['max_loss'] = net_premium
        
        # Initial cost
        results['initial_cost'] = net_premium
        
        # Description
        results['description'] = """
        **Bull Call Spread Strategy:**
        
        A bull call spread involves buying a call option at a lower strike price and selling a call option at a higher strike price, both with the same expiration date.
        
        - **Maximum Loss:** Limited to the net premium paid.
        - **Maximum Profit:** Limited to the difference between strike prices minus the net premium paid.
        - **Breakeven Point:** Lower Strike Price + Net Premium Paid
        - **When to Use:** When you expect a moderate increase in the underlying stock price.
        """
    
    # Bear Put Spread Strategy
    elif strategy == 'Bear Put Spread':
        strike_price_lower = kwargs.get('strike_price_lower', current_price * 0.95)
        strike_price_higher = kwargs.get('strike_price_higher', current_price * 1.05)
        option_price_lower = kwargs.get('option_price_lower', 2.0)
        option_price_higher = kwargs.get('option_price_higher', 5.0)
        
        # Calculate net premium paid
        net_premium = option_price_higher - option_price_lower
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            # Long put payoff
            long_put = max(0, strike_price_higher - price)
            
            # Short put payoff
            short_put = -max(0, strike_price_lower - price)
            
            p = long_put + short_put - net_premium
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven point
        results['breakeven_upper'] = strike_price_higher - net_premium
        
        # Calculate max profit and max loss
        results['max_profit'] = strike_price_higher - strike_price_lower - net_premium
        results['max_loss'] = net_premium
        
        # Initial cost
        results['initial_cost'] = net_premium
        
        # Description
        results['description'] = """
        **Bear Put Spread Strategy:**
        
        A bear put spread involves buying a put option at a higher strike price and selling a put option at a lower strike price, both with the same expiration date.
        
        - **Maximum Loss:** Limited to the net premium paid.
        - **Maximum Profit:** Limited to the difference between strike prices minus the net premium paid.
        - **Breakeven Point:** Higher Strike Price - Net Premium Paid
        - **When to Use:** When you expect a moderate decrease in the underlying stock price.
        """
    
    # Straddle Strategy
    elif strategy == 'Straddle':
        strike_price = kwargs.get('strike_price', current_price)
        call_price = kwargs.get('call_price', 5.0)
        put_price = kwargs.get('put_price', 5.0)
        
        # Calculate total premium paid
        total_premium = call_price + put_price
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            # Call option payoff
            call_payoff = max(0, price - strike_price)
            
            # Put option payoff
            put_payoff = max(0, strike_price - price)
            
            p = call_payoff + put_payoff - total_premium
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven points
        results['breakeven_lower'] = strike_price - total_premium
        results['breakeven_upper'] = strike_price + total_premium
        
        # Calculate max profit and max loss
        results['max_profit'] = 'Unlimited'
        results['max_loss'] = total_premium
        
        # Initial cost
        results['initial_cost'] = total_premium
        
        # Description
        results['description'] = """
        **Straddle Strategy:**
        
        A straddle involves buying both a call option and a put option with the same strike price and expiration date.
        
        - **Maximum Loss:** Limited to the total premium paid for both options.
        - **Maximum Profit:** Unlimited, as the stock price can theoretically rise or fall significantly.
        - **Breakeven Points:** 
          * Upper: Strike Price + Total Premium Paid
          * Lower: Strike Price - Total Premium Paid
        - **When to Use:** When you expect significant movement in the underlying stock price but are unsure of the direction.
        """
    
    # Strangle Strategy
    elif strategy == 'Strangle':
        call_strike = kwargs.get('call_strike', current_price * 1.05)
        put_strike = kwargs.get('put_strike', current_price * 0.95)
        call_price = kwargs.get('call_price', 3.0)
        put_price = kwargs.get('put_price', 3.0)
        
        # Calculate total premium paid
        total_premium = call_price + put_price
        
        # Calculate payoff at each potential stock price
        payoff = []
        for price in price_range:
            # Call option payoff
            call_payoff = max(0, price - call_strike)
            
            # Put option payoff
            put_payoff = max(0, put_strike - price)
            
            p = call_payoff + put_payoff - total_premium
            payoff.append(p)
        
        results['payoff'] = payoff
        
        # Calculate breakeven points
        results['breakeven_lower'] = put_strike - total_premium
        results['breakeven_upper'] = call_strike + total_premium
        
        # Calculate max profit and max loss
        results['max_profit'] = 'Unlimited'
        results['max_loss'] = total_premium
        
        # Initial cost
        results['initial_cost'] = total_premium
        
        # Description
        results['description'] = """
        **Strangle Strategy:**
        
        A strangle involves buying an out-of-the-money call option and an out-of-the-money put option with the same expiration date.
        
        - **Maximum Loss:** Limited to the total premium paid for both options.
        - **Maximum Profit:** Unlimited, as the stock price can theoretically rise or fall significantly.
        - **Breakeven Points:** 
          * Upper: Call Strike Price + Total Premium Paid
          * Lower: Put Strike Price - Total Premium Paid
        - **When to Use:** When you expect significant movement in the underlying stock price but are unsure of the direction. 
          Compared to a straddle, a strangle typically costs less but requires a larger price movement to be profitable.
        """
    
    return results
