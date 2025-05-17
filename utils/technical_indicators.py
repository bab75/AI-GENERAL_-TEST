import pandas as pd
import numpy as np

def add_all_indicators(df):
    """
    Add all technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicators
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure the dataframe has the expected columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_copy.columns for col in required_columns):
        # If columns are missing, return the original dataframe
        return df_copy
    
    try:
        # RSI (Relative Strength Index)
        delta = df_copy['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df_copy['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_copy['Close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = ema_12 - ema_26
        df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        df_copy['MACD_hist'] = df_copy['MACD'] - df_copy['MACD_signal']
        
        # Bollinger Bands
        sma_20 = df_copy['Close'].rolling(window=20).mean()
        std_20 = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_upper'] = sma_20 + (std_20 * 2)
        df_copy['BB_middle'] = sma_20
        df_copy['BB_lower'] = sma_20 - (std_20 * 2)
        
        # Stochastic Oscillator
        low_14 = df_copy['Low'].rolling(window=14).min()
        high_14 = df_copy['High'].rolling(window=14).max()
        df_copy['%K'] = 100 * ((df_copy['Close'] - low_14) / (high_14 - low_14))
        df_copy['%D'] = df_copy['%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = df_copy['High'] - df_copy['Low']
        tr2 = abs(df_copy['High'] - df_copy['Close'].shift())
        tr3 = abs(df_copy['Low'] - df_copy['Close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df_copy['ATR'] = tr.rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()
        
        # Simple Moving Averages
        df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
        df_copy['SMA_200'] = df_copy['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
        df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
        
        # Average Directional Index (ADX) - simplified implementation
        # First calculate +DM, -DM, +DI, -DI, and DX
        plus_dm = df_copy['High'].diff()
        minus_dm = df_copy['Low'].diff(-1).abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / df_copy['ATR'])
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / df_copy['ATR'])
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
        df_copy['ADX'] = dx.rolling(window=14).mean()
        
        # Commodity Channel Index (CCI)
        tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        ma_tp = tp.rolling(window=20).mean()
        md_tp = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        df_copy['CCI'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # Williams %R
        highest_high = df_copy['High'].rolling(window=14).max()
        lowest_low = df_copy['Low'].rolling(window=14).min()
        df_copy['WILLR'] = -100 * (highest_high - df_copy['Close']) / (highest_high - lowest_low)
        
        # Rate of Change (ROC)
        df_copy['ROC'] = df_copy['Close'].pct_change(10) * 100
        
        # Money Flow Index (MFI)
        tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        mf = tp * df_copy['Volume']
        
        pos_mf = mf.copy()
        neg_mf = mf.copy()
        
        pos_mf[tp.diff() < 0] = 0
        neg_mf[tp.diff() > 0] = 0
        
        pos_mf_sum = pos_mf.rolling(window=14).sum()
        neg_mf_sum = neg_mf.rolling(window=14).sum()
        
        money_ratio = pos_mf_sum / neg_mf_sum
        df_copy['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # Moving Average Envelope
        df_copy['Upper_Envelope'] = df_copy['SMA_20'] * 1.025
        df_copy['Lower_Envelope'] = df_copy['SMA_20'] * 0.975
        
        # Ichimoku Cloud (simplified version)
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df_copy['High'].rolling(window=9).max()
        nine_period_low = df_copy['Low'].rolling(window=9).min()
        df_copy['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        twentysix_period_high = df_copy['High'].rolling(window=26).max()
        twentysix_period_low = df_copy['Low'].rolling(window=26).min()
        df_copy['Kijun_sen'] = (twentysix_period_high + twentysix_period_low) / 2
        
        return df_copy
    
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return df_copy

def add_selected_indicators(df, indicators):
    """
    Add only the selected technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data
    indicators : list
        List of indicator names to add
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added selected technical indicators
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure the dataframe has the expected columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_copy.columns for col in required_columns):
        # If columns are missing, return the original dataframe
        return df_copy
    
    try:
        # Add only the requested indicators
        if 'RSI' in indicators:
            delta = df_copy['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        if 'MACD' in indicators:
            ema_12 = df_copy['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_copy['Close'].ewm(span=26, adjust=False).mean()
            df_copy['MACD'] = ema_12 - ema_26
            df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
            df_copy['MACD_hist'] = df_copy['MACD'] - df_copy['MACD_signal']
        
        if 'Bollinger Bands' in indicators:
            sma_20 = df_copy['Close'].rolling(window=20).mean()
            std_20 = df_copy['Close'].rolling(window=20).std()
            df_copy['BB_upper'] = sma_20 + (std_20 * 2)
            df_copy['BB_middle'] = sma_20
            df_copy['BB_lower'] = sma_20 - (std_20 * 2)
        
        if 'Stochastic Oscillator' in indicators:
            low_14 = df_copy['Low'].rolling(window=14).min()
            high_14 = df_copy['High'].rolling(window=14).max()
            df_copy['%K'] = 100 * ((df_copy['Close'] - low_14) / (high_14 - low_14))
            df_copy['%D'] = df_copy['%K'].rolling(window=3).mean()
        
        if 'ATR' in indicators:
            tr1 = df_copy['High'] - df_copy['Low']
            tr2 = abs(df_copy['High'] - df_copy['Close'].shift())
            tr3 = abs(df_copy['Low'] - df_copy['Close'].shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df_copy['ATR'] = tr.rolling(window=14).mean()
        
        if 'OBV' in indicators:
            df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()
        
        if 'Moving Averages' in indicators:
            df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
            df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
            df_copy['SMA_200'] = df_copy['Close'].rolling(window=200).mean()
            df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
            df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
        
        if 'ADX' in indicators:
            # Calculate ATR if not already done
            if 'ATR' not in df_copy.columns:
                tr1 = df_copy['High'] - df_copy['Low']
                tr2 = abs(df_copy['High'] - df_copy['Close'].shift())
                tr3 = abs(df_copy['Low'] - df_copy['Close'].shift())
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                df_copy['ATR'] = tr.rolling(window=14).mean()
            
            # First calculate +DM, -DM, +DI, -DI, and DX
            plus_dm = df_copy['High'].diff()
            minus_dm = df_copy['Low'].diff(-1).abs()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / df_copy['ATR'])
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / df_copy['ATR'])
            
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
            df_copy['ADX'] = dx.rolling(window=14).mean()
        
        if 'CCI' in indicators:
            tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
            ma_tp = tp.rolling(window=20).mean()
            md_tp = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
            df_copy['CCI'] = (tp - ma_tp) / (0.015 * md_tp)
        
        if 'Williams %R' in indicators:
            highest_high = df_copy['High'].rolling(window=14).max()
            lowest_low = df_copy['Low'].rolling(window=14).min()
            df_copy['WILLR'] = -100 * (highest_high - df_copy['Close']) / (highest_high - lowest_low)
        
        if 'ROC' in indicators:
            df_copy['ROC'] = df_copy['Close'].pct_change(10) * 100
        
        if 'MFI' in indicators:
            tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
            mf = tp * df_copy['Volume']
            
            pos_mf = mf.copy()
            neg_mf = mf.copy()
            
            pos_mf[tp.diff() < 0] = 0
            neg_mf[tp.diff() > 0] = 0
            
            pos_mf_sum = pos_mf.rolling(window=14).sum()
            neg_mf_sum = neg_mf.rolling(window=14).sum()
            
            money_ratio = pos_mf_sum / neg_mf_sum
            df_copy['MFI'] = 100 - (100 / (1 + money_ratio))
        
        return df_copy
    
    except Exception as e:
        print(f"Error adding selected indicators: {e}")
        return df_copy
