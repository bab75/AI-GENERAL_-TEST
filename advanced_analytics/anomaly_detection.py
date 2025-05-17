"""
AI-Powered Anomaly Detection Module

This module provides advanced anomaly detection capabilities for stock prices,
volume, volatility, and other indicators, using statistical and AI-powered methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
import os
from typing import Dict, List, Tuple, Union, Optional

# Optional AI enhancement when API keys are available
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = "ANTHROPIC_API_KEY" in os.environ
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = "OPENAI_API_KEY" in os.environ
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting anomalies in stock data using statistical methods.
    
    The detector uses multiple approaches:
    1. Z-score method for outlier detection
    2. Moving average deviation analysis
    3. Volume-price divergence detection
    4. Volatility analysis
    5. Pattern break detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the anomaly detector with configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters
                - z_score_threshold: Z-score threshold for anomaly detection (default: 2.5)
                - window_size: Window size for moving average calculations (default: 20)
                - volume_change_threshold: Threshold for volume change (default: 200%)
                - volatility_window: Window for volatility calculation (default: 20)
                - volatility_threshold: Threshold for volatility anomalies (default: 2.0)
                - use_ai: Whether to use AI for anomaly interpretation (default: False)
        """
        self.config = config or {}
        self.z_score_threshold = self.config.get('z_score_threshold', 2.5)
        self.window_size = self.config.get('window_size', 20)
        self.volume_change_threshold = self.config.get('volume_change_threshold', 2.0)  # 200%
        self.volatility_window = self.config.get('volatility_window', 20)
        self.volatility_threshold = self.config.get('volatility_threshold', 2.0)
        self.use_ai = self.config.get('use_ai', False)
        
        # Initialize optional AI clients
        self.anthropic_client = None
        self.openai_client = None
        
        if self.use_ai:
            self._initialize_ai_clients()
    
    def _initialize_ai_clients(self):
        """Initialize AI clients if API keys are available"""
        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
                
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
    
    def detect_price_anomalies(self, data: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """
        Detect anomalies in price data using Z-scores.
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate Z-scores for the specified column
        result[f'{column}_zscore'] = stats.zscore(result[column].fillna(method='ffill'))
        
        # Flag anomalies based on Z-score threshold
        result[f'{column}_anomaly'] = result[f'{column}_zscore'].abs() > self.z_score_threshold
        
        # Calculate rolling mean and standard deviation
        result[f'{column}_rolling_mean'] = result[column].rolling(window=self.window_size).mean()
        result[f'{column}_rolling_std'] = result[column].rolling(window=self.window_size).std()
        
        # Calculate deviation from rolling mean
        result[f'{column}_deviation'] = (result[column] - result[f'{column}_rolling_mean']) / result[f'{column}_rolling_std']
        
        # Flag anomalies based on deviation threshold
        result[f'{column}_deviation_anomaly'] = result[f'{column}_deviation'].abs() > self.z_score_threshold
        
        # Combined anomaly flag (either Z-score or deviation)
        result[f'{column}_combined_anomaly'] = result[f'{column}_anomaly'] | result[f'{column}_deviation_anomaly']
        
        return result
    
    def detect_volume_anomalies(self, data: pd.DataFrame, column: str = 'Volume') -> pd.DataFrame:
        """
        Detect anomalies in trading volume.
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Volume')
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate Z-scores for volume
        result[f'{column}_zscore'] = stats.zscore(result[column].fillna(method='ffill'))
        
        # Flag anomalies based on Z-score threshold
        result[f'{column}_anomaly'] = result[f'{column}_zscore'] > self.z_score_threshold
        
        # Calculate volume change percentage
        result[f'{column}_pct_change'] = result[column].pct_change() * 100
        
        # Flag anomalies based on percentage change threshold
        result[f'{column}_change_anomaly'] = result[f'{column}_pct_change'] > (self.volume_change_threshold * 100)
        
        # Calculate rolling average volume
        result[f'{column}_rolling_avg'] = result[column].rolling(window=self.window_size).mean()
        
        # Calculate ratio to rolling average
        result[f'{column}_rolling_ratio'] = result[column] / result[f'{column}_rolling_avg']
        
        # Flag anomalies based on ratio to rolling average
        result[f'{column}_ratio_anomaly'] = result[f'{column}_rolling_ratio'] > self.volume_change_threshold
        
        # Combined anomaly flag
        result[f'{column}_combined_anomaly'] = (
            result[f'{column}_anomaly'] | 
            result[f'{column}_change_anomaly'] | 
            result[f'{column}_ratio_anomaly']
        )
        
        return result
    
    def detect_volume_price_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect divergence between price and volume movements.
        
        Args:
            data: DataFrame with stock data (must contain 'Close' and 'Volume' columns)
            
        Returns:
            DataFrame with divergence flags and scores
        """
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            logger.error("Data must contain 'Close' and 'Volume' columns")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate price and volume changes
        result['price_pct_change'] = result['Close'].pct_change() * 100
        result['volume_pct_change'] = result['Volume'].pct_change() * 100
        
        # Calculate correlation between price and volume changes in rolling window
        result['price_volume_correlation'] = (
            result['price_pct_change'].rolling(window=self.window_size)
            .corr(result['volume_pct_change'])
        )
        
        # Detect negative correlation (divergence)
        result['price_volume_divergence'] = result['price_volume_correlation'] < -0.5
        
        # Detect high volume with price decline
        result['high_volume_price_decline'] = (
            (result['volume_pct_change'] > 50) & 
            (result['price_pct_change'] < -1)
        )
        
        # Detect low volume with price increase
        result['low_volume_price_increase'] = (
            (result['volume_pct_change'] < -50) & 
            (result['price_pct_change'] > 1)
        )
        
        # Combined divergence flag
        result['divergence_anomaly'] = (
            result['price_volume_divergence'] | 
            result['high_volume_price_decline'] | 
            result['low_volume_price_increase']
        )
        
        return result
    
    def detect_volatility_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in price volatility.
        
        Args:
            data: DataFrame with stock data (must contain 'Close' column)
            
        Returns:
            DataFrame with volatility anomaly flags
        """
        if 'Close' not in data.columns:
            logger.error("Data must contain 'Close' column")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate daily returns
        result['daily_return'] = result['Close'].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        result['volatility'] = result['daily_return'].rolling(window=self.volatility_window).std() * np.sqrt(252)  # Annualized
        
        # Calculate Z-score of volatility
        result['volatility_zscore'] = stats.zscore(result['volatility'].fillna(method='ffill'))
        
        # Flag volatility anomalies
        result['volatility_anomaly'] = result['volatility_zscore'].abs() > self.volatility_threshold
        
        # Calculate volatility change
        result['volatility_change'] = result['volatility'].pct_change()
        
        # Flag sudden volatility changes
        result['volatility_change_anomaly'] = result['volatility_change'].abs() > 0.5  # 50% change in volatility
        
        # Combined volatility anomaly flag
        result['combined_volatility_anomaly'] = result['volatility_anomaly'] | result['volatility_change_anomaly']
        
        return result
    
    def detect_pattern_breaks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect breaks in established price patterns.
        
        Args:
            data: DataFrame with stock data (must contain 'Close' column)
            
        Returns:
            DataFrame with pattern break flags
        """
        if 'Close' not in data.columns:
            logger.error("Data must contain 'Close' column")
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Calculate moving averages
        result['ma_short'] = result['Close'].rolling(window=10).mean()
        result['ma_medium'] = result['Close'].rolling(window=30).mean()
        result['ma_long'] = result['Close'].rolling(window=50).mean()
        
        # Calculate previous trend (using short-term MA slope)
        result['ma_short_slope'] = result['ma_short'].pct_change(5)
        
        # Identify trend direction
        result['uptrend'] = result['ma_short'] > result['ma_medium']
        result['downtrend'] = result['ma_short'] < result['ma_medium']
        
        # Calculate distance from moving averages
        result['distance_from_ma_medium'] = (result['Close'] - result['ma_medium']) / result['ma_medium']
        
        # Detect pattern breaks
        result['uptrend_break'] = (
            result['uptrend'].shift(1) & 
            ~result['uptrend'] & 
            (result['distance_from_ma_medium'] < -0.02)
        )
        
        result['downtrend_break'] = (
            result['downtrend'].shift(1) & 
            ~result['downtrend'] & 
            (result['distance_from_ma_medium'] > 0.02)
        )
        
        # Combined pattern break flag
        result['pattern_break'] = result['uptrend_break'] | result['downtrend_break']
        
        return result
    
    def detect_all_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all anomaly detection methods and compile results.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            DataFrame with all anomaly flags
        """
        # Check if dataframe is empty
        if data.empty:
            logger.error("Empty DataFrame provided")
            return pd.DataFrame()
            
        # Required columns
        required_columns = ['Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Required column '{col}' not found in data")
                return pd.DataFrame()
        
        # Run all detection methods
        price_anomalies = self.detect_price_anomalies(data)
        volume_anomalies = self.detect_volume_anomalies(data)
        divergence_anomalies = self.detect_volume_price_divergence(data)
        volatility_anomalies = self.detect_volatility_anomalies(data)
        pattern_breaks = self.detect_pattern_breaks(data)
        
        # Combine all results
        result = data.copy()
        
        # Add price anomaly columns
        for col in price_anomalies.columns:
            if col not in data.columns:
                result[col] = price_anomalies[col]
                
        # Add volume anomaly columns
        for col in volume_anomalies.columns:
            if col not in data.columns and col not in result.columns:
                result[col] = volume_anomalies[col]
                
        # Add divergence anomaly columns
        for col in divergence_anomalies.columns:
            if col not in data.columns and col not in result.columns:
                result[col] = divergence_anomalies[col]
                
        # Add volatility anomaly columns
        for col in volatility_anomalies.columns:
            if col not in data.columns and col not in result.columns:
                result[col] = volatility_anomalies[col]
                
        # Add pattern break columns
        for col in pattern_breaks.columns:
            if col not in data.columns and col not in result.columns:
                result[col] = pattern_breaks[col]
        
        # Create aggregate anomaly score
        anomaly_columns = [
            'Close_combined_anomaly',
            'Volume_combined_anomaly',
            'divergence_anomaly',
            'combined_volatility_anomaly',
            'pattern_break'
        ]
        
        # Count the number of anomaly flags set to True
        result['anomaly_score'] = 0
        for col in anomaly_columns:
            if col in result.columns:
                result['anomaly_score'] += result[col].astype(int)
        
        # Add a single anomaly flag for any type of anomaly detected
        result['is_anomaly'] = result['anomaly_score'] > 0
        
        return result
    
    def get_anomaly_dates(self, data: pd.DataFrame, min_score: int = 1) -> List[str]:
        """
        Get dates of detected anomalies.
        
        Args:
            data: DataFrame with anomaly detection results
            min_score: Minimum anomaly score to include (default: 1)
            
        Returns:
            List of dates with anomalies
        """
        if 'anomaly_score' not in data.columns:
            logger.error("Anomaly score column not found in data")
            return []
            
        # Filter dates with anomaly score >= min_score
        anomaly_dates = data[data['anomaly_score'] >= min_score].index
        
        # Convert to string format
        return [date.strftime('%Y-%m-%d') for date in anomaly_dates]
    
    def get_strongest_anomalies(self, data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Get the strongest anomalies based on anomaly score.
        
        Args:
            data: DataFrame with anomaly detection results
            top_n: Number of top anomalies to return (default: 5)
            
        Returns:
            DataFrame with top anomalies
        """
        if 'anomaly_score' not in data.columns:
            logger.error("Anomaly score column not found in data")
            return pd.DataFrame()
            
        # Sort by anomaly score in descending order
        sorted_anomalies = data[data['anomaly_score'] > 0].sort_values('anomaly_score', ascending=False)
        
        # Return top N anomalies
        return sorted_anomalies.head(top_n)
    
    def interpret_anomalies(self, data: pd.DataFrame, ticker: str) -> Dict:
        """
        Interpret detected anomalies and provide human-readable explanations.
        
        Args:
            data: DataFrame with anomaly detection results
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with anomaly interpretations
        """
        if data.empty or 'anomaly_score' not in data.columns:
            logger.error("Invalid data for anomaly interpretation")
            return {"error": "Invalid data for anomaly interpretation"}
            
        # Get dates with anomalies
        anomaly_dates = self.get_anomaly_dates(data)
        
        if not anomaly_dates:
            return {"message": "No anomalies detected"}
            
        # Get top anomalies
        top_anomalies = self.get_strongest_anomalies(data)
        
        # Create basic interpretations
        interpretations = {}
        
        for date, row in top_anomalies.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            
            # Initialize interpretation for this date
            interpretations[date_str] = {
                "date": date_str,
                "anomaly_score": row['anomaly_score'],
                "close_price": row['Close'],
                "volume": row['Volume'],
                "anomaly_types": [],
                "interpretation": ""
            }
            
            # Add anomaly types
            if row.get('Close_combined_anomaly', False):
                interpretations[date_str]["anomaly_types"].append("Price Anomaly")
            
            if row.get('Volume_combined_anomaly', False):
                interpretations[date_str]["anomaly_types"].append("Volume Anomaly")
            
            if row.get('divergence_anomaly', False):
                interpretations[date_str]["anomaly_types"].append("Price-Volume Divergence")
            
            if row.get('combined_volatility_anomaly', False):
                interpretations[date_str]["anomaly_types"].append("Volatility Anomaly")
            
            if row.get('pattern_break', False):
                interpretations[date_str]["anomaly_types"].append("Pattern Break")
            
            # Create basic interpretation text
            basic_interpretation = f"Anomaly detected on {date_str} with score {row['anomaly_score']}. "
            
            # Add details based on anomaly types
            if "Price Anomaly" in interpretations[date_str]["anomaly_types"]:
                zscore = row.get('Close_zscore', 0)
                direction = "increase" if zscore > 0 else "decrease"
                basic_interpretation += f"Unusual price {direction} detected. "
            
            if "Volume Anomaly" in interpretations[date_str]["anomaly_types"]:
                volume_change = row.get('Volume_pct_change', 0)
                basic_interpretation += f"Unusual trading volume ({volume_change:.2f}% change). "
            
            if "Price-Volume Divergence" in interpretations[date_str]["anomaly_types"]:
                basic_interpretation += "Price and volume are moving in unexpected directions. "
            
            if "Volatility Anomaly" in interpretations[date_str]["anomaly_types"]:
                volatility = row.get('volatility', 0)
                basic_interpretation += f"Abnormal volatility detected ({volatility:.2f}). "
            
            if "Pattern Break" in interpretations[date_str]["anomaly_types"]:
                if row.get('uptrend_break', False):
                    basic_interpretation += "Uptrend pattern broken. "
                elif row.get('downtrend_break', False):
                    basic_interpretation += "Downtrend pattern broken. "
            
            interpretations[date_str]["interpretation"] = basic_interpretation
            
            # If AI is available, enhance the interpretation
            if self.use_ai and (self.anthropic_client or self.openai_client):
                ai_interpretation = self._get_ai_interpretation(row, ticker, date_str)
                if ai_interpretation:
                    interpretations[date_str]["ai_interpretation"] = ai_interpretation
        
        return {
            "anomaly_count": len(anomaly_dates),
            "anomalies": interpretations
        }
    
    def _get_ai_interpretation(self, anomaly_data: pd.Series, ticker: str, date: str) -> str:
        """
        Get AI-powered interpretation of anomaly data.
        
        Args:
            anomaly_data: Series with anomaly data for a specific date
            ticker: Stock ticker symbol
            date: Date string
            
        Returns:
            AI-generated interpretation text
        """
        # Prepare data for prompt
        data_summary = "\n".join([
            f"Date: {date}",
            f"Ticker: {ticker}",
            f"Close Price: ${anomaly_data.get('Close', 'N/A')}",
            f"Volume: {anomaly_data.get('Volume', 'N/A')}",
            f"Anomaly Score: {anomaly_data.get('anomaly_score', 'N/A')}",
            f"Price Z-Score: {anomaly_data.get('Close_zscore', 'N/A'):.2f}",
            f"Volume Z-Score: {anomaly_data.get('Volume_zscore', 'N/A'):.2f}",
            f"Price-Volume Correlation: {anomaly_data.get('price_volume_correlation', 'N/A'):.2f}",
            f"Volatility: {anomaly_data.get('volatility', 'N/A'):.4f}",
            f"Pattern Break: {'Yes' if anomaly_data.get('pattern_break', False) else 'No'}"
        ])
        
        # First try Anthropic if available
        if self.anthropic_client:
            try:
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=300,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""You are a stock market analyst. Interpret the following stock anomaly detection results and 
                            provide a concise explanation (max 3 sentences) of what this anomaly might indicate and what 
                            a trader should watch for:
                            
                            {data_summary}
                            
                            Focus only on facts, do not make specific predictions about future price movements.
                            Be direct and concise."""
                        }
                    ]
                )
                return response.content[0].text.strip()
            except Exception as e:
                logger.error(f"Error getting Anthropic interpretation: {str(e)}")
        
        # Try OpenAI if Anthropic failed or is not available
        if self.openai_client:
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a stock market analyst providing concise interpretations of anomaly detection results."
                        },
                        {
                            "role": "user",
                            "content": f"""Interpret the following stock anomaly detection results and 
                            provide a concise explanation (max 3 sentences) of what this anomaly might indicate and what 
                            a trader should watch for:
                            
                            {data_summary}
                            
                            Focus only on facts, do not make specific predictions about future price movements.
                            Be direct and concise."""
                        }
                    ],
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error getting OpenAI interpretation: {str(e)}")
        
        # Return empty string if both methods fail
        return ""


def detect_anomalies(ticker: str, data: pd.DataFrame, config: Dict = None) -> Dict:
    """
    Detect anomalies in stock data.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        config: Configuration parameters for anomaly detection
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        detector = AnomalyDetector(config)
        results = detector.detect_all_anomalies(data)
        
        # Get anomaly dates and interpretations
        anomaly_dates = detector.get_anomaly_dates(results)
        interpretations = detector.interpret_anomalies(results, ticker)
        
        # Only return the data with anomalies
        anomaly_data = results[results['is_anomaly'] == True].copy() if not results.empty else pd.DataFrame()
        
        return {
            "ticker": ticker,
            "anomaly_count": len(anomaly_dates),
            "anomaly_dates": anomaly_dates,
            "interpretations": interpretations,
            "anomaly_data": anomaly_data
        }
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e)
        }


def detect_multi_stock_anomalies(tickers: List[str], data_dict: Dict[str, pd.DataFrame], 
                                config: Dict = None) -> Dict:
    """
    Detect anomalies across multiple stocks.
    
    Args:
        tickers: List of stock ticker symbols
        data_dict: Dictionary with ticker symbols as keys and DataFrames as values
        config: Configuration parameters for anomaly detection
        
    Returns:
        Dictionary with anomaly detection results for all stocks
    """
    results = {}
    
    for ticker in tickers:
        if ticker in data_dict and not data_dict[ticker].empty:
            results[ticker] = detect_anomalies(ticker, data_dict[ticker], config)
        else:
            results[ticker] = {"ticker": ticker, "error": "No data available"}
    
    # Calculate overall statistics
    total_anomalies = sum(result.get("anomaly_count", 0) for result in results.values())
    
    return {
        "stocks": results,
        "total_anomalies": total_anomalies,
        "anomaly_count_by_stock": {ticker: result.get("anomaly_count", 0) for ticker, result in results.items()}
    }