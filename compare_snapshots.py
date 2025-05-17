#!/usr/bin/env python3
"""
Stock Market Snapshot Comparison Module

This standalone module analyzes two stock market snapshots (from different time periods)
to identify trends, changes, and notable market movements. It accepts CSV or Excel files
and generates various visualizations and analyses.

Each snapshot file should contain columns:
Symbol, Name, Last Sale, Net Change, % Change, Market Cap, Country, IPO Year, Volume, Sector, Industry
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import re
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union
import webbrowser
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Output directory for generated visualizations
OUTPUT_DIR = "snapshot_analysis"


def prepare_safe_scatter_data(df, size_column, hover_columns=None):
    """
    Helper function to safely prepare data for scatter plots by handling NaN values
    
    Args:
        df: DataFrame to process
        size_column: Column to use for marker size
        hover_columns: List of columns to clean for hover data
        
    Returns:
        DataFrame with NaN values handled safely
    """
    # Create a copy to avoid modifying the original
    safe_df = df.copy()
    
    # Fill NaN values in the size column
    if size_column in safe_df.columns:
        safe_df[size_column] = safe_df[size_column].fillna(1.0)
        
        # Replace any infinities
        safe_df = safe_df.replace([np.inf, -np.inf], np.nan)
        
        # Ensure numeric type
        safe_df[size_column] = pd.to_numeric(safe_df[size_column], errors='coerce')
        
        # Final check for NaN values
        safe_df[size_column] = safe_df[size_column].fillna(1.0)
        
        # Drop any rows still containing NaN in the size column
        safe_df = safe_df.dropna(subset=[size_column])
    
    # Clean hover columns if specified
    if hover_columns:
        for col in hover_columns:
            if col in safe_df.columns:
                safe_df[col] = safe_df[col].fillna(0)
    
    return safe_df


class SnapshotAnalyzer:
    """Main class for analyzing stock market snapshots"""
    
    def __init__(self, previous_file: str, current_file: str, output_dir: str = OUTPUT_DIR):
        """
        Initialize the analyzer with file paths
        
        Args:
            previous_file: Path to previous snapshot file (CSV or Excel)
            current_file: Path to current snapshot file (CSV or Excel)
            output_dir: Directory to save output visualizations
        """
        self.previous_file = previous_file
        self.current_file = current_file
        self.output_dir = output_dir
        self.previous_df = None
        self.current_df = None
        self.merged_df = None
        self.sector_analysis = None
        self.created_files = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _detect_file_type(self, file_path: str) -> str:
        """
        Auto-detect whether file is CSV or Excel
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: 'csv' or 'excel'
        """
        if file_path.lower().endswith('.csv'):
            return 'csv'
        elif file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
            return 'excel'
        else:
            # Try to infer from content
            with open(file_path, 'r', errors='ignore') as f:
                header = f.readline()
                if header.count(',') > 3:  # If multiple commas, likely CSV
                    return 'csv'
            return 'excel'  # Default to excel if can't determine
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize DataFrame columns and values
        
        Args:
            df: DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure required columns exist (with various possible names)
        column_mapping = {
            'Symbol': ['Symbol', 'Ticker', 'symbol', 'ticker'],
            'Name': ['Name', 'Company', 'CompanyName', 'name', 'company_name'],
            'Last Sale': ['Last Sale', 'LastSale', 'Price', 'Close', 'last_sale', 'price'],
            'Net Change': ['Net Change', 'Change', 'net_change', 'change'],
            '% Change': ['% Change', 'PctChange', 'PercentChange', 'pct_change', 'percent_change'],
            'Market Cap': ['Market Cap', 'MarketCap', 'market_cap'],
            'Country': ['Country', 'country'],
            'IPO Year': ['IPO Year', 'IPOYear', 'ipo_year'],
            'Volume': ['Volume', 'volume'],
            'Sector': ['Sector', 'sector'],
            'Industry': ['Industry', 'industry']
        }
        
        # Standardize column names and create missing ones
        standard_columns = {}
        data_issues = []
        
        for standard_name, possible_names in column_mapping.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    standard_columns[standard_name] = df[name]
                    found = True
                    break
            if not found:
                # Create empty column if missing
                standard_columns[standard_name] = np.nan
                issue = f"Column {standard_name} not found in data, created empty column"
                data_issues.append({"issue_type": "missing_column", "issue": issue})
                logger.warning(issue)
        
        # Create a new DataFrame with standardized columns
        cleaned_df = pd.DataFrame(standard_columns)
        
        # Track issues with data cleaning
        cleaning_stats = {
            "total_rows": len(cleaned_df),
            "issues_detected": 0,
            "data_issues": data_issues,
            "columns_fixed": [],
            "abnormal_values_fixed": 0
        }
        
        # Convert numeric columns and track issues
        numeric_cols = ['Last Sale', 'Net Change', '% Change', 'Market Cap', 'Volume']
        for col in numeric_cols:
            # Check for non-numeric values
            non_numeric_count = cleaned_df[col].astype(str).str.replace('[$%,()]', '', regex=True).str.isnumeric().value_counts().get(False, 0)
            
            if non_numeric_count > 0:
                issue = f"Column {col} has {non_numeric_count} non-numeric values"
                data_issues.append({"issue_type": "non_numeric", "column": col, "count": non_numeric_count})
                cleaning_stats["issues_detected"] += non_numeric_count
                cleaning_stats["columns_fixed"].append(col)
                logger.warning(issue)
            
            # Convert to numeric
            cleaned_df[col] = pd.to_numeric(
                cleaned_df[col].astype(str).str.replace('[$%,()]', '', regex=True), 
                errors='coerce'
            )
            
            # Check for NaN values after conversion
            na_count = cleaned_df[col].isna().sum()
            if na_count > 0:
                issue = f"Column {col} has {na_count} null values after cleaning"
                data_issues.append({"issue_type": "null_values", "column": col, "count": na_count})
                cleaning_stats["issues_detected"] += na_count
                logger.warning(issue)
        
        # Check for abnormal values in numeric columns
        for col in numeric_cols:
            # Skip columns with all NaN
            if cleaned_df[col].isna().all():
                continue
                
            # Check for extreme outliers (beyond 3 standard deviations)
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            
            if not np.isnan(mean) and not np.isnan(std) and std > 0:
                lower_bound = mean - 3*std
                upper_bound = mean + 3*std
                
                outliers_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    issue = f"Column {col} has {outlier_count} outlier values"
                    data_issues.append({"issue_type": "outliers", "column": col, "count": outlier_count})
                    cleaning_stats["issues_detected"] += outlier_count
                    cleaning_stats["abnormal_values_fixed"] += outlier_count
                    logger.warning(issue)
                    
                    # Winsorize the outliers (cap them at the boundaries)
                    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
        
        # Handle IPO Year
        cleaned_df['IPO Year'] = pd.to_numeric(
            cleaned_df['IPO Year'].astype(str).str.extract(r'(\d{4})', expand=False), 
            errors='coerce'
        )
        
        # Fill missing values
        cleaned_df['Sector'].fillna('Unknown', inplace=True)
        cleaned_df['Industry'].fillna('Unknown', inplace=True)
        cleaned_df['Country'].fillna('Unknown', inplace=True)
        
        # Store cleaning stats in the DataFrame metadata
        cleaned_df.attrs['cleaning_stats'] = cleaning_stats
        
        return cleaned_df
        
    def _load_data(self) -> None:
        """Load and prepare data from input files"""
        # Load previous snapshot
        prev_type = self._detect_file_type(self.previous_file)
        if prev_type == 'csv':
            self.previous_df = pd.read_csv(self.previous_file)
        else:
            self.previous_df = pd.read_excel(self.previous_file)
            
        # Load current snapshot
        curr_type = self._detect_file_type(self.current_file)
        if curr_type == 'csv':
            self.current_df = pd.read_csv(self.current_file)
        else:
            self.current_df = pd.read_excel(self.current_file)
        
        # Clean and standardize both dataframes
        self.previous_df = self._clean_dataframe(self.previous_df)
        self.current_df = self._clean_dataframe(self.current_df)
        
        # Add snapshot identifier columns
        self.previous_df['Snapshot'] = 'Previous'
        self.current_df['Snapshot'] = 'Current'
        
        # Extract filenames for labels
        self.prev_label = os.path.splitext(os.path.basename(self.previous_file))[0]
        self.curr_label = os.path.splitext(os.path.basename(self.current_file))[0]
        
        # Merge DataFrames for comparative analysis
        self._merge_data()
        
        logger.info(f"Loaded previous snapshot with {len(self.previous_df)} stocks")
        logger.info(f"Loaded current snapshot with {len(self.current_df)} stocks")
        logger.info(f"Found {len(self.merged_df)} stocks for comparative analysis")
        
    def _merge_data(self) -> None:
        """Merge previous and current snapshots for comparative analysis"""
        # Rename columns in the previous df for clarity
        prev_cols = {
            'Last Sale': 'Previous_Last_Sale',
            'Net Change': 'Previous_Net_Change',
            '% Change': 'Previous_Pct_Change',
            'Market Cap': 'Previous_Market_Cap',
            'Volume': 'Previous_Volume',
        }
        prev_df = self.previous_df.rename(columns=prev_cols)
        
        # Rename columns in the current df
        curr_cols = {
            'Last Sale': 'Current_Last_Sale',
            'Net Change': 'Current_Net_Change',
            '% Change': 'Current_Pct_Change',
            'Market Cap': 'Current_Market_Cap',
            'Volume': 'Current_Volume',
        }
        curr_df = self.current_df.rename(columns=curr_cols)
        
        # Merge the DataFrames on Symbol
        self.merged_df = pd.merge(
            prev_df[['Symbol', 'Name', 'Previous_Last_Sale', 'Previous_Net_Change', 
                    'Previous_Pct_Change', 'Previous_Market_Cap', 'Previous_Volume', 
                    'Sector', 'Industry', 'Country', 'IPO Year']],
            curr_df[['Symbol', 'Current_Last_Sale', 'Current_Net_Change', 
                   'Current_Pct_Change', 'Current_Market_Cap', 'Current_Volume']],
            on='Symbol',
            how='inner'
        )
        
        # Calculate changes between snapshots
        self.merged_df['Price_Change'] = self.merged_df['Current_Last_Sale'] - self.merged_df['Previous_Last_Sale']
        self.merged_df['Price_Change_Pct'] = (self.merged_df['Price_Change'] / self.merged_df['Previous_Last_Sale']) * 100
        self.merged_df['Volume_Change'] = self.merged_df['Current_Volume'] - self.merged_df['Previous_Volume']
        self.merged_df['Volume_Change_Pct'] = (self.merged_df['Volume_Change'] / self.merged_df['Previous_Volume']) * 100
        self.merged_df['Market_Cap_Change'] = self.merged_df['Current_Market_Cap'] - self.merged_df['Previous_Market_Cap']
        self.merged_df['Market_Cap_Change_Pct'] = (self.merged_df['Market_Cap_Change'] / self.merged_df['Previous_Market_Cap']) * 100
        
        # Add profit/loss status column
        self.merged_df['Profit_Loss_Status'] = np.where(
            self.merged_df['Price_Change'] > 0, 'Profit', 
            np.where(self.merged_df['Price_Change'] < 0, 'Loss', 'Neutral')
        )
        
        # Add performance category
        conditions = [
            (self.merged_df['Price_Change_Pct'] >= 10),  # Strong gain
            (self.merged_df['Price_Change_Pct'] >= 3) & (self.merged_df['Price_Change_Pct'] < 10),  # Moderate gain
            (self.merged_df['Price_Change_Pct'] > -3) & (self.merged_df['Price_Change_Pct'] < 3),  # Neutral
            (self.merged_df['Price_Change_Pct'] <= -3) & (self.merged_df['Price_Change_Pct'] > -10),  # Moderate loss
            (self.merged_df['Price_Change_Pct'] <= -10)  # Strong loss
        ]
        
        categories = [
            'Strong Gain', 
            'Moderate Gain', 
            'Neutral', 
            'Moderate Loss', 
            'Strong Loss'
        ]
        
        self.merged_df['Performance_Category'] = np.select(conditions, categories, default='Neutral')
        
        # Add volume trend category
        vol_conditions = [
            (self.merged_df['Volume_Change_Pct'] >= 100),  # High volume increase
            (self.merged_df['Volume_Change_Pct'] >= 20) & (self.merged_df['Volume_Change_Pct'] < 100),  # Moderate volume increase
            (self.merged_df['Volume_Change_Pct'] > -20) & (self.merged_df['Volume_Change_Pct'] < 20),  # Stable volume
            (self.merged_df['Volume_Change_Pct'] <= -20)  # Volume decrease
        ]
        
        vol_categories = [
            'High Volume Increase', 
            'Moderate Volume Increase', 
            'Stable Volume', 
            'Volume Decrease'
        ]
        
        self.merged_df['Volume_Trend'] = np.select(vol_conditions, vol_categories, default='Stable Volume')
        
        # Define Yahoo Finance URL for each symbol
        self.merged_df['Yahoo_URL'] = 'https://finance.yahoo.com/quote/' + self.merged_df['Symbol']
        
        # Initialize filtered_df to the full merged dataset
        self.filtered_df = self.merged_df.copy()
        
        # Store cleaning statistics from both dataframes
        if hasattr(self.previous_df, 'attrs') and 'cleaning_stats' in self.previous_df.attrs:
            self.previous_cleaning_stats = self.previous_df.attrs['cleaning_stats']
        else:
            self.previous_cleaning_stats = {"total_rows": len(self.previous_df), "issues_detected": 0}
            
        if hasattr(self.current_df, 'attrs') and 'cleaning_stats' in self.current_df.attrs:
            self.current_cleaning_stats = self.current_df.attrs['cleaning_stats']
        else:
            self.current_cleaning_stats = {"total_rows": len(self.current_df), "issues_detected": 0}
    
    def apply_filters(self, min_price: float = 0.0, max_price: float = 1000.0,
                   min_gain: float = 0.0, max_loss: float = -5.0, 
                   min_market_cap: float = 10.0, max_market_cap: float = 100.0,
                   min_volume: int = 10000, max_volume_change: float = 500.0,
                   selected_sectors: list = []):
        """
        Apply filters to the merged dataset before analysis
        
        Args:
            min_price: Minimum stock price to include
            max_price: Maximum stock price to include
            min_gain: Minimum gain percentage to include
            max_loss: Maximum loss percentage to include
            min_market_cap: Minimum market cap in billions
            max_market_cap: Maximum market cap in billions
            min_volume: Minimum trading volume
            max_volume_change: Maximum volume change percentage
            selected_sectors: List of sectors to include (empty or ['All'] means all sectors)
            
        Returns:
            None - modifies the filtered_df property directly
        """
        logger.info("Applying filters to data before analysis...")
        
        # Start with a fresh copy of the merged data
        self.filtered_df = self.merged_df.copy()
        
        # Apply price filters
        self.filtered_df = self.filtered_df[
            (self.filtered_df['Current_Last_Sale'] >= min_price) & 
            (self.filtered_df['Current_Last_Sale'] <= max_price)
        ]
        
        # Apply market cap filters (convert to billions)
        self.filtered_df = self.filtered_df[
            (self.filtered_df['Current_Market_Cap'] / 1e9 >= min_market_cap) & 
            (self.filtered_df['Current_Market_Cap'] / 1e9 <= max_market_cap)
        ]
        
        # Apply volume filters
        self.filtered_df = self.filtered_df[
            (self.filtered_df['Current_Volume'] >= min_volume) &
            (self.filtered_df['Volume_Change_Pct'] <= max_volume_change)
        ]
        
        # Apply sector filter if specific sectors are selected
        if selected_sectors and len(selected_sectors) > 0 and 'All' not in selected_sectors:
            self.filtered_df = self.filtered_df[self.filtered_df['Sector'].isin(selected_sectors)]
            
        logger.info(f"After filtering: {len(self.filtered_df)} stocks remain for analysis")
    
    def analyze_price_movement(self, top_n: int = 10, min_price: float = 0.0, max_price: float = 1000.0, 
                          min_gain: float = 0.0, max_loss: float = -5.0, min_market_cap: float = 10.0,
                          max_market_cap: float = 100.0, selected_sectors: list = []) -> Dict:
        """
        Analyze price movements between snapshots
        
        Args:
            top_n: Number of top/bottom stocks to identify
            min_price: Minimum price to include (used only if no prior filtering)
            max_price: Maximum price to include (used only if no prior filtering)
            min_gain: Minimum gain percentage to include
            max_loss: Maximum loss percentage to include
            min_market_cap: Minimum market cap in billions (used only if no prior filtering)
            max_market_cap: Maximum market cap in billions (used only if no prior filtering)
            selected_sectors: List of sectors to include (used only if no prior filtering)
            
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing price movements...")
        results = {}
        
        # Use the already filtered dataframe
        working_df = self.filtered_df
        
        # Apply gain/loss filters
        gainers_df = working_df[working_df['Price_Change_Pct'] >= min_gain]
        losers_df = working_df[working_df['Price_Change_Pct'] <= max_loss]
        
        # Get top gainers and losers by percentage
        gainers = gainers_df.sort_values('Price_Change_Pct', ascending=False).head(top_n)
        losers = losers_df.sort_values('Price_Change_Pct', ascending=True).head(top_n)
        
        # Create horizontal bar chart for top gainers and losers
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top Gainers (%)", "Top Losers (%)"),
            horizontal_spacing=0.2
        )
        
        # Add traces for gainers
        fig.add_trace(
            go.Bar(
                y=gainers['Symbol'],
                x=gainers['Price_Change_Pct'],
                text=[f"{x:.2f}%" for x in gainers['Price_Change_Pct']],
                textposition='auto',
                orientation='h',
                marker_color='green',
                hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Current Price: $%{customdata[0]:.2f}<br>Previous Price: $%{customdata[1]:.2f}<extra></extra>',
                customdata=np.stack((gainers['Current_Last_Sale'], gainers['Previous_Last_Sale']), axis=-1),
                name="Gainers"
            ),
            row=1, col=1
        )
        
        # Add traces for losers
        fig.add_trace(
            go.Bar(
                y=losers['Symbol'],
                x=losers['Price_Change_Pct'],
                text=[f"{x:.2f}%" for x in losers['Price_Change_Pct']],
                textposition='auto',
                orientation='h',
                marker_color='red',
                hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Current Price: $%{customdata[0]:.2f}<br>Previous Price: $%{customdata[1]:.2f}<extra></extra>',
                customdata=np.stack((losers['Current_Last_Sale'], losers['Previous_Last_Sale']), axis=-1),
                name="Losers"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Price Movement Analysis",
            height=500,
            width=1000,
            showlegend=False,
            template="plotly_white"
        )
        
        # Add hyperlinks to stock symbols (clickable to Yahoo Finance)
        for i, symbol in enumerate(gainers['Symbol']):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                go.layout.Annotation(
                    text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                    x=0,
                    y=symbol,
                    xref="x",
                    yref="y",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-5,
                    font=dict(color="blue", size=12)
                )
            )
            
        for i, symbol in enumerate(losers['Symbol']):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                go.layout.Annotation(
                    text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                    x=0,
                    y=symbol,
                    xref="x2",
                    yref="y2",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-5,
                    font=dict(color="blue", size=12)
                )
            )
            
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'price_movement.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create scatter plot of all stocks by sector
        # Prepare data for scatter plot using the helper function
        scatter_df = self.merged_df.copy()
        scatter_df['MarketCapSize'] = scatter_df['Current_Market_Cap'].fillna(1000000)  # Default size for NaN
        
        # Safely prepare data for scatter plot
        hover_columns = ['Name', 'Price_Change_Pct', 'Current_Last_Sale', 'Previous_Last_Sale']
        scatter_df = prepare_safe_scatter_data(scatter_df, 'MarketCapSize', hover_columns)
        
        # Cap very large values to maintain visual scale
        if len(scatter_df) > 5:
            max_size = scatter_df['MarketCapSize'].quantile(0.95)  # Use 95th percentile to avoid outliers
            scatter_df['MarketCapSize'] = scatter_df['MarketCapSize'].clip(upper=max_size)
        
        fig_scatter = px.scatter(
            scatter_df,
            x='Previous_Last_Sale',
            y='Current_Last_Sale',
            color='Sector',
            size='MarketCapSize',
            hover_name='Symbol',
            hover_data=['Name', 'Price_Change_Pct', 'Current_Last_Sale', 'Previous_Last_Sale'],
            log_x=True,
            log_y=True,
            title='Price Comparison (Log Scale)',
            labels={
                'Previous_Last_Sale': f'Previous Price ({self.prev_label})',
                'Current_Last_Sale': f'Current Price ({self.curr_label})'
            },
            height=800
        )
        
        # Add reference line (no change)
        max_val = max(self.merged_df['Previous_Last_Sale'].max(), self.merged_df['Current_Last_Sale'].max())
        min_val = min(self.merged_df['Previous_Last_Sale'].min(), self.merged_df['Current_Last_Sale'].min())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='No Change'
            )
        )
        
        # Save interactive HTML scatter plot
        html_file2 = os.path.join(self.output_dir, 'price_scatter.html')
        fig_scatter.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file2)
        
        # Create a histogram of price changes
        fig_hist = px.histogram(
            working_df,
            x='Price_Change_Pct',
            color='Sector',
            marginal='box',
            hover_data=['Symbol', 'Name', 'Current_Last_Sale'],
            title='Distribution of Price Changes',
            labels={'Price_Change_Pct': 'Price Change (%)'},
            template='plotly_white'
        )
        
        # Save interactive HTML histogram
        html_file3 = os.path.join(self.output_dir, 'price_distribution.html')
        fig_hist.write_html(html_file3, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file3)
        
        # Store results
        results['gainers'] = gainers[['Symbol', 'Name', 'Previous_Last_Sale', 'Current_Last_Sale', 'Price_Change', 'Price_Change_Pct', 'Yahoo_URL']].to_dict('records')
        results['losers'] = losers[['Symbol', 'Name', 'Previous_Last_Sale', 'Current_Last_Sale', 'Price_Change', 'Price_Change_Pct', 'Yahoo_URL']].to_dict('records')
        results['files'] = [html_file, html_file2, html_file3]
        
        logger.info(f"Price movement analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_volume_spikes(self, threshold_pct: float = 100, top_n: int = 20) -> Dict:
        """
        Analyze volume changes between snapshots
        
        Args:
            threshold_pct: Minimum percentage increase to qualify as spike
            top_n: Number of top volume changes to identify
            
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing volume changes...")
        results = {}
        
        # Use filtered data for analysis
        working_df = self.filtered_df
        
        # Filter for stocks with significant volume increases
        volume_filter = (working_df['Volume_Change_Pct'] >= threshold_pct) & (working_df['Current_Volume'] > 0)
        volume_spikes = working_df[volume_filter].sort_values('Volume_Change_Pct', ascending=False).head(top_n)
        
        # Create a bar chart for volume spikes
        fig = px.bar(
            volume_spikes,
            y='Symbol',
            x='Volume_Change_Pct',
            orientation='h',
            color='Price_Change_Pct',
            color_continuous_scale=['red', 'lightgray', 'green'],
            color_continuous_midpoint=0,
            hover_data=['Name', 'Current_Volume', 'Previous_Volume', 'Price_Change_Pct'],
            title=f'Top {top_n} Volume Spikes (≥{threshold_pct}% Increase)',
            labels={
                'Volume_Change_Pct': 'Volume Increase (%)',
                'Price_Change_Pct': 'Price Change (%)'
            },
            height=800
        )
        
        # Add clickable stock symbols (to Yahoo Finance)
        for i, row in enumerate(volume_spikes.itertuples()):
            url = f"https://finance.yahoo.com/quote/{row.Symbol}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{row.Symbol}</a>",
                x=0,
                y=row.Symbol,
                xref="x",
                yref="y",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5,
                font=dict(color="blue", size=12)
            )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'volume_spikes.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # HTML export only (removed PNG generation)
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create a scatter plot of volume vs price changes
        # Prepare data for scatter plot
        scatter_df = working_df.copy()
        scatter_df['MarketCapSize'] = scatter_df['Current_Market_Cap'].fillna(1000000)  # Default size for NaN
        
        # Also fill NaN values in hover data columns to prevent issues
        hover_columns = ['Name', 'Current_Volume', 'Previous_Volume']
        for col in hover_columns:
            if col in scatter_df.columns:
                scatter_df[col] = scatter_df[col].fillna(0)
        
        # Handle NaN values in the MarketCapSize column to avoid scatter plot errors
        scatter_df['MarketCapSize'] = scatter_df['MarketCapSize'].fillna(1.0)  # Default size for NaN values
        
        # Cap very large values to maintain visual scale
        if len(scatter_df) > 5:
            max_size = scatter_df['MarketCapSize'].quantile(0.95)  # Use 95th percentile to avoid outliers
            scatter_df['MarketCapSize'] = scatter_df['MarketCapSize'].clip(upper=max_size)
        
        fig_scatter = px.scatter(
            scatter_df,
            x='Volume_Change_Pct',
            y='Price_Change_Pct',
            color='Sector',
            size='MarketCapSize',
            hover_name='Symbol',
            hover_data=['Name', 'Current_Volume', 'Previous_Volume'],
            labels={
                'Volume_Change_Pct': 'Volume Change (%)',
                'Price_Change_Pct': 'Price Change (%)'
            },
            title='Volume vs Price Changes',
            template='plotly_white'
        )
        
        # Add quadrant lines
        fig_scatter.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        fig_scatter.add_annotation(x=50, y=5, text="Volume Up, Price Up<br>(Accumulation?)", showarrow=False)
        fig_scatter.add_annotation(x=50, y=-5, text="Volume Up, Price Down<br>(Distribution?)", showarrow=False)
        fig_scatter.add_annotation(x=-50, y=5, text="Volume Down, Price Up<br>(Lack of Sellers?)", showarrow=False)
        fig_scatter.add_annotation(x=-50, y=-5, text="Volume Down, Price Down<br>(Lack of Buyers?)", showarrow=False)
        
        # Save interactive HTML scatter plot
        html_file2 = os.path.join(self.output_dir, 'volume_price_relationship.html')
        fig_scatter.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file2)
        
        # Store results
        results['volume_spikes'] = volume_spikes[['Symbol', 'Name', 'Previous_Volume', 'Current_Volume', 'Volume_Change', 'Volume_Change_Pct', 'Price_Change_Pct', 'Yahoo_URL']].to_dict('records')
        results['files'] = [html_file, html_file2]
        
        logger.info(f"Volume analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_sectors(self) -> Dict:
        """
        Analyze sector-level performance
        
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing sector performance...")
        results = {}
        
        # Use filtered data for analysis
        working_df = self.filtered_df
        
        # Group by sector to calculate aggregated metrics
        sector_data = working_df.groupby('Sector').agg({
            'Symbol': 'count',
            'Previous_Market_Cap': 'sum',
            'Current_Market_Cap': 'sum',
            'Price_Change_Pct': 'mean',
            'Current_Last_Sale': 'mean',
            'Previous_Last_Sale': 'mean'
        }).reset_index()
        
        # Calculate sector-level changes
        sector_data['Market_Cap_Change'] = sector_data['Current_Market_Cap'] - sector_data['Previous_Market_Cap']
        sector_data['Market_Cap_Change_Pct'] = (sector_data['Market_Cap_Change'] / sector_data['Previous_Market_Cap']) * 100
        sector_data['Avg_Price_Change_Pct'] = sector_data['Price_Change_Pct']
        sector_data.rename(columns={'Symbol': 'Stock_Count'}, inplace=True)
        
        # Save to instance for later use
        self.sector_analysis = sector_data
        
        # Create a bar chart for sector performance
        fig = px.bar(
            sector_data.sort_values('Avg_Price_Change_Pct'),
            y='Sector',
            x='Avg_Price_Change_Pct',
            orientation='h',
            color='Avg_Price_Change_Pct',
            color_continuous_scale=['red', 'lightgray', 'green'],
            color_continuous_midpoint=0,
            hover_data=['Stock_Count', 'Current_Market_Cap', 'Market_Cap_Change_Pct'],
            title='Sector Performance (Average Price Change %)',
            labels={
                'Avg_Price_Change_Pct': 'Average Price Change (%)',
                'Stock_Count': 'Number of Stocks'
            },
            height=600
        )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'sector_performance.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create a treemap of market cap by sector
        fig_treemap = px.treemap(
            sector_data,
            path=['Sector'],
            values='Current_Market_Cap',
            color='Avg_Price_Change_Pct',
            color_continuous_scale=['red', 'lightgray', 'green'],
            color_continuous_midpoint=0,
            title='Market Cap by Sector (Colored by Price Change %)',
            hover_data=['Stock_Count', 'Market_Cap_Change_Pct'],
        )
        
        # Save interactive HTML treemap
        html_file2 = os.path.join(self.output_dir, 'sector_market_cap.html')
        fig_treemap.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file2)
        
        # Create a waterfall chart for sector market cap change
        sector_waterfall = sector_data.sort_values('Market_Cap_Change').copy()
        sector_waterfall['Color'] = sector_waterfall['Market_Cap_Change'].apply(
            lambda x: 'green' if x > 0 else 'red'
        )
        
        # Calculate baseline for waterfall chart
        total_previous = sector_waterfall['Previous_Market_Cap'].sum()
        total_current = sector_waterfall['Current_Market_Cap'].sum()
        total_change = total_current - total_previous
        total_change_pct = (total_change / total_previous) * 100
        
        fig_waterfall = go.Figure()
        
        # Add total starting point
        fig_waterfall.add_trace(go.Bar(
            name='Starting Total',
            y=['Total'],
            x=[total_previous],
            marker_color='lightgray',
            orientation='h',
            text=f'${total_previous/1e9:.1f}B',
            textposition='auto',
            hoverinfo='text',
            hovertext=f'Previous Total: ${total_previous/1e9:.1f}B'
        ))
        
        # Add sector changes
        for i, row in sector_waterfall.iterrows():
            fig_waterfall.add_trace(go.Bar(
                name=row['Sector'],
                y=['Changes'],
                x=[row['Market_Cap_Change']],
                marker_color=row['Color'],
                orientation='h',
                text=f"{row['Sector']}: {row['Market_Cap_Change']/1e9:.1f}B ({row['Market_Cap_Change_Pct']:.1f}%)",
                textposition='auto',
                hoverinfo='text',
                hovertext=f"{row['Sector']}<br>Change: ${row['Market_Cap_Change']/1e9:.1f}B<br>Percent: {row['Market_Cap_Change_Pct']:.1f}%"
            ))
        
        # Add total ending point
        fig_waterfall.add_trace(go.Bar(
            name='Ending Total',
            y=['Total'],
            x=[total_current],
            marker_color='lightblue',
            orientation='h',
            text=f'${total_current/1e9:.1f}B',
            textposition='auto',
            hoverinfo='text',
            hovertext=f'Current Total: ${total_current/1e9:.1f}B<br>Change: {total_change_pct:.1f}%'
        ))
        
        fig_waterfall.update_layout(
            title=f'Market Cap Changes by Sector (${total_change/1e9:.1f}B, {total_change_pct:.1f}%)',
            barmode='relative',
            height=400,
            width=1000,
            template='plotly_white'
        )
        
        # Save interactive HTML waterfall
        html_file3 = os.path.join(self.output_dir, 'sector_waterfall.html')
        fig_waterfall.write_html(html_file3, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file3)
        
        # Store results
        results['sector_data'] = sector_data.to_dict('records')
        results['total_change'] = {
            'previous_total': total_previous,
            'current_total': total_current,
            'absolute_change': total_change,
            'percent_change': total_change_pct
        }
        results['files'] = [html_file, html_file2, html_file3]
        
        logger.info(f"Sector analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_market_cap(self) -> Dict:
        """
        Analyze market cap movements overall and by sector
        
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing market cap changes...")
        results = {}
        
        # Use filtered data for analysis
        working_df = self.filtered_df
        
        # Calculate total market cap changes
        total_prev_cap = working_df['Previous_Market_Cap'].sum()
        total_curr_cap = working_df['Current_Market_Cap'].sum()
        total_change = total_curr_cap - total_prev_cap
        total_change_pct = (total_change / total_prev_cap) * 100
        
        # Get top market cap gainers and losers
        cap_gainers = working_df.sort_values('Market_Cap_Change', ascending=False).head(10)
        cap_losers = working_df.sort_values('Market_Cap_Change', ascending=True).head(10)
        
        # Create subplot with 2 bars
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Top Market Cap Gainers (${cap_gainers['Market_Cap_Change'].sum()/1e9:.1f}B)", 
                f"Top Market Cap Losers (-${abs(cap_losers['Market_Cap_Change'].sum())/1e9:.1f}B)"
            ),
            horizontal_spacing=0.15
        )
        
        # Add traces for gainers
        fig.add_trace(
            go.Bar(
                y=cap_gainers['Symbol'],
                x=cap_gainers['Market_Cap_Change'] / 1e9,  # Convert to billions
                text=[f"${x/1e9:.1f}B" for x in cap_gainers['Market_Cap_Change']],
                textposition='auto',
                orientation='h',
                marker_color='green',
                hovertemplate='<b>%{y}</b><br>Change: $%{x:.2f}B<br>Current: $%{customdata[0]:.1f}B<br>Previous: $%{customdata[1]:.1f}B<extra></extra>',
                customdata=np.stack(
                    (cap_gainers['Current_Market_Cap']/1e9, cap_gainers['Previous_Market_Cap']/1e9), 
                    axis=-1
                ),
                name="Gainers"
            ),
            row=1, col=1
        )
        
        # Add traces for losers
        fig.add_trace(
            go.Bar(
                y=cap_losers['Symbol'],
                x=cap_losers['Market_Cap_Change'] / 1e9,  # Convert to billions
                text=[f"${x/1e9:.1f}B" for x in cap_losers['Market_Cap_Change']],
                textposition='auto',
                orientation='h',
                marker_color='red',
                hovertemplate='<b>%{y}</b><br>Change: $%{x:.2f}B<br>Current: $%{customdata[0]:.1f}B<br>Previous: $%{customdata[1]:.1f}B<extra></extra>',
                customdata=np.stack(
                    (cap_losers['Current_Market_Cap']/1e9, cap_losers['Previous_Market_Cap']/1e9), 
                    axis=-1
                ),
                name="Losers"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Market Cap Changes (Total: ${total_change/1e9:.1f}B, {total_change_pct:.1f}%)",
            height=500,
            width=1000,
            showlegend=False,
            template="plotly_white"
        )
        
        # Add x-axis titles
        fig.update_xaxes(title_text="Change in Billions ($)", row=1, col=1)
        fig.update_xaxes(title_text="Change in Billions ($)", row=1, col=2)
        
        # Add hyperlinks to stock symbols
        for i, symbol in enumerate(cap_gainers['Symbol']):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                x=0,
                y=symbol,
                xref="x",
                yref="y",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5,
                font=dict(color="blue", size=12)
            )
            
        for i, symbol in enumerate(cap_losers['Symbol']):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                x=0,
                y=symbol,
                xref="x2",
                yref="y2",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5,
                font=dict(color="blue", size=12)
            )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'market_cap_changes.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create a bubble chart of companies by market cap
        # Prepare data for bubble chart
        bubble_df = working_df.copy()
        bubble_df['MarketCapSize'] = bubble_df['Current_Market_Cap'].fillna(1000000)  # Default size for NaN
        
        # Safely prepare data for scatter plot
        hover_columns = ['Symbol', 'Name', 'Previous_Market_Cap', 'Current_Market_Cap', 'Market_Cap_Change_Pct']
        bubble_df = prepare_safe_scatter_data(bubble_df, 'MarketCapSize', hover_columns)
        
        # Cap very large values to maintain visual scale
        if len(bubble_df) > 5:
            max_size = bubble_df['MarketCapSize'].quantile(0.95)  # Use 95th percentile to avoid outliers
            bubble_df['MarketCapSize'] = bubble_df['MarketCapSize'].clip(upper=max_size)
        
        fig_bubble = px.scatter(
            bubble_df,
            x='Previous_Market_Cap',
            y='Current_Market_Cap',
            size='MarketCapSize',
            color='Sector',
            hover_name='Symbol',
            log_x=True,
            log_y=True,
            hover_data=['Name', 'Market_Cap_Change_Pct'],
            title='Market Cap Comparison (Log Scale)',
            labels={
                'Previous_Market_Cap': f'Previous Market Cap ({self.prev_label})',
                'Current_Market_Cap': f'Current Market Cap ({self.curr_label})'
            },
            height=800
        )
        
        # Add reference line (no change)
        max_val = max(working_df['Previous_Market_Cap'].max(), working_df['Current_Market_Cap'].max())
        min_val = max(1, min(working_df['Previous_Market_Cap'].min(), working_df['Current_Market_Cap'].min()))
        fig_bubble.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='No Change'
            )
        )
        
        # Save interactive HTML bubble chart
        html_file2 = os.path.join(self.output_dir, 'market_cap_bubble.html')
        fig_bubble.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file2)
        
        # Store results
        results['market_cap_summary'] = {
            'previous_total': total_prev_cap,
            'current_total': total_curr_cap,
            'absolute_change': total_change,
            'percent_change': total_change_pct
        }
        results['cap_gainers'] = cap_gainers[['Symbol', 'Name', 'Previous_Market_Cap', 'Current_Market_Cap', 'Market_Cap_Change', 'Market_Cap_Change_Pct', 'Yahoo_URL']].to_dict('records')
        results['cap_losers'] = cap_losers[['Symbol', 'Name', 'Previous_Market_Cap', 'Current_Market_Cap', 'Market_Cap_Change', 'Market_Cap_Change_Pct', 'Yahoo_URL']].to_dict('records')
        results['files'] = [html_file, html_file2]
        
        logger.info(f"Market cap analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_ipos(self, recent_year: int = 2022) -> Dict:
        """
        Analyze performance of recent IPOs
        
        Args:
            recent_year: Minimum year to consider as recent IPO
            
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info(f"Analyzing recent IPOs (since {recent_year})...")
        results = {}
        
        # Filter for recent IPOs
        recent_ipos = self.merged_df[self.merged_df['IPO Year'] >= recent_year].copy()
        recent_ipos = recent_ipos.sort_values('Market_Cap_Change_Pct', ascending=False)
        
        if len(recent_ipos) == 0:
            logger.warning(f"No IPOs found since {recent_year}")
            results['recent_ipos'] = []
            results['files'] = []
            return results
        
        # Create bubble chart of recent IPOs by performance
        # Prepare data for bubble chart
        ipo_df = recent_ipos.copy()
        ipo_df['MarketCapSize'] = ipo_df['Current_Market_Cap'].fillna(1000000)  # Default size for NaN values
        
        # Safely prepare data for scatter plot
        hover_columns = ['Name', 'Current_Last_Sale', 'Price_Change_Pct', 'Market_Cap_Change_Pct']
        ipo_df = prepare_safe_scatter_data(ipo_df, 'MarketCapSize', hover_columns)
        
        # Cap very large values to maintain visual scale
        if len(ipo_df) > 5:
            max_size = ipo_df['MarketCapSize'].quantile(0.95)
            ipo_df['MarketCapSize'] = ipo_df['MarketCapSize'].clip(upper=max_size)
        elif len(ipo_df) > 0:
            max_size = ipo_df['MarketCapSize'].max()
            ipo_df['MarketCapSize'] = ipo_df['MarketCapSize'].clip(upper=max_size)
        
        fig = px.scatter(
            ipo_df,
            x='IPO Year',
            y='Price_Change_Pct',
            size='MarketCapSize',
            color='Sector',
            hover_name='Symbol',
            hover_data=['Name', 'Current_Last_Sale', 'Price_Change_Pct', 'Market_Cap_Change_Pct'],
            title=f'Recent IPOs Performance (Since {recent_year})',
            height=600,
            template='plotly_white'
        )
        
        # Add a reference line at y=0
        fig.add_hline(y=0, line_dash='dash', line_color='gray')
        
        # Add hyperlinks to stock symbols with annotations
        for _, row in recent_ipos.iterrows():
            url = f"https://finance.yahoo.com/quote/{row['Symbol']}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{row['Symbol']}</a>",
                x=row['IPO Year'],
                y=row['Price_Change_Pct'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#636363",
                ax=0,
                ay=-20,
                font=dict(color="blue", size=10)
            )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'recent_ipos.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create a bar chart of recent IPOs
        if len(recent_ipos) > 0:
            # Get top 20 and bottom 20 performers if there are enough IPOs
            top_ipos = recent_ipos.head(min(10, len(recent_ipos)))
            
            # Ensure we have some bottom performers if there are enough
            bottom_ipos = pd.DataFrame()
            if len(recent_ipos) > 10:
                bottom_ipos = recent_ipos.tail(min(10, len(recent_ipos) - 10))
            
            # Combine for the chart
            plot_ipos = pd.concat([top_ipos, bottom_ipos])
            
            fig_bar = px.bar(
                plot_ipos,
                y='Symbol',
                x='Price_Change_Pct',
                orientation='h',
                color='Price_Change_Pct',
                color_continuous_scale=['red', 'lightgray', 'green'],
                color_continuous_midpoint=0,
                hover_data=['Name', 'IPO Year', 'Current_Last_Sale', 'Current_Market_Cap'],
                title=f'Recent IPOs Price Performance (Since {recent_year})',
                height=max(400, len(plot_ipos) * 25),
                template='plotly_white'
            )
            
            # Add hyperlinks to stock symbols
            for i, row in enumerate(plot_ipos.itertuples()):
                url = f"https://finance.yahoo.com/quote/{row.Symbol}"
                fig_bar.add_annotation(
                    text=f"<a href='{url}' target='_blank'>{row.Symbol}</a>",
                    x=0,
                    y=row.Symbol,
                    xref="x",
                    yref="y",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    xshift=-5,
                    font=dict(color="blue", size=12)
                )
            
            # Save interactive HTML bar chart
            html_file2 = os.path.join(self.output_dir, 'recent_ipos_performance.html')
            fig_bar.write_html(html_file2, include_plotlyjs='cdn')
            
            # Track created files
            self.created_files.append(html_file2)
            
            # Group by IPO year and Sector
            ipo_by_year_sector = recent_ipos.groupby(['IPO Year', 'Sector']).agg({
                'Symbol': 'count',
                'Price_Change_Pct': 'mean',
                'Current_Market_Cap': 'sum'
            }).reset_index()
            
            ipo_by_year_sector.rename(columns={
                'Symbol': 'Count',
                'Price_Change_Pct': 'Avg_Price_Change_Pct'
            }, inplace=True)
            
            # Create heatmap of IPO performance by year and sector
            if len(ipo_by_year_sector) > 3:  # Only create if we have enough data
                # Pivot for the heatmap
                ipo_pivot = ipo_by_year_sector.pivot_table(
                    values='Avg_Price_Change_Pct', 
                    index='Sector', 
                    columns='IPO Year'
                )
                
                # Create heatmap figure
                fig_heat = px.imshow(
                    ipo_pivot,
                    text_auto=True,
                    color_continuous_scale=['red', 'white', 'green'],
                    color_continuous_midpoint=0,
                    title=f'IPO Performance by Sector and Year (Since {recent_year})',
                    labels=dict(x="IPO Year", y="Sector", color="Avg % Change"),
                    height=max(400, len(ipo_pivot) * 40),
                    width=800
                )
                
                # Save interactive HTML heatmap
                html_file3 = os.path.join(self.output_dir, 'ipo_heatmap.html')
                fig_heat.write_html(html_file3, include_plotlyjs='cdn')
                
                # Track created files
                self.created_files.append(html_file3)
                
                # Include in results
                results['files'] = [html_file, html_file2, html_file3]
            else:
                results['files'] = [html_file, html_file2]
        else:
            results['files'] = [html_file]
        
        # Store results
        results['recent_ipos'] = recent_ipos[['Symbol', 'Name', 'IPO Year', 'Sector', 'Previous_Last_Sale', 'Current_Last_Sale', 'Price_Change_Pct', 'Current_Market_Cap', 'Yahoo_URL']].to_dict('records')
        
        logger.info(f"IPO analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_relative_strength(self) -> Dict:
        """
        Analyze relative strength of stocks compared to their sectors
        
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing relative strength...")
        results = {}
        
        # Calculate sector average performance if not already done
        if self.sector_analysis is None:
            self.analyze_sectors()
        
        # Add sector average performance to main dataframe
        sector_map = self.sector_analysis.set_index('Sector')['Avg_Price_Change_Pct'].to_dict()
        self.merged_df['Sector_Avg_Change'] = self.merged_df['Sector'].map(sector_map)
        
        # Calculate relative strength
        self.merged_df['Relative_Strength'] = self.merged_df['Price_Change_Pct'] - self.merged_df['Sector_Avg_Change']
        
        # Get outperformers and underperformers
        outperformers = self.merged_df[self.merged_df['Relative_Strength'] > 0].sort_values('Relative_Strength', ascending=False).head(15)
        underperformers = self.merged_df[self.merged_df['Relative_Strength'] < 0].sort_values('Relative_Strength', ascending=True).head(15)
        
        # Create subplots for outperformers and underperformers
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "Top Sector Outperformers", 
                "Top Sector Underperformers"
            ),
            horizontal_spacing=0.15
        )
        
        # Add traces for outperformers
        fig.add_trace(
            go.Bar(
                y=outperformers['Symbol'],
                x=outperformers['Relative_Strength'],
                text=[f"+{x:.1f}%" for x in outperformers['Relative_Strength']],
                textposition='auto',
                orientation='h',
                marker_color='green',
                hovertemplate='<b>%{y}</b> (%{customdata[0]})<br>Stock: %{customdata[1]:.1f}%<br>Sector: %{customdata[2]:.1f}%<br>Diff: +%{x:.1f}%<extra></extra>',
                customdata=np.stack((
                    outperformers['Sector'], 
                    outperformers['Price_Change_Pct'],
                    outperformers['Sector_Avg_Change']
                ), axis=-1),
                name="Outperformers"
            ),
            row=1, col=1
        )
        
        # Add traces for underperformers (reverse order for display)
        fig.add_trace(
            go.Bar(
                y=underperformers['Symbol'][::-1],  # Reverse order
                x=-underperformers['Relative_Strength'][::-1],  # Negate for proper display
                text=[f"-{abs(x):.1f}%" for x in underperformers['Relative_Strength'][::-1]],
                textposition='auto',
                orientation='h',
                marker_color='red',
                hovertemplate='<b>%{y}</b> (%{customdata[0]})<br>Stock: %{customdata[1]:.1f}%<br>Sector: %{customdata[2]:.1f}%<br>Diff: -%{x:.1f}%<extra></extra>',
                customdata=np.stack((
                    underperformers['Sector'][::-1], 
                    underperformers['Price_Change_Pct'][::-1],
                    underperformers['Sector_Avg_Change'][::-1]
                ), axis=-1),
                name="Underperformers"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Relative Strength vs. Sector Average",
            height=600,
            width=1000,
            showlegend=False,
            template="plotly_white"
        )
        
        # Add hyperlinks to stock symbols
        for i, symbol in enumerate(outperformers['Symbol']):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                x=0,
                y=symbol,
                xref="x",
                yref="y",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5,
                font=dict(color="blue", size=12)
            )
            
        for i, symbol in enumerate(underperformers['Symbol'][::-1]):
            url = f"https://finance.yahoo.com/quote/{symbol}"
            fig.add_annotation(
                text=f"<a href='{url}' target='_blank'>{symbol}</a>",
                x=0,
                y=symbol,
                xref="x2",
                yref="y2",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                xshift=-5,
                font=dict(color="blue", size=12)
            )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'relative_strength.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file)
        
        # Create distribution chart of relative strength
        fig_hist = px.histogram(
            self.merged_df,
            x='Relative_Strength',
            color='Sector',
            marginal='box',
            hover_data=['Symbol', 'Name', 'Price_Change_Pct', 'Sector_Avg_Change'],
            title='Distribution of Relative Strength to Sector',
            labels={'Relative_Strength': 'Outperformance / Underperformance (%)'},
            template='plotly_white'
        )
        
        # Add a reference line at x=0
        fig_hist.add_vline(x=0, line_dash='dash', line_color='black')
        
        # Save interactive HTML histogram
        html_file2 = os.path.join(self.output_dir, 'relative_strength_distribution.html')
        fig_hist.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.append(html_file2)
        
        # Store results
        results['outperformers'] = outperformers[[
            'Symbol', 'Name', 'Sector', 'Price_Change_Pct', 
            'Sector_Avg_Change', 'Relative_Strength', 'Yahoo_URL'
        ]].to_dict('records')
        
        results['underperformers'] = underperformers[[
            'Symbol', 'Name', 'Sector', 'Price_Change_Pct', 
            'Sector_Avg_Change', 'Relative_Strength', 'Yahoo_URL'
        ]].to_dict('records')
        
        results['files'] = [html_file, html_file2]
        
        logger.info(f"Relative strength analysis completed, files saved: {results['files']}")
        return results
    
    def analyze_correlation_network(self, min_correlation: float = 0.7, max_correlations: int = 100) -> Dict:
        """
        Create a correlation network of stocks
        
        Args:
            min_correlation: Minimum correlation value to include
            max_correlations: Maximum number of correlation pairs to show
            
        Returns:
            Dict: Analysis results and file paths
        """
        logger.info("Analyzing stock correlations...")
        results = {}
        
        # Calculate price change correlations
        price_changes = pd.pivot_table(
            self.merged_df, 
            values='Price_Change_Pct', 
            index=None, 
            columns='Symbol'
        )
        
        # Calculate correlation matrix
        corr_matrix = price_changes.corr()
        
        # Set self-correlations to NaN (diagonal)
        np.fill_diagonal(corr_matrix.values, np.nan)
        
        # Get pairs with strong correlations
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if not np.isnan(corr_matrix.iloc[i, j]) and abs(corr_matrix.iloc[i, j]) >= min_correlation:
                    pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Sort by correlation strength and limit to max_correlations
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        pairs = pairs[:max_correlations]
        
        if not pairs:
            logger.warning(f"No pairs with correlation >= {min_correlation} found")
            results['correlation_pairs'] = []
            results['files'] = []
            return results
        
        # Create a network graph of correlations
        edges = []
        for stock1, stock2, corr in pairs:
            # Get sector information
            stock1_sector = self.merged_df[self.merged_df['Symbol'] == stock1]['Sector'].iloc[0]
            stock2_sector = self.merged_df[self.merged_df['Symbol'] == stock2]['Sector'].iloc[0]
            
            edges.append({
                'source': stock1,
                'target': stock2,
                'correlation': corr,
                'weight': abs(corr),
                'color': 'green' if corr > 0 else 'red',
                'source_sector': stock1_sector,
                'target_sector': stock2_sector
            })
        
        # Create a DataFrame for the edges
        edges_df = pd.DataFrame(edges)
        
        # Get unique stocks in the network
        all_stocks = list(set(edges_df['source'].tolist() + edges_df['target'].tolist()))
        
        # Create a DataFrame for the nodes
        nodes = []
        for stock in all_stocks:
            stock_data = self.merged_df[self.merged_df['Symbol'] == stock].iloc[0]
            nodes.append({
                'id': stock,
                'sector': stock_data['Sector'],
                'market_cap': stock_data['Current_Market_Cap'],
                'price_change': stock_data['Price_Change_Pct'],
                'url': f"https://finance.yahoo.com/quote/{stock}"
            })
        
        nodes_df = pd.DataFrame(nodes)
        
        # Create a force-directed network graph
        # This part requires networkx and plotly
        import networkx as nx
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(
                node['id'],
                sector=node['sector'],
                market_cap=node['market_cap'],
                price_change=node['price_change'],
                url=node['url']
            )
        
        # Add edges
        for edge in edges:
            G.add_edge(
                edge['source'],
                edge['target'],
                weight=edge['weight'],
                correlation=edge['correlation'],
                color=edge['color']
            )
        
        # Create positions for nodes using a spring layout
        pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)
        
        # Create a Plotly figure for the network
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        edge_texts = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.append(edge[2]['color'])
            edge_widths.append(edge[2]['weight'] * 3)  # Scale width by correlation
            
            # Create hover text
            corr_val = edge[2]['correlation']
            corr_text = f"{corr_val:.2f}"
            edge_texts.append(f"{edge[0]} — {edge[1]}: {corr_text}")
        
        # Create the scatter plot for edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='text',
            mode='lines',
            line_width=edge_widths,
            line_color=edge_colors,
            text=edge_texts
        )
        
        # Create node traces by sector
        sectors = nodes_df['sector'].unique()
        node_traces = []
        
        for sector in sectors:
            sector_nodes = nodes_df[nodes_df['sector'] == sector]
            node_x = []
            node_y = []
            node_symbols = []
            node_sizes = []
            node_colors = []
            node_texts = []
            node_urls = []
            
            for _, node in sector_nodes.iterrows():
                x, y = pos[node['id']]
                node_x.append(x)
                node_y.append(y)
                
                # Node size based on market cap 
                size = np.sqrt(node['market_cap']) / 5000  # Scale for visibility
                node_sizes.append(max(10, min(50, size)))  # Constrain size
                
                # Color by price change
                node_colors.append(node['price_change'])
                
                # Create hover text
                node_text = f"{node['id']}<br>Sector: {node['sector']}<br>Change: {node['price_change']:.2f}%"
                node_texts.append(node_text)
                
                # URL for clickable nodes
                node_urls.append(node['url'])
            
            # Create scatter plot for this sector's nodes
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Price Change (%)") if sector == sectors[0] else None,
                    line=dict(width=1, color='black')
                ),
                text=sector_nodes['id'],
                textposition="top center",
                hovertext=node_texts,
                hoverinfo='text',
                name=sector
            )
            
            node_traces.append(node_trace)
        
        # Create the figure
        fig = go.Figure(data=[edge_trace] + node_traces)
        
        # Update the layout
        fig.update_layout(
            title=f"Stock Correlation Network (|r| ≥ {min_correlation})",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            height=800,
            width=1000,
            template='plotly_white'
        )
        
        # Save interactive HTML figure
        html_file = os.path.join(self.output_dir, 'correlation_network.html')
        fig.write_html(html_file, include_plotlyjs='cdn')
        
        # Create a heatmap of sector correlations
        sector_price_changes = self.merged_df.groupby('Sector')['Price_Change_Pct'].mean().reset_index()
        sector_pivot = pd.pivot_table(sector_price_changes, values='Price_Change_Pct', index='Sector', columns=None)
        sector_corr = sector_pivot.T.corr()
        
        # Create heatmap figure
        fig_heatmap = px.imshow(
            sector_corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Sector Correlation Heatmap',
            height=700,
            width=700
        )
        
        # Save interactive HTML heatmap
        html_file2 = os.path.join(self.output_dir, 'sector_correlation.html')
        fig_heatmap.write_html(html_file2, include_plotlyjs='cdn')
        
        # Track created files
        self.created_files.extend([html_file, html_file2])
        
        # Store results
        results['correlation_pairs'] = [{
            'stock1': pair[0],
            'stock2': pair[1],
            'correlation': pair[2],
            'stock1_url': f"https://finance.yahoo.com/quote/{pair[0]}",
            'stock2_url': f"https://finance.yahoo.com/quote/{pair[1]}"
        } for pair in pairs]
        
        results['files'] = [html_file, html_file2]
        
        logger.info(f"Correlation analysis completed, files saved: {results['files']}")
        return results
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report
        
        Returns:
            Dict: Summary report and file paths
        """
        logger.info("Generating summary report...")
        results = {}
        
        # Overall market summary
        total_prev_cap = self.merged_df['Previous_Market_Cap'].sum()
        total_curr_cap = self.merged_df['Current_Market_Cap'].sum()
        total_change = total_curr_cap - total_prev_cap
        total_change_pct = (total_change / total_prev_cap) * 100
        
        avg_price_change = self.merged_df['Price_Change_Pct'].mean()
        median_price_change = self.merged_df['Price_Change_Pct'].median()
        
        stocks_up = (self.merged_df['Price_Change_Pct'] > 0).sum()
        stocks_down = (self.merged_df['Price_Change_Pct'] < 0).sum()
        stocks_unchanged = len(self.merged_df) - stocks_up - stocks_down
        
        # Create summary HTML
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Market Snapshot Analysis Summary</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .summary-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .summary-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .summary-label {{ color: #7f8c8d; font-size: 14px; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Market Snapshot Analysis</h1>
            <p>Comparison between <strong>{self.prev_label}</strong> and <strong>{self.curr_label}</strong></p>
            
            <div class="section">
                <h2>Market Overview</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Total Market Cap Change</div>
                        <div class="summary-value {('positive' if total_change > 0 else 'negative')}">${total_change/1e9:.1f}B ({total_change_pct:.1f}%)</div>
                        <div>{self.prev_label}: ${total_prev_cap/1e9:.1f}B → {self.curr_label}: ${total_curr_cap/1e9:.1f}B</div>
                    </div>
                    
                    <div class="summary-card">
                        <div class="summary-label">Average Stock Price Change</div>
                        <div class="summary-value {('positive' if avg_price_change > 0 else 'negative')}">{avg_price_change:.2f}%</div>
                        <div>Median: {median_price_change:.2f}%</div>
                    </div>
                    
                    <div class="summary-card">
                        <div class="summary-label">Market Breadth</div>
                        <div class="summary-value">{stocks_up} ↑ / {stocks_down} ↓ / {stocks_unchanged} =</div>
                        <div>{stocks_up/len(self.merged_df)*100:.1f}% of stocks up, {stocks_down/len(self.merged_df)*100:.1f}% down</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Sector Performance</h2>
                <table>
                    <tr>
                        <th>Sector</th>
                        <th>Avg Price Change (%)</th>
                        <th>Total Market Cap ($B)</th>
                        <th>Market Cap Change ($B)</th>
                        <th>Market Cap Change (%)</th>
                        <th># Stocks</th>
                    </tr>
        """
        
        # Add sector data
        if self.sector_analysis is None:
            self.analyze_sectors()
            
        for _, row in self.sector_analysis.sort_values('Avg_Price_Change_Pct', ascending=False).iterrows():
            price_class = 'positive' if row['Avg_Price_Change_Pct'] > 0 else 'negative'
            cap_class = 'positive' if row['Market_Cap_Change'] > 0 else 'negative'
            
            html_content += f"""
                <tr>
                    <td>{row['Sector']}</td>
                    <td class="{price_class}">{row['Avg_Price_Change_Pct']:.2f}%</td>
                    <td>${row['Current_Market_Cap']/1e9:.1f}B</td>
                    <td class="{cap_class}">${row['Market_Cap_Change']/1e9:.1f}B</td>
                    <td class="{cap_class}">{row['Market_Cap_Change_Pct']:.2f}%</td>
                    <td>{row['Stock_Count']}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Top Performers</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Name</th>
                        <th>Sector</th>
                        <th>Previous Price</th>
                        <th>Current Price</th>
                        <th>Change (%)</th>
                        <th>Market Cap ($B)</th>
                    </tr>
        """
        
        # Add top gainers
        top_gainers = self.merged_df.sort_values('Price_Change_Pct', ascending=False).head(10)
        for _, row in top_gainers.iterrows():
            html_content += f"""
                <tr>
                    <td><a href="{row['Yahoo_URL']}" target="_blank">{row['Symbol']}</a></td>
                    <td>{row['Name']}</td>
                    <td>{row['Sector']}</td>
                    <td>${row['Previous_Last_Sale']:.2f}</td>
                    <td>${row['Current_Last_Sale']:.2f}</td>
                    <td class="positive">{row['Price_Change_Pct']:.2f}%</td>
                    <td>${row['Current_Market_Cap']/1e9:.2f}B</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Bottom Performers</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Name</th>
                        <th>Sector</th>
                        <th>Previous Price</th>
                        <th>Current Price</th>
                        <th>Change (%)</th>
                        <th>Market Cap ($B)</th>
                    </tr>
        """
        
        # Add top losers
        top_losers = self.merged_df.sort_values('Price_Change_Pct', ascending=True).head(10)
        for _, row in top_losers.iterrows():
            html_content += f"""
                <tr>
                    <td><a href="{row['Yahoo_URL']}" target="_blank">{row['Symbol']}</a></td>
                    <td>{row['Name']}</td>
                    <td>{row['Sector']}</td>
                    <td>${row['Previous_Last_Sale']:.2f}</td>
                    <td>${row['Current_Last_Sale']:.2f}</td>
                    <td class="negative">{row['Price_Change_Pct']:.2f}%</td>
                    <td>${row['Current_Market_Cap']/1e9:.2f}B</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Volume Spikes</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Name</th>
                        <th>Sector</th>
                        <th>Previous Volume</th>
                        <th>Current Volume</th>
                        <th>Volume Change (%)</th>
                        <th>Price Change (%)</th>
                    </tr>
        """
        
        # Add volume spikes
        volume_spikes = self.merged_df[self.merged_df['Volume_Change_Pct'] > 100].sort_values('Volume_Change_Pct', ascending=False).head(10)
        for _, row in volume_spikes.iterrows():
            price_class = 'positive' if row['Price_Change_Pct'] > 0 else 'negative'
            html_content += f"""
                <tr>
                    <td><a href="{row['Yahoo_URL']}" target="_blank">{row['Symbol']}</a></td>
                    <td>{row['Name']}</td>
                    <td>{row['Sector']}</td>
                    <td>{row['Previous_Volume']:,.0f}</td>
                    <td>{row['Current_Volume']:,.0f}</td>
                    <td>{row['Volume_Change_Pct']:.2f}%</td>
                    <td class="{price_class}">{row['Price_Change_Pct']:.2f}%</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Available Analysis Files</h2>
                <table>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Files</th>
                    </tr>
        """
        
        # Create a simple organization of files by type
        file_types = {
            'Price Movement': [f for f in self.created_files if 'price' in f],
            'Volume Analysis': [f for f in self.created_files if 'volume' in f],
            'Sector Analysis': [f for f in self.created_files if 'sector' in f],
            'Market Cap': [f for f in self.created_files if 'market_cap' in f or 'cap' in f],
            'IPO Analysis': [f for f in self.created_files if 'ipo' in f],
            'Correlation': [f for f in self.created_files if 'correlation' in f or 'network' in f],
            'Relative Strength': [f for f in self.created_files if 'relative' in f or 'strength' in f],
            'Other': []
        }
        
        # Add any files not caught above to Other
        for file in self.created_files:
            if not any(file in files for files in file_types.values()):
                file_types['Other'].append(file)
        
        # Add file links to summary
        for analysis_type, files in file_types.items():
            if files:
                links = []
                for file in files:
                    file_name = os.path.basename(file)
                    file_path = os.path.relpath(file, self.output_dir)
                    links.append(f'<a href="{file_path}" target="_blank">{file_name}</a>')
                
                html_content += f"""
                    <tr>
                        <td>{analysis_type}</td>
                        <td>{', '.join(links)}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            <footer style="margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; color: #7f8c8d; font-size: 12px;">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        
        # Write the HTML file
        summary_file = os.path.join(self.output_dir, 'summary_report.html')
        with open(summary_file, 'w') as f:
            f.write(html_content)
        
        # Track created files
        self.created_files.append(summary_file)
        
        # Store results
        results['summary_report'] = summary_file
        results['files'] = [summary_file]
        
        logger.info(f"Summary report generated and saved to {summary_file}")
        return results
    
    def run_all_analyses(self) -> Dict:
        """
        Run all analysis functions and create a comprehensive report
        
        Returns:
            Dict: All analysis results and file paths
        """
        logger.info("Running all analyses...")
        results = {}
        
        # Run each analysis
        results['price_movement'] = self.analyze_price_movement()
        results['volume_spikes'] = self.analyze_volume_spikes()
        results['sectors'] = self.analyze_sectors()
        results['market_cap'] = self.analyze_market_cap()
        results['ipos'] = self.analyze_ipos()
        results['relative_strength'] = self.analyze_relative_strength()
        
        try:
            # Network analysis requires networkx, which might not be installed
            import networkx
            results['correlation_network'] = self.analyze_correlation_network()
        except ImportError:
            logger.warning("NetworkX not available, skipping correlation network analysis")
            results['correlation_network'] = {'files': []}
        
        # Generate summary report
        results['summary'] = self.generate_summary_report()
        
        # Return combined results
        return results


def detect_file_type(file_path):
    """
    Auto-detect whether file is CSV or Excel
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: 'csv' or 'excel'
    """
    if file_path.lower().endswith('.csv'):
        return 'csv'
    elif file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
        return 'excel'
    else:
        # Try to infer from content
        with open(file_path, 'r', errors='ignore') as f:
            header = f.readline()
            if header.count(',') > 3:  # If multiple commas, likely CSV
                return 'csv'
        return 'excel'  # Default to excel if can't determine


def load_snapshot(file_path):
    """
    Load a snapshot file from CSV or Excel
    
    Args:
        file_path: Path to the file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    file_type = detect_file_type(file_path)
    if file_type == 'csv':
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def main():
    """Command-line interface for the module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Market Snapshot Comparison Tool")
    parser.add_argument("previous_file", help="Path to previous snapshot file (CSV or Excel)")
    parser.add_argument("current_file", help="Path to current snapshot file (CSV or Excel)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Output directory for analysis files (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    try:
        # Run the analysis
        analyzer = SnapshotAnalyzer(args.previous_file, args.current_file, args.output_dir)
        results = analyzer.run_all_analyses()
        
        # Print summary of generated files
        print(f"\nAnalysis complete! All files saved to: {args.output_dir}")
        print(f"Summary report: {results['summary']['summary_report']}")
        print(f"Total files generated: {len(analyzer.created_files)}")
        
        # Open the summary report in a browser
        summary_file = results['summary']['summary_report']
        print(f"\nOpening summary report in your browser: {summary_file}")
        webbrowser.open(f"file://{os.path.abspath(summary_file)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in snapshot analysis: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        return 1


if __name__ == "__main__":
    exit(main())