"""
Enhanced Report Generation Module

This module provides advanced report generation capabilities using FPDF,
a pure Python PDF generation library with no external dependencies.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
from io import BytesIO
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockAnalysisReport(FPDF):
    """
    Enhanced PDF report generator using FPDF with no external dependencies.
    Replaces the pdfkit implementation that required wkhtmltopdf.
    """
    
    def __init__(self, title="Stock Analysis Report", orientation='P', unit='mm', format='A4'):
        """
        Initialize the PDF report with customized settings.
        
        Args:
            title: Report title
            orientation: 'P' for portrait or 'L' for landscape
            unit: Measurement unit ('mm', 'cm', 'in')
            format: Page format ('A4', 'Letter', 'Legal', etc.)
        """
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.title = title
        self.set_author("Stock Analysis Platform")
        self.set_title(title)
        
        # Add custom fonts and styles
        self.add_page()
        self.set_font('Arial', 'B', 16)
        
    def header(self):
        """Add a header to each page"""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        """Add a footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 14)
        self.ln(10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def section_title(self, title):
        """Add a section title"""
        self.set_font('Arial', 'B', 12)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
        
    def body_text(self, text):
        """Add body text"""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(5)
        
    def add_table(self, data, headers=None, col_widths=None):
        """
        Add a table to the report
        
        Args:
            data: List of lists containing table data
            headers: List of column headers
            col_widths: List of column widths
        """
        self.set_font('Arial', '', 10)
        
        # Calculate column widths if not provided
        if col_widths is None:
            page_width = self.w - 2 * self.l_margin
            col_widths = [page_width / len(data[0])] * len(data[0])
            
        # Add headers if provided
        if headers:
            self.set_font('Arial', 'B', 10)
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 7, str(header), 1, 0, 'C')
            self.ln()
            
        # Add data
        self.set_font('Arial', '', 10)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), 1, 0, 'C')
            self.ln()
        self.ln(5)
        
    def add_dataframe(self, df, max_rows=20):
        """
        Add a pandas DataFrame as a table
        
        Args:
            df: Pandas DataFrame
            max_rows: Maximum number of rows to display
        """
        if len(df) > max_rows:
            df = df.head(max_rows)
            truncated = True
        else:
            truncated = False
            
        data = df.values.tolist()
        headers = df.columns.tolist()
        
        # Calculate column widths based on content
        col_widths = []
        page_width = self.w - 2 * self.l_margin
        for col in headers:
            # Use 1/len(headers) of page width as the default
            col_widths.append(page_width / len(headers))
            
        self.add_table(data, headers, col_widths)
        
        if truncated:
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f'Note: Table truncated to {max_rows} rows.', 0, 1, 'L')
            
    def add_matplotlib_figure(self, fig):
        """
        Add a matplotlib figure to the report
        
        Args:
            fig: Matplotlib figure object
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        img_width = 180  # mm
        img_height = 120  # mm
        
        self.image(buf, x=10, y=None, w=img_width, h=img_height)
        self.ln(5)
        
    def add_plotly_figure(self, fig, width=180, height=120):
        """
        Add a plotly figure to the report
        
        Args:
            fig: Plotly figure object
            width: Width in mm
            height: Height in mm
        """
        # Convert plotly figure to a static image
        img_bytes = fig.to_image(format="png", width=width*30, height=height*30)
        
        # Save to a BytesIO object
        buf = BytesIO(img_bytes)
        
        # Add to PDF
        self.image(buf, x=10, y=None, w=width, h=height)
        self.ln(5)
        
    def add_plotly_from_html(self, html_file, width=180, height=120):
        """
        Take screenshot of an HTML file (containing a plotly figure) and add to report
        
        Args:
            html_file: Path to HTML file
            width: Width in mm
            height: Height in mm
        """
        try:
            # Use matplotlib to render HTML - a simple alternative
            # For a production version, a proper HTML renderer would be needed
            self.body_text(f"Figure reference: {html_file}")
            self.ln(5)
        except Exception as e:
            logger.error(f"Error adding plotly from HTML: {str(e)}")
            self.body_text(f"Error rendering figure from {html_file}")
            
    def add_summary_page(self, title, summary_text, key_metrics=None):
        """
        Add a summary page to the report
        
        Args:
            title: Page title
            summary_text: Summary text
            key_metrics: Dictionary of key metrics to highlight
        """
        self.add_page()
        self.chapter_title(title)
        self.body_text(summary_text)
        
        if key_metrics:
            self.ln(5)
            self.section_title("Key Metrics")
            
            # Format key metrics as a table
            data = [[k, v] for k, v in key_metrics.items()]
            headers = ["Metric", "Value"]
            col_widths = [100, 80]
            self.add_table(data, headers, col_widths)

def generate_snapshot_report(analyzer, output_file, include_figures=True):
    """
    Generate a comprehensive PDF report for snapshot comparison analysis
    
    Args:
        analyzer: SnapshotAnalyzer instance with completed analyses
        output_file: Path to save the PDF report
        include_figures: Whether to include figures in the report
        
    Returns:
        str: Path to the generated PDF file
    """
    report = StockAnalysisReport(title=f"Stock Market Snapshot Analysis Report")
    
    # Add summary information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.body_text(f"Report generated on: {timestamp}")
    report.body_text(f"Previous snapshot: {os.path.basename(analyzer.previous_file)}")
    report.body_text(f"Current snapshot: {os.path.basename(analyzer.current_file)}")
    
    # Add overall market summary
    report.chapter_title("Market Overview")
    
    if hasattr(analyzer, 'summary_stats'):
        summary_text = "\n".join([
            f"Total stocks analyzed: {analyzer.summary_stats.get('total_stocks', 'N/A')}",
            f"Average price change: {analyzer.summary_stats.get('avg_price_change_pct', 'N/A'):.2f}%",
            f"Average volume change: {analyzer.summary_stats.get('avg_volume_change_pct', 'N/A'):.2f}%",
            f"Total market cap change: {analyzer.summary_stats.get('total_market_cap_change_pct', 'N/A'):.2f}%",
        ])
        report.body_text(summary_text)
    
    # Add price movement analysis
    report.chapter_title("Price Movement Analysis")
    
    if hasattr(analyzer, 'price_movement_results'):
        # Add top gainers table
        report.section_title("Top Gainers")
        if 'top_gainers' in analyzer.price_movement_results:
            top_gainers = pd.DataFrame(analyzer.price_movement_results['top_gainers'])
            if not top_gainers.empty:
                # Format percentage columns
                for col in top_gainers.columns:
                    if 'Pct' in col or '%' in col:
                        top_gainers[col] = top_gainers[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                report.add_dataframe(top_gainers)
        
        # Add top losers table
        report.section_title("Top Losers")
        if 'top_losers' in analyzer.price_movement_results:
            top_losers = pd.DataFrame(analyzer.price_movement_results['top_losers'])
            if not top_losers.empty:
                # Format percentage columns
                for col in top_losers.columns:
                    if 'Pct' in col or '%' in col:
                        top_losers[col] = top_losers[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                report.add_dataframe(top_losers)
    
    # Add volume analysis
    report.chapter_title("Volume Analysis")
    
    if hasattr(analyzer, 'volume_results'):
        # Add volume spikes table
        report.section_title("Notable Volume Spikes")
        if 'volume_spikes' in analyzer.volume_results:
            volume_spikes = pd.DataFrame(analyzer.volume_results['volume_spikes'])
            if not volume_spikes.empty:
                # Format percentage columns
                for col in volume_spikes.columns:
                    if 'Pct' in col or '%' in col:
                        volume_spikes[col] = volume_spikes[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                report.add_dataframe(volume_spikes)
    
    # Add sector analysis
    report.chapter_title("Sector Analysis")
    
    if hasattr(analyzer, 'sector_results'):
        # Add sector performance table
        report.section_title("Sector Performance")
        if 'sector_performance' in analyzer.sector_results:
            sector_performance = pd.DataFrame(analyzer.sector_results['sector_performance'])
            if not sector_performance.empty:
                # Format percentage columns
                for col in sector_performance.columns:
                    if 'Pct' in col or '%' in col:
                        sector_performance[col] = sector_performance[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                report.add_dataframe(sector_performance)
    
    # Finalize and save the report
    try:
        report.output(output_file)
        logger.info(f"PDF report successfully generated and saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def generate_technical_analysis_report(ticker, data, indicators, output_file):
    """
    Generate a PDF report for technical analysis
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        indicators: Dictionary of technical indicators
        output_file: Path to save the PDF report
        
    Returns:
        str: Path to the generated PDF file
    """
    report = StockAnalysisReport(title=f"Technical Analysis Report: {ticker}")
    
    # Add summary information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.body_text(f"Report generated on: {timestamp}")
    report.body_text(f"Analysis period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Add stock overview
    report.chapter_title("Stock Overview")
    
    # Calculate basic statistics
    latest_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
    price_change_pct = price_change / data['Close'].iloc[0] * 100
    avg_volume = data['Volume'].mean()
    
    overview_text = "\n".join([
        f"Latest price: ${latest_price:.2f}",
        f"Price change: {'📈' if price_change >= 0 else '📉'} ${abs(price_change):.2f} ({price_change_pct:.2f}%)",
        f"Average volume: {avg_volume:.0f}",
        f"52-week high: ${data['High'].max():.2f}",
        f"52-week low: ${data['Low'].min():.2f}",
    ])
    report.body_text(overview_text)
    
    # Add price table
    report.section_title("Recent Price Data")
    recent_data = data.tail(10).copy()
    # Format the DataFrame for display
    recent_data = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    recent_data.index = recent_data.index.strftime('%Y-%m-%d')
    recent_data = recent_data.reset_index().rename(columns={'index': 'Date'})
    report.add_dataframe(recent_data)
    
    # Add technical indicators section
    report.chapter_title("Technical Indicators")
    
    for indicator_name, indicator_data in indicators.items():
        report.section_title(indicator_name)
        
        # Add description if available
        if 'description' in indicator_data:
            report.body_text(indicator_data['description'])
        
        # Add indicator values if available
        if 'values' in indicator_data and not indicator_data['values'].empty:
            # Get the last few values
            last_values = indicator_data['values'].tail(5).copy()
            if isinstance(last_values, pd.Series):
                last_values = pd.DataFrame(last_values)
            
            last_values.index = last_values.index.strftime('%Y-%m-%d')
            last_values = last_values.reset_index().rename(columns={'index': 'Date'})
            report.add_dataframe(last_values)
        
        # Add interpretation if available
        if 'interpretation' in indicator_data:
            report.section_title("Interpretation")
            report.body_text(indicator_data['interpretation'])
    
    # Add trading signals if available
    if any('signal' in indicator_data for indicator_data in indicators.values()):
        report.chapter_title("Trading Signals")
        
        signals_data = []
        for indicator_name, indicator_data in indicators.items():
            if 'signal' in indicator_data:
                signals_data.append([indicator_name, indicator_data['signal']])
        
        if signals_data:
            report.add_table(signals_data, headers=["Indicator", "Signal"])
    
    # Finalize and save the report
    try:
        report.output(output_file)
        logger.info(f"Technical analysis PDF report successfully generated and saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating technical analysis PDF report: {str(e)}")
        return None

def generate_comparison_report(tickers, data_dict, analysis_results, output_file):
    """
    Generate a PDF report for multi-stock comparison
    
    Args:
        tickers: List of stock ticker symbols
        data_dict: Dictionary with ticker symbols as keys and DataFrames as values
        analysis_results: Dictionary with analysis results
        output_file: Path to save the PDF report
        
    Returns:
        str: Path to the generated PDF file
    """
    report = StockAnalysisReport(title=f"Multi-Stock Comparison Report")
    
    # Add summary information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.body_text(f"Report generated on: {timestamp}")
    report.body_text(f"Stocks compared: {', '.join(tickers)}")
    
    # Add performance comparison
    report.chapter_title("Performance Comparison")
    
    if 'performance_metrics' in analysis_results:
        perf_metrics = pd.DataFrame(analysis_results['performance_metrics'])
        report.add_dataframe(perf_metrics)
    
    # Add correlation analysis
    report.chapter_title("Correlation Analysis")
    
    if 'correlation_matrix' in analysis_results:
        corr_matrix = pd.DataFrame(analysis_results['correlation_matrix'])
        report.add_dataframe(corr_matrix)
    
    # Add risk metrics
    report.chapter_title("Risk Metrics")
    
    if 'risk_metrics' in analysis_results:
        risk_metrics = pd.DataFrame(analysis_results['risk_metrics'])
        report.add_dataframe(risk_metrics)
    
    # Finalize and save the report
    try:
        report.output(output_file)
        logger.info(f"Comparison PDF report successfully generated and saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating comparison PDF report: {str(e)}")
        return None