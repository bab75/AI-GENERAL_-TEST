import base64
import io
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
import tempfile
import pdfkit
import time
from datetime import datetime

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    
    Parameters:
    -----------
    object_to_download : pandas.DataFrame, str, or bytes
        The object to be downloaded
    download_filename : str
        filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text : str
        Text to display for download link
    
    Returns:
    --------
    str
        HTML download link as a string
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=True)

    # Some strings <-> bytes conversions necessary here
    if isinstance(object_to_download, str):
        object_to_download = object_to_download.encode()
    
    b64 = base64.b64encode(object_to_download).decode()
    
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def export_to_csv(df, filename="stock_data.csv"):
    """
    Export DataFrame to CSV file for download.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    filename : str, default "stock_data.csv"
        Filename for the download
    
    Returns:
    --------
    str
        CSV string representation
    """
    csv = df.to_csv(index=True)
    return csv, filename

def export_to_excel(df, filename="stock_data.xlsx", include_stats=False, stats_df=None):
    """
    Export DataFrame to Excel file for download.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    filename : str, default "stock_data.xlsx"
        Filename for the download
    include_stats : bool, default False
        Whether to include stats in a separate sheet
    stats_df : pandas.DataFrame, default None
        DataFrame containing statistics to include
    
    Returns:
    --------
    bytes
        Excel file as bytes
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Data", index=True)
        
        if include_stats and stats_df is not None:
            stats_df.to_excel(writer, sheet_name="Statistics", index=False)
    
    return buffer.getvalue(), filename

def export_to_json(df, filename="stock_data.json"):
    """
    Export DataFrame to JSON file for download.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    filename : str, default "stock_data.json"
        Filename for the download
    
    Returns:
    --------
    str
        JSON string representation
    """
    # Convert datetime index to strings
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
    
    json_data = df_copy.to_json(orient="index")
    return json_data, filename

def export_plot_to_html(fig, filename="stock_chart.html"):
    """
    Export Plotly figure to interactive HTML file for download.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Plotly figure to export
    filename : str, default "stock_chart.html"
        Filename for the download
    
    Returns:
    --------
    str
        HTML string representation
    """
    html_str = fig.to_html(include_plotlyjs="cdn")
    return html_str, filename

def create_hyperlinked_df(df, symbol_col='Symbol', base_url='https://finance.yahoo.com/quote/'):
    """
    Create a DataFrame with hyperlinked symbols that point to Yahoo Finance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a symbol column
    symbol_col : str, default 'Symbol'
        Column name containing the stock symbols
    base_url : str, default 'https://finance.yahoo.com/quote/'
        Base URL for the hyperlinks
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with hyperlinked symbols
    """
    df_html = df.copy()
    
    # Create hyperlinks for the symbols
    if symbol_col in df.columns:
        df_html[symbol_col] = df_html[symbol_col].apply(
            lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>' if pd.notnull(x) else ''
        )
    
    return df_html


def style_dataframe(df, performance_col=None, profit_loss_col=None):
    """
    Style a DataFrame with appropriate coloring for performance metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to style
    performance_col : str, default None
        Column name containing performance percentages
    profit_loss_col : str, default None
        Column name containing profit/loss status
    
    Returns:
    --------
    pandas.io.formats.style.Styler
        Styled DataFrame
    """
    styled_df = df.copy()
    
    # Apply styling based on performance column
    if performance_col and performance_col in styled_df.columns:
        styled_df = styled_df.style.apply(
            lambda row: [
                'background-color: #c6ecc6' if cell > 2 else
                'background-color: #ffcccb' if cell < -2 else
                'background-color: #f5f5f5'
                for cell in row
            ], 
            subset=[performance_col]
        )
    
    # Apply styling based on profit/loss column
    if profit_loss_col and profit_loss_col in styled_df.columns:
        styled_df = styled_df.style.apply(
            lambda row: [
                'background-color: #c6ecc6' if cell == 'Profit' else
                'background-color: #ffcccb' if cell == 'Loss' else
                'background-color: #f5f5f5'
                for cell in row
            ], 
            subset=[profit_loss_col]
        )
    
    return styled_df


def export_to_pdf_with_charts(df, ticker_data, charts, filename="stock_analysis.pdf"):
    """
    Export DataFrame and charts to an interactive PDF file for download.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    ticker_data : dict
        Dictionary containing ticker-specific data
    charts : list
        List of Plotly figures to include
    filename : str, default "stock_analysis.pdf"
        Filename for the download
    
    Returns:
    --------
    bytes
        PDF file as bytes
    """
    # Create a temporary directory for HTML files
    temp_dir = tempfile.mkdtemp()
    html_path = os.path.join(temp_dir, "ticker_report.html")
    pdf_path = os.path.join(temp_dir, filename)
    
    # Create HTML content with DataFrame and interactive charts
    html_content = f"""
    <html>
    <head>
        <title>Stock Analysis Report - {ticker_data.get('Symbol', '')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .chart-container {{ margin: 20px 0; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Stock Analysis Report - {ticker_data.get('Symbol', '')}: {ticker_data.get('Name', '')}</h1>
        <div class="container">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Previous Price</th>
                    <td>${ticker_data.get('Previous_Last_Sale', 0):.2f}</td>
                    <th>Current Price</th>
                    <td>${ticker_data.get('Current_Last_Sale', 0):.2f}</td>
                </tr>
                <tr>
                    <th>Change</th>
                    <td class="{('positive' if ticker_data.get('Price_Change', 0) > 0 else 'negative')}">
                        ${ticker_data.get('Price_Change', 0):.2f} ({ticker_data.get('Price_Change_Pct', 0):.2f}%)
                    </td>
                    <th>Status</th>
                    <td class="{('positive' if ticker_data.get('Profit_Loss_Status', '') == 'Profit' else 'negative')}">
                        {ticker_data.get('Profit_Loss_Status', '')}
                    </td>
                </tr>
                <tr>
                    <th>Previous Volume</th>
                    <td>{ticker_data.get('Previous_Volume', 0):,}</td>
                    <th>Current Volume</th>
                    <td>{ticker_data.get('Current_Volume', 0):,}</td>
                </tr>
                <tr>
                    <th>Volume Change</th>
                    <td>{ticker_data.get('Volume_Change', 0):,} ({ticker_data.get('Volume_Change_Pct', 0):.2f}%)</td>
                    <th>Volume Trend</th>
                    <td>{ticker_data.get('Volume_Trend', '')}</td>
                </tr>
                <tr>
                    <th>Sector</th>
                    <td>{ticker_data.get('Sector', '')}</td>
                    <th>Industry</th>
                    <td>{ticker_data.get('Industry', '')}</td>
                </tr>
                <tr>
                    <th>Previous Market Cap</th>
                    <td>${ticker_data.get('Previous_Market_Cap', 0)/1e9:.2f}B</td>
                    <th>Current Market Cap</th>
                    <td>${ticker_data.get('Current_Market_Cap', 0)/1e9:.2f}B</td>
                </tr>
                <tr>
                    <th>Performance Category</th>
                    <td colspan="3">{ticker_data.get('Performance_Category', '')}</td>
                </tr>
            </table>
            
            <h2>Analysis Data</h2>
            {df.to_html(escape=False, index=False)}
        </div>
    """
    
    # Add interactive charts
    for i, fig in enumerate(charts):
        chart_div = f'chart{i}'
        html_content += f"""
        <div class="container">
            <h2>Chart {i+1}</h2>
            <div id="{chart_div}" class="chart-container" style="height: 500px;"></div>
            <script>
                var plotlyData = {fig.to_json()};
                Plotly.newPlot('{chart_div}', 
                    plotlyData.data, 
                    plotlyData.layout, 
                    {{responsive: true, displayModeBar: true}}
                );
            </script>
        </div>
        """
    
    # Add final HTML tags
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to temporary file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Convert HTML to PDF
    try:
        pdfkit.from_file(html_path, pdf_path)
        
        # Read the PDF file
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
            
        # Clean up temporary files
        os.remove(html_path)
        os.remove(pdf_path)
        os.rmdir(temp_dir)
        
        return pdf_bytes, filename
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None, filename


def create_summary_stats(df, ticker):
    """
    Create a DataFrame with summary statistics for the given stock data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing summary statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate daily returns
    df_with_returns = df.copy()
    df_with_returns['Daily_Return'] = df_with_returns['Close'].pct_change()
    
    # Basic stats
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    days = (df.index[-1] - df.index[0]).days
    
    price_start = df['Close'].iloc[0]
    price_end = df['Close'].iloc[-1]
    price_change = price_end - price_start
    price_change_pct = (price_change / price_start) * 100
    
    # Create stats DataFrame
    stats_data = {
        'Metric': [
            'Ticker', 'Start Date', 'End Date', 'Days', 
            'Start Price', 'End Price', 'Price Change', 'Price Change (%)',
            'Min Price', 'Max Price', 'Average Price', 
            'Average Volume', 'Total Volume',
            'Daily Return (Avg)', 'Daily Return (Min)', 'Daily Return (Max)',
            'Daily Return (Std)', 'Annualized Volatility'
        ],
        'Value': [
            ticker, start_date, end_date, days,
            f"${price_start:.2f}", f"${price_end:.2f}", 
            f"${price_change:.2f}", f"{price_change_pct:.2f}%",
            f"${df['Low'].min():.2f}", f"${df['High'].max():.2f}", 
            f"${df['Close'].mean():.2f}",
            f"{df['Volume'].mean():.0f}", f"{df['Volume'].sum():.0f}",
            f"{df_with_returns['Daily_Return'].mean() * 100:.2f}%",
            f"{df_with_returns['Daily_Return'].min() * 100:.2f}%", 
            f"{df_with_returns['Daily_Return'].max() * 100:.2f}%",
            f"{df_with_returns['Daily_Return'].std() * 100:.2f}%",
            f"{df_with_returns['Daily_Return'].std() * (252 ** 0.5) * 100:.2f}%"
        ]
    }
    
    return pd.DataFrame(stats_data)
