import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import compare_snapshots
from compare_snapshots import SnapshotAnalyzer
from streamlit_extras.let_it_rain import rain
import zipfile
from io import BytesIO

st.set_page_config(
    page_title="Snapshot Comparison - Stock Analysis Platform",
    page_icon="📸",
    layout="wide"
)

def main():
    st.title("📸 Market Snapshot Comparison")
    
    # Main description
    st.markdown("""
    Compare two snapshots of the market to identify trends, changes, and notable market movements.
    """)
    
    # Help expander
    with st.expander("ℹ️ How to use Snapshot Comparison"):
        st.markdown("""
        ### Snapshot Comparison Help
        
        This tool allows you to compare two market snapshots from different time periods to identify trends, changes, and notable market movements.
        
        **Required Data Format:**
        Upload two CSV or Excel files containing stock data. Each file should contain these columns:
        - Symbol, Name, Last Sale, Net Change, % Change, Market Cap, Country, IPO Year, Volume, Sector, Industry
        
        **Steps to Use:**
        1. Upload a previous snapshot (older time period)
        2. Upload a current snapshot (newer time period)
        3. Configure filters and analysis options (optional)
        4. Click "Run Comparison Analysis"
        5. Explore the generated visualizations and results
        
        **Analysis Types:**
        - Price Movement Analysis: Identify biggest gainers and losers
        - Volume Analysis: Detect unusual volume activity
        - Sector Performance: Compare sector performance
        - Market Cap Analysis: Track changes in company valuations
        - IPO Analysis: Examine recent IPO performance
        - Relative Strength: Identify stocks outperforming their sectors
        
        **Tips:**
        - For best results, use market snapshots that are separated by at least a week
        - Use the Advanced Filters to focus on specific market segments
        - Export individual stock reports for deeper analysis
        
        [View Full User Guide](docs/user_guide.md)
        """)
        
    # Data format reminder card
    # Removed Required Columns section from sidebar as requested
    
    # Visual separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Output directory for generated visualizations
    output_dir = "snapshot_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize session state for file uploads
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'temp_file_paths' not in st.session_state:
        st.session_state.temp_file_paths = {'previous': None, 'current': None}
    
    # Function to save uploaded file to temp location and store path in session state
    def save_uploaded_file(uploaded_file, file_type):
        if uploaded_file is not None:
            # Create temp directory if it doesn't exist
            os.makedirs('tmp', exist_ok=True)
            
            # Save the file
            file_path = os.path.join('tmp', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Update session state
            st.session_state.temp_file_paths[file_type] = file_path
            if file_type == 'previous':
                st.session_state.previous_file = uploaded_file
            else:
                st.session_state.current_file = uploaded_file
                
            return file_path
        return None
    
    # File upload section with improved UI (dark mode compatible)
    st.header("Upload Market Snapshots")
    
    # Custom CSS for file upload section
    st.markdown("""
    <style>
    .file-upload-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #444;
    }
    .file-upload-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #4a86e8;
    }
    .file-upload-description {
        margin-bottom: 15px;
        font-size: 14px;
        color: #cccccc;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
        st.markdown('<div class="file-upload-header">Previous Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="file-upload-description">Upload an older market snapshot file (CSV or Excel) to use as the baseline for comparison.</div>', unsafe_allow_html=True)
        
        # File uploader with session state
        previous_file = st.file_uploader("Upload previous snapshot file", 
                                        type=['csv', 'xlsx', 'xls'], 
                                        key="previous_uploader")
        
        # If a new file is uploaded, save it
        if previous_file is not None and previous_file != st.session_state.previous_file:
            save_uploaded_file(previous_file, 'previous')
            
        # Display the current file
        if st.session_state.previous_file is not None:
            st.success(f"File loaded: {st.session_state.previous_file.name}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
        st.markdown('<div class="file-upload-header">Current Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="file-upload-description">Upload a recent market snapshot file (CSV or Excel) to compare against the previous snapshot.</div>', unsafe_allow_html=True)
        
        # File uploader with session state
        current_file = st.file_uploader("Upload current snapshot file", 
                                      type=['csv', 'xlsx', 'xls'], 
                                      key="current_uploader")
        
        # If a new file is uploaded, save it
        if current_file is not None and current_file != st.session_state.current_file:
            save_uploaded_file(current_file, 'current')
            
        # Display the current file
        if st.session_state.current_file is not None:
            st.success(f"File loaded: {st.session_state.current_file.name}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis options with improved UI
    st.header("Analysis Options")
    
    # Custom CSS for analysis options
    st.markdown("""
    <style>
    .option-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #0078ff;
    }
    .option-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #0078ff;
    }
    .option-description {
        font-size: 12px;
        color: #666;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<div class="option-title">Stock Correlation Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="option-description">Control the threshold for network analysis connections</div>', unsafe_allow_html=True)
        min_correlation = st.slider("Minimum correlation threshold", 0.5, 0.95, 0.7, 0.05,
                                   help="Higher values show only stronger correlations between stocks")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<div class="option-title">Volume Change Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="option-description">Set the sensitivity for volume spike detection</div>', unsafe_allow_html=True)
        volume_spike_threshold = st.slider("Volume spike threshold (%)", 50, 500, 100, 10,
                                         help="Minimum percentage increase in volume to qualify as a spike")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.markdown('<div class="option-title">IPO Analysis Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="option-description">Configure the time period for recent IPO analysis</div>', unsafe_allow_html=True)
        recent_ipo_year = st.number_input("Recent IPO year cutoff", 2015, datetime.now().year, 2022,
                                        help="Companies that went public on or after this year will be analyzed as recent IPOs")
    
    # Advanced Filters
    with st.expander("Advanced Filters", expanded=False):
        st.subheader("Price & Volume Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Minimum Price ($)", 0.0, 10000.0, 1.0, 0.1)
            min_gain = st.number_input("Minimum Gain (%)", -100.0, 1000.0, 0.0, 0.5)
            min_volume = st.number_input("Minimum Volume", 0, 1000000000, 10000, 10000)
            min_market_cap = st.number_input("Minimum Market Cap ($M)", 0.0, 5000.0, 10.0, 10.0)
        
        with col2:
            max_price = st.number_input("Maximum Price ($)", 0.0, 10000.0, 1000.0, 10.0)
            max_loss = st.number_input("Maximum Loss (%)", -100.0, 0.0, -5.0, 0.5)
            max_volume_change = st.number_input("Maximum Volume Change (%)", 0.0, 10000.0, 500.0, 50.0)
            max_market_cap = st.number_input("Maximum Market Cap ($B)", 0.0, 5000.0, 100.0, 5.0)
        
        st.subheader("Sector & Industry Filters")
        selected_sectors = st.multiselect("Filter by Sectors", 
                                          ["All", "Technology", "Healthcare", "Financial Services", 
                                           "Consumer Cyclical", "Industrials", "Communication Services",
                                           "Consumer Defensive", "Energy", "Basic Materials", "Real Estate", "Utilities"],
                                          default=["All"])
        
        st.subheader("Display Options")
        top_n_results = st.slider("Number of top/bottom results to show", 5, 100, 20, 5)
        show_charts = st.checkbox("Show interactive charts", value=True)
    
    # Create session state variables if they don't exist
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'files_uploaded' not in st.session_state:
        st.session_state.files_uploaded = False
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'prev_path' not in st.session_state:
        st.session_state.prev_path = None
    if 'curr_path' not in st.session_state:
        st.session_state.curr_path = None
    
    # Analysis selection with improved UI
    st.subheader("Select Analyses to Run")
    
    # CSS for analysis selection - compact side-by-side layout
    st.markdown("""
    <style>
    .analysis-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 10px;
    }
    .analysis-item {
        display: flex;
        align-items: center;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 5px 10px;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .analysis-icon {
        font-size: 16px;
        margin-right: 6px;
        color: #0078ff;
    }
    .analysis-title {
        font-size: 14px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Analysis options with descriptions and detailed help text
    analyses = [
        {"id": "price", "icon": "📈", "title": "Price Movement", 
         "description": "Identify biggest gainers and losers between snapshots",
         "help": "Analyzes price changes between snapshot periods to identify the stocks with the most significant price gains and losses. Includes scatter plots for price visualization and distribution analysis of market movements."},
        {"id": "volume", "icon": "📊", "title": "Volume Spikes", 
         "description": "Detect stocks with unusual trading volume changes",
         "help": "Identifies stocks with abnormal trading volume increases or decreases compared to the previous period. Helps detect possible institutional interest, news-driven trading, or other significant market events affecting liquidity."},
        {"id": "sector", "icon": "🏢", "title": "Sector Performance", 
         "description": "Compare performance across different market sectors",
         "help": "Compares stock performance grouped by market sectors to identify industry trends. Shows which sectors are leading or lagging the broader market and provides detailed sector rotation analysis."},
        {"id": "marketcap", "icon": "💰", "title": "Market Cap", 
         "description": "Track changes in company valuations",
         "help": "Analyzes changes in company valuations (market capitalization) between snapshot periods. Identifies companies gaining or losing significant market value and shows market cap distribution across sectors."},
        {"id": "ipo", "icon": "🚀", "title": "Recent IPOs",
         "description": "Analyze performance of newly public companies",
         "help": "Focuses on recently public companies to track their post-IPO performance. Helps identify trends in new listings and compare performance of recent IPOs against established companies in the same sectors."},
        {"id": "correlation", "icon": "🕸️", "title": "Correlation Network", 
         "description": "Visualize relationships between stocks",
         "help": "Creates a visual network chart showing correlations between stocks based on price movements. Helps identify clusters of related stocks and potential diversification opportunities by revealing hidden relationships."},
        {"id": "strength", "icon": "💪", "title": "Relative Strength", 
         "description": "Find stocks outperforming their sectors",
         "help": "Measures how individual stocks perform relative to their broader sector. Identifies outperformers that show strength against their peer group, which can signal potential leadership stocks with continued momentum."}
    ]
    
    # Create improved analysis selection UI with proper state handling
    if 'analysis_selections' not in st.session_state:
        st.session_state.analysis_selections = {a["id"]: True for a in analyses}
    
    analysis_selections = {}
    
    # Custom CSS for better analysis option display
    st.markdown("""
    <style>
    .analysis-container {
        background-color: #f7f7f7;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        border-left: 3px solid #4a86e8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create rows with 2 analysis options per row (with improved styling)
    for i in range(0, len(analyses), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(analyses):
                analysis = analyses[i]
                with st.container():
                    st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="analysis-item">
                        <div class="analysis-icon">{analysis["icon"]}</div>
                        <div class="analysis-title">{analysis["title"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Use the title as checkbox label with better help text
                    analysis_selections[analysis["id"]] = st.checkbox(
                        f"{analysis['title']}", 
                        value=st.session_state.analysis_selections[analysis["id"]], 
                        key=f"run_{analysis['id']}",
                        help=f"{analysis['description']}. {analysis['help']}"
                    )
                    # Store selection in session state
                    st.session_state.analysis_selections[analysis["id"]] = analysis_selections[analysis["id"]]
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if i+1 < len(analyses):
                analysis = analyses[i+1]
                with st.container():
                    st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="analysis-item">
                        <div class="analysis-icon">{analysis["icon"]}</div>
                        <div class="analysis-title">{analysis["title"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Use the title as checkbox label with better help text
                    analysis_selections[analysis["id"]] = st.checkbox(
                        f"{analysis['title']}", 
                        value=st.session_state.analysis_selections[analysis["id"]], 
                        key=f"run_{analysis['id']}",
                        help=f"{analysis['description']}. {analysis['help']}"
                    )
                    # Store selection in session state
                    st.session_state.analysis_selections[analysis["id"]] = analysis_selections[analysis["id"]]
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Assign checkbox values to individual variables
    run_price = analysis_selections["price"]
    run_volume = analysis_selections["volume"]
    run_sector = analysis_selections["sector"]
    run_marketcap = analysis_selections["marketcap"]
    run_ipo = analysis_selections["ipo"]
    run_correlation = analysis_selections["correlation"]
    run_strength = analysis_selections["strength"]
    
    # Custom button styling
    st.markdown("""
    <style>
    .step-button-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .step-number {
        display: inline-block;
        width: 30px;
        height: 30px;
        background-color: #0078ff;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        margin-right: 10px;
        font-weight: bold;
    }
    .step-instructions {
        font-size: 14px;
        color: #666;
        margin: 10px 0 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Two-step workflow with styled buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="step-button-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">1</div> <b>Apply Filters</b>', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-instructions">
        First, upload your snapshot files and apply the selected filters to prepare your dataset for analysis.
        </div>
        """, unsafe_allow_html=True)
        
        apply_filters_button = st.button(
            "Apply Filters and Prepare Data", 
            type="secondary", 
            disabled=(st.session_state.previous_file is None or st.session_state.current_file is None),
            use_container_width=True,
            key="apply_filters_button"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="step-button-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">2</div> <b>Run Analysis</b>', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-instructions">
        After applying filters, run the selected analyses to generate visualizations and insights.
        </div>
        """, unsafe_allow_html=True)
        
        analyze_button = st.button(
            "Run Selected Analyses", 
            type="primary", 
            disabled=(st.session_state.previous_file is None or st.session_state.current_file is None),
            key="analyze_button",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 1: Handle file upload and apply filters
    if st.session_state.previous_file is not None and st.session_state.current_file is not None and apply_filters_button:
        # Use the already saved temporary files
        with st.spinner("Processing uploaded files..."):
            # Get paths from session state
            prev_path = st.session_state.temp_file_paths['previous']
            curr_path = st.session_state.temp_file_paths['current']
            
            if prev_path is None or curr_path is None:
                # Fallback if temp files weren't created
                prev_path = os.path.join(output_dir, f"previous_{int(time.time())}.csv")
                with open(prev_path, 'wb') as f:
                    f.write(st.session_state.previous_file.getvalue())
                    
                curr_path = os.path.join(output_dir, f"current_{int(time.time())}.csv")
                with open(curr_path, 'wb') as f:
                    f.write(st.session_state.current_file.getvalue())
            
            # Store the paths in session state for access during analysis
            st.session_state.prev_path = prev_path
            st.session_state.curr_path = curr_path
            st.session_state.files_uploaded = True
            
            st.success(f"Files processed successfully: {os.path.basename(prev_path)} and {os.path.basename(curr_path)}")
        
        # Create the analyzer
        with st.spinner("Initializing analyzer..."):
            try:
                analyzer = SnapshotAnalyzer(prev_path, curr_path, output_dir)
                st.session_state.analyzer = analyzer
                st.success(f"Loaded previous snapshot with {len(analyzer.previous_df)} stocks")
                st.success(f"Loaded current snapshot with {len(analyzer.current_df)} stocks")
                st.success(f"Found {len(analyzer.merged_df)} stocks for comparative analysis")
                
                # Apply all filters before running any analysis
                with st.spinner("Applying filters..."):
                    analyzer.apply_filters(
                        min_price=min_price,
                        max_price=max_price,
                        min_gain=min_gain,
                        max_loss=max_loss,
                        min_market_cap=min_market_cap,
                        max_market_cap=max_market_cap,
                        min_volume=min_volume,
                        max_volume_change=max_volume_change,
                        selected_sectors=selected_sectors
                    )
                    filtered_count = len(analyzer.filtered_df)
                    st.session_state.filtered_df = analyzer.filtered_df
                    st.session_state.filters_applied = True
                    
                    # Display a preview of the filtered data
                    st.subheader(f"Filter Preview: {filtered_count} stocks remain for analysis")
                    
                    # Show snow animation when filters are applied
                    rain(
                        emoji="❄️",
                        font_size=54,
                        falling_speed=5,
                        animation_length=1,
                    )
                    
                    if filtered_count > 0:
                        # Get cleaning stats if available
                        cleaning_summary = ""
                        if hasattr(analyzer, 'previous_cleaning_stats'):
                            previous_issues = analyzer.previous_cleaning_stats.get('issues_detected', 0)
                            cleaning_summary += f"Previous snapshot: {previous_issues} data issues fixed\n"
                        if hasattr(analyzer, 'current_cleaning_stats'):
                            current_issues = analyzer.current_cleaning_stats.get('issues_detected', 0)
                            cleaning_summary += f"Current snapshot: {current_issues} data issues fixed"
                        
                        if cleaning_summary:
                            st.info(cleaning_summary)
                            
                        # Show data cleaning details in an expander
                        with st.expander("Data Cleaning Details"):
                            if hasattr(analyzer, 'previous_cleaning_stats') and 'data_issues' in analyzer.previous_cleaning_stats:
                                st.subheader("Previous Snapshot Issues")
                                for issue in analyzer.previous_cleaning_stats['data_issues']:
                                    st.write(f"• {issue.get('issue_type', 'Issue')}: {issue.get('issue', '')}")
                            
                            if hasattr(analyzer, 'current_cleaning_stats') and 'data_issues' in analyzer.current_cleaning_stats:
                                st.subheader("Current Snapshot Issues")
                                for issue in analyzer.current_cleaning_stats['data_issues']:
                                    st.write(f"• {issue.get('issue_type', 'Issue')}: {issue.get('issue', '')}")
                                    
                        # Create a styled dataframe with hyperlinks for the preview
                        preview_df = analyzer.filtered_df.head(10).copy()
                        
                        # Define columns for the preview
                        display_cols = [
                            'Symbol', 'Name', 'Previous_Last_Sale', 'Current_Last_Sale', 
                            'Price_Change', 'Price_Change_Pct', 'Profit_Loss_Status', 
                            'Performance_Category', 'Sector'
                        ]
                        
                        # Filter to columns that exist in the dataframe
                        preview_cols = [col for col in display_cols if col in preview_df.columns]
                        preview_df = preview_df[preview_cols]
                        
                        # Create HTML version with hyperlinks
                        from utils.export_utils import create_hyperlinked_df
                        html_df = create_hyperlinked_df(preview_df)
                        
                        # Display the styled dataframe
                       # st.write("Sample of filtered data (Click on symbols to view on Yahoo Finance):")
                        #st.write(html_df.to_html(escape=False), unsafe_allow_html=True)
                        
                        # Show the full dataset in an expander
                        with st.expander("View Complete Filtered Dataset"):
                            # Initialize session state for pagination if not exists
                            if 'df_page' not in st.session_state:
                                st.session_state.df_page = 1
                            if 'df_page_size' not in st.session_state:
                                st.session_state.df_page_size = 50
                            
                            # Function to update page
                            def update_page(new_page):
                                st.session_state.df_page = new_page
                            
                            # Function to update page size
                            def update_page_size(new_size):
                                old_size = st.session_state.df_page_size
                                st.session_state.df_page_size = new_size
                                # Adjust page number to maintain approximate position in dataset
                                if old_size != 0:  # Prevent division by zero
                                    current_pos = (st.session_state.df_page - 1) * old_size
                                    st.session_state.df_page = (current_pos // new_size) + 1
                            
                            # Calculate total pages based on current page size
                            page_size = st.session_state.df_page_size
                            total_pages = max(1, (len(analyzer.filtered_df) + page_size - 1) // page_size)
                            
                            # Ensure current page is valid
                            st.session_state.df_page = min(st.session_state.df_page, total_pages)
                            
                            # Create pagination UI with dropdowns instead of number inputs
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                # Create page dropdown with page options
                                page_options = list(range(1, total_pages + 1))
                                current_page_index = min(st.session_state.df_page - 1, len(page_options) - 1)
                                selected_page = st.selectbox(
                                    "Page", 
                                    options=page_options,
                                    index=current_page_index,
                                    key="page_dropdown"
                                )
                                # Update session state when page changes
                                if selected_page != st.session_state.df_page:
                                    update_page(selected_page)
                            
                            with col2:
                                # Page size dropdown
                                size_options = [20, 50, 100, 200]
                                current_size_index = size_options.index(st.session_state.df_page_size) if st.session_state.df_page_size in size_options else 1
                                selected_size = st.selectbox(
                                    "Records per page", 
                                    options=size_options,
                                    index=current_size_index,
                                    key="page_size_dropdown"
                                )
                                # Update session state when size changes
                                if selected_size != st.session_state.df_page_size:
                                    update_page_size(selected_size)
                            
                            with col3:
                                # Show total pages info
                                st.markdown(f"<div style='padding-top: 32px;'><strong>Total:</strong> {total_pages} pages</div>", unsafe_allow_html=True)
                            
                            # Calculate start and end indices for the current page
                            start_idx = (st.session_state.df_page - 1) * st.session_state.df_page_size
                            end_idx = min(start_idx + st.session_state.df_page_size, len(analyzer.filtered_df))
                            
                            # Get the page of data
                            page_df = analyzer.filtered_df.iloc[start_idx:end_idx].copy()
                            
                            # Create HTML version with hyperlinks
                            page_html_df = create_hyperlinked_df(page_df)
                            
                            # Add CSS for better table formatting with dark mode compatibility
                            st.markdown("""
                            <style>
                            .dataframe {
                                width: 100%;
                                font-size: 14px;
                                border-collapse: collapse;
                            }
                            .dataframe th {
                                background-color: #262730;
                                color: #ffffff;
                                padding: 8px;
                                text-align: left;
                            }
                            .dataframe td {
                                padding: 6px;
                                border-bottom: 1px solid #555;
                                color: #fafafa;
                                background-color: #0e1117;
                            }
                            .dataframe tr:nth-child(even) {
                                background-color: #1e1e1e;
                            }
                            .dataframe tr:hover {
                                background-color: #2e2e2e;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Display the page with better formatting (dark mode compatible)
                            st.markdown(f"""
                            <div style='background-color: #262730; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                                <strong>Showing records {start_idx+1} to {end_idx} of {len(analyzer.filtered_df)}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display the page
                            st.write(page_html_df.to_html(escape=False, classes='dataframe'), unsafe_allow_html=True)
                        
                        # Export options
                        with st.expander("Export Filtered Data"):
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                from utils.export_utils import download_link, export_to_excel
                                excel_data, excel_file = export_to_excel(analyzer.filtered_df, f"filtered_stocks_{int(time.time())}.xlsx")
                                st.markdown(download_link(excel_data, excel_file, "Download Excel File"), unsafe_allow_html=True)
                            
                            with export_col2:
                                from utils.export_utils import export_to_csv
                                csv_data, csv_file = export_to_csv(analyzer.filtered_df, f"filtered_stocks_{int(time.time())}.csv")
                                st.markdown(download_link(csv_data, csv_file, "Download CSV File"), unsafe_allow_html=True)
                        
                        st.success("Filters applied successfully! Click 'Step 2: Analyze Snapshots' to run analysis.")
                    else:
                        st.warning("No stocks match your filter criteria. Please adjust your filters and try again.")
                        st.session_state.filters_applied = False
            
            except Exception as e:
                st.error(f"Error initializing analyzer: {e}")
                st.session_state.filters_applied = False
                st.stop()
    
    # Step 2: Handle analysis
    if st.session_state.filters_applied and analyze_button:
        # Show balloon animation when analysis starts
        rain(
            emoji="🎈",
            font_size=54,
            falling_speed=5,
            animation_length=1,
        )
        
        # Use the analyzer from session state
        analyzer = st.session_state.analyzer
        st.success("Starting analysis with the filtered dataset...")
        
        # Run selected analyses
        results = {}
        
        if run_price:
            with st.spinner("Analyzing price movements..."):
                results['price_movement'] = analyzer.analyze_price_movement(
                    top_n=top_n_results,
                    min_price=min_price,
                    max_price=max_price,
                    min_gain=min_gain,
                    max_loss=max_loss,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    selected_sectors=selected_sectors
                )
                
                # Show top gainers and losers
                st.subheader("Top Price Gainers & Losers")
                
                tab1, tab2 = st.tabs(["HTML Visualization", "Data Table"])
                
                with tab1:
                    # Display the HTML file directly
                    html_file = results['price_movement']['files'][0]
                    with open(html_file, 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=550)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top Gainers")
                        gainers_df = pd.DataFrame(results['price_movement']['gainers'])
                        st.dataframe(gainers_df[['Symbol', 'Name', 'Previous_Last_Sale', 
                                                'Current_Last_Sale', 'Price_Change_Pct']])
                    
                    with col2:
                        st.subheader("Top Losers")
                        losers_df = pd.DataFrame(results['price_movement']['losers'])
                        st.dataframe(losers_df[['Symbol', 'Name', 'Previous_Last_Sale', 
                                               'Current_Last_Sale', 'Price_Change_Pct']])
                
                # Show price scatter plot
                st.subheader("Price Comparison")
                # Safety check to ensure file index exists
                if len(results['price_movement']['files']) > 2:
                    with open(results['price_movement']['files'][2], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800)
                else:
                    st.warning("Price comparison visualization is not available")
        
        if run_volume:
            with st.spinner("Analyzing volume spikes..."):
                results['volume_spikes'] = analyzer.analyze_volume_spikes(threshold_pct=volume_spike_threshold)
                
                st.subheader("Volume Spike Analysis")
                
                tab1, tab2 = st.tabs(["HTML Visualization", "Data Table"])
                
                with tab1:
                    with open(results['volume_spikes']['files'][0], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800)
                
                with tab2:
                    # Create a DataFrame from the results
                    volume_df = pd.DataFrame(results['volume_spikes']['volume_spikes'])
                    st.dataframe(volume_df[['Symbol', 'Name', 'Previous_Volume', 
                                           'Current_Volume', 'Volume_Change_Pct', 'Price_Change_Pct']])
                
                # Show volume vs price scatter plot
                st.subheader("Volume vs Price Relationship")
                
                # Safety check to make sure the file exists and index is valid
                if len(results['volume_spikes']['files']) > 1:
                    # Use the second file (index 1) which is the volume_price_relationship.html
                    with open(results['volume_spikes']['files'][1], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                else:
                    st.warning("Volume vs price relationship visualization is not available")
        
        if run_sector:
            with st.spinner("Analyzing sector performance..."):
                results['sectors'] = analyzer.analyze_sectors()
                
                st.subheader("Sector Performance")
                
                # Show sector performance bar chart
                with open(results['sectors']['files'][0], 'r') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)
                
                # Show sector treemap
                st.subheader("Market Cap by Sector")
                if len(results['sectors']['files']) > 1:
                    with open(results['sectors']['files'][1], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                else:
                    st.warning("Market cap by sector visualization is not available")
                
                # Show sector waterfall chart
                st.subheader("Sector Market Cap Changes")
                
                # Safety check to make sure the file index exists
                if len(results['sectors']['files']) > 2:
                    with open(results['sectors']['files'][2], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=400)
                else:
                    st.warning("Sector market cap changes visualization is not available")
        
        if run_marketcap:
            with st.spinner("Analyzing market cap changes..."):
                results['market_cap'] = analyzer.analyze_market_cap()
                
                st.subheader("Market Cap Changes")
                
                # Show top gainers and losers by market cap
                with open(results['market_cap']['files'][0], 'r') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=550)
                
                # Show bubble chart
                st.subheader("Market Cap Bubble Chart")
                if len(results['market_cap']['files']) > 1:
                    with open(results['market_cap']['files'][1], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800)
                else:
                    st.warning("Market cap bubble chart visualization is not available")
                
                # Show summary metrics
                summary = results['market_cap']['market_cap_summary']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Market Cap Change",
                        value=f"${summary['absolute_change']/1e9:.1f}B",
                        delta=f"{summary['percent_change']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Previous Total",
                        value=f"${summary['previous_total']/1e9:.1f}B"
                    )
                
                with col3:
                    st.metric(
                        label="Current Total",
                        value=f"${summary['current_total']/1e9:.1f}B"
                    )
        
        if run_ipo:
            with st.spinner("Analyzing recent IPOs..."):
                results['ipos'] = analyzer.analyze_ipos(recent_year=recent_ipo_year)
                
                if 'files' in results['ipos'] and results['ipos']['files']:
                    st.subheader(f"Recent IPOs (Since {recent_ipo_year})")
                    
                    with open(results['ipos']['files'][0], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                    
                    if len(results['ipos']['files']) > 1:
                        st.subheader("Recent IPOs Performance")
                        with open(results['ipos']['files'][1], 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                else:
                    st.info(f"No IPOs found since {recent_ipo_year}")
        
        if run_strength:
            with st.spinner("Analyzing relative strength..."):
                results['relative_strength'] = analyzer.analyze_relative_strength()
                
                st.subheader("Relative Strength vs. Sector")
                
                with open(results['relative_strength']['files'][0], 'r') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)
                
                st.subheader("Relative Strength Distribution")
                if len(results['relative_strength']['files']) > 1:
                    with open(results['relative_strength']['files'][1], 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500)
                else:
                    st.warning("Relative strength distribution visualization is not available")
        
        if run_correlation:
            with st.spinner("Analyzing correlation network..."):
                try:
                    results['correlation_network'] = analyzer.analyze_correlation_network(min_correlation=min_correlation)
                    
                    if 'files' in results['correlation_network'] and results['correlation_network']['files']:
                        st.subheader("Stock Correlation Network")
                        with open(results['correlation_network']['files'][0], 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800)
                        
                        st.subheader("Sector Correlation Heatmap")
                        if len(results['correlation_network']['files']) > 1:
                            with open(results['correlation_network']['files'][1], 'r') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=700)
                        else:
                            st.warning("Sector correlation heatmap visualization is not available")
                except Exception as e:
                    st.error(f"Error in correlation analysis: {e}")
        
        # Generate summary report
        with st.spinner("Generating summary report..."):
            summary_results = analyzer.generate_summary_report()
            
            st.subheader("Summary Report")
            st.success(f"Analysis complete! Summary report generated.")
            
            # Display link to summary report
            summary_file = summary_results['summary_report']
            with open(summary_file, 'r') as f:
                html_content = f.read()
            
            if st.button("Show Summary Report"):
                st.components.v1.html(html_content, height=800, scrolling=True)
            
            st.download_button(
                label="Download Summary Report",
                data=open(summary_file, 'r').read(),
                file_name="market_snapshot_analysis.html",
                mime="text/html",
                key=f"summary_report_download_{int(time.time())}"
            )
            
            # Add comprehensive export options
            st.subheader("Comprehensive Export Options")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Create a comprehensive Excel export with all analysis data
                try:
                    # Create a BytesIO object for the Excel workbook
                    import io
                    excel_buffer = io.BytesIO()
                    
                    # Create Excel writer
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Write filtered dataset
                        if hasattr(st.session_state, 'filtered_df') and not st.session_state.filtered_df.empty:
                            st.session_state.filtered_df.to_excel(writer, sheet_name='Filtered_Data', index=False)
                        
                        # Write price movement results
                        if 'price_movement' in results and 'gainers' in results['price_movement']:
                            gainers_df = pd.DataFrame(results['price_movement']['gainers'])
                            gainers_df.to_excel(writer, sheet_name='Top_Gainers', index=False)
                        
                        if 'price_movement' in results and 'losers' in results['price_movement']:
                            losers_df = pd.DataFrame(results['price_movement']['losers'])
                            losers_df.to_excel(writer, sheet_name='Top_Losers', index=False)
                        
                        # Write volume spikes
                        if 'volume_spikes' in results and 'volume_spikes' in results['volume_spikes']:
                            volume_df = pd.DataFrame(results['volume_spikes']['volume_spikes'])
                            volume_df.to_excel(writer, sheet_name='Volume_Spikes', index=False)
                        
                        # Write sector performance
                        if 'sectors' in results and 'sector_performance' in results['sectors']:
                            sector_df = pd.DataFrame(results['sectors']['sector_performance'])
                            sector_df.to_excel(writer, sheet_name='Sector_Performance', index=False)
                        
                        # Write market cap analysis
                        if 'market_cap' in results and 'market_cap_changes' in results['market_cap']:
                            market_cap_df = pd.DataFrame(results['market_cap']['market_cap_changes'])
                            market_cap_df.to_excel(writer, sheet_name='Market_Cap_Changes', index=False)
                        
                        # Get the workbook and worksheet objects
                        workbook = writer.book
                        
                        # Add a format for headers
                        header_format = workbook.add_format({
                            'bold': True,
                            'text_wrap': True,
                            'valign': 'top',
                            'bg_color': '#D3D3D3',
                            'border': 1
                        })
                        
                        # Apply the header format to each worksheet
                        for sheet_name, worksheet in writer.sheets.items():
                            # Get column names from the dataframe that was written to this sheet
                            if sheet_name == 'Filtered_Data' and hasattr(st.session_state, 'filtered_df'):
                                column_names = st.session_state.filtered_df.columns
                            elif sheet_name == 'Top_Gainers' and 'price_movement' in results and 'gainers' in results['price_movement']:
                                column_names = pd.DataFrame(results['price_movement']['gainers']).columns
                            elif sheet_name == 'Top_Losers' and 'price_movement' in results and 'losers' in results['price_movement']:
                                column_names = pd.DataFrame(results['price_movement']['losers']).columns
                            elif sheet_name == 'Volume_Spikes' and 'volume_spikes' in results and 'volume_spikes' in results['volume_spikes']:
                                column_names = pd.DataFrame(results['volume_spikes']['volume_spikes']).columns
                            elif sheet_name == 'Sector_Performance' and 'sectors' in results and 'sector_performance' in results['sectors']:
                                column_names = pd.DataFrame(results['sectors']['sector_performance']).columns
                            elif sheet_name == 'Market_Cap_Changes' and 'market_cap' in results and 'market_cap_changes' in results['market_cap']:
                                column_names = pd.DataFrame(results['market_cap']['market_cap_changes']).columns
                            else:
                                continue
                                
                            # Write headers with formatting
                            for col_num, column_name in enumerate(column_names):
                                worksheet.write(0, col_num, column_name, header_format)
                                # Set column width
                                worksheet.set_column(col_num, col_num, 15)
                    
                    # Create download button
                    st.download_button(
                        "Download Complete Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        key="download_complete_excel"
                    )
                except Exception as e:
                    st.error(f"Error creating Excel report: {str(e)}")
            
            with export_col2:
                # Add HTML export option for individual stocks
                st.markdown("#### Individual Stock HTML Reports")
                
                # Individual stock details - HTML report instead of PDF
                stock_symbols = []
                if hasattr(st.session_state, 'filtered_df') and not st.session_state.filtered_df.empty:
                    stock_symbols = st.session_state.filtered_df['Symbol'].unique().tolist()
                
                if stock_symbols:
                    selected_symbol = st.selectbox(
                        "Select a stock for detailed HTML report:",
                        options=stock_symbols,
                        key="html_stock_selection"
                    )
                    
                    if selected_symbol:
                        # Get the stock data
                        stock_data = st.session_state.filtered_df[st.session_state.filtered_df['Symbol'] == selected_symbol].iloc[0].to_dict()
                        
                        # Create dataframe with relevant stock details
                        stock_detail_df = pd.DataFrame([stock_data])
                        
                        # Display stock details in an expander
                        with st.expander(f"Details for {selected_symbol}", expanded=True):
                            # Display basic info
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "Current Price", 
                                    f"${stock_data.get('Current_Last_Sale', 0):.2f}",
                                    f"{stock_data.get('Price_Change_Pct', 0):.2f}%"
                                )
                                st.write(f"**Sector:** {stock_data.get('Sector', 'N/A')}")
                            with col2:
                                st.metric(
                                    "Volume", 
                                    f"{stock_data.get('Current_Volume', 0):,.0f}",
                                    f"{stock_data.get('Volume_Change_Pct', 0):.2f}%"
                                )
                                st.write(f"**Industry:** {stock_data.get('Industry', 'N/A')}")
                            
                            # Display full details table
                            st.subheader("All Stock Data")
                            st.dataframe(stock_detail_df)
                            
                            # Display info instead of download button
                            st.info("Stock details shown above. Full reports are available in the download section below.")
                else:
                    st.info("Apply filters and run analysis first to view stock details.")
        
        # Download links for all generated files
        st.header("Download Analysis Files")
        
        # Create tabs for different file categories
        tab_price, tab_volume, tab_sector, tab_cap, tab_other = st.tabs([
            "Price Analysis", "Volume Analysis", "Sector Analysis", 
            "Market Cap", "Other Files"
        ])
        
        with tab_price:
            price_files = [f for f in analyzer.created_files if 'price' in f.lower()]
            if price_files:
                for i, file in enumerate(price_files):
                    file_name = os.path.basename(file)
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"price_download_{i}_{file_name}"
                        )
            else:
                st.info("No price analysis files generated")
        
        with tab_volume:
            volume_files = [f for f in analyzer.created_files if 'volume' in f.lower()]
            if volume_files:
                for i, file in enumerate(volume_files):
                    file_name = os.path.basename(file)
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"volume_download_{i}_{file_name}"
                        )
            else:
                st.info("No volume analysis files generated")
        
        with tab_sector:
            sector_files = [f for f in analyzer.created_files if 'sector' in f.lower()]
            if sector_files:
                for i, file in enumerate(sector_files):
                    file_name = os.path.basename(file)
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"sector_download_{i}_{file_name}"
                        )
            else:
                st.info("No sector analysis files generated")
        
        with tab_cap:
            cap_files = [f for f in analyzer.created_files if 'cap' in f.lower() or 'market' in f.lower()]
            if cap_files:
                for i, file in enumerate(cap_files):
                    file_name = os.path.basename(file)
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"cap_download_{i}_{file_name}"
                        )
            else:
                st.info("No market cap analysis files generated")
        
        with tab_other:
            all_tabs = ['price', 'volume', 'sector', 'cap', 'market']
            other_files = [f for f in analyzer.created_files if not any(keyword in f.lower() for keyword in all_tabs)]
            if other_files:
                for i, file in enumerate(other_files):
                    file_name = os.path.basename(file)
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"Download {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"other_download_{i}_{file_name}"
                        )
            else:
                st.info("No other analysis files generated")
        
        # Add a download all button for batch downloading
        if hasattr(analyzer, 'created_files') and analyzer.created_files:
            st.divider()
            st.subheader("Download All Files")
            
            # Create a zip file with all generated files
            files_dict = {}
            for file_path in analyzer.created_files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    with open(file_path, 'rb') as f:
                        files_dict[file_name] = f.read()
            
            # Create the zip file
            if files_dict:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_name, file_content in files_dict.items():
                        zip_file.writestr(file_name, file_content)
                
                zip_buffer.seek(0)
                
                # Provide the download button
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=zip_buffer,
                    file_name=f"snapshot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    help="Download all analysis files in a single ZIP archive",
                    use_container_width=True
                )
    
    # Instructions when no files are uploaded
    elif not analyze_button:
        st.info("Upload two snapshot files and click 'Analyze Snapshots' to begin the analysis.")
        
        # Add reset button in the sidebar
        with st.sidebar:
            # Add a Clear/Reset button
            if st.button("🔄 Reset Analysis", use_container_width=True):
                # Clear session state to reset the analysis
                for key in list(st.session_state.keys()):
                    if key in ['previous', 'current', 'analyzer', 'filtered_df', 
                              'files_uploaded', 'filters_applied', 'prev_path', 'curr_path']:
                        del st.session_state[key]
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This tool performs comparative analysis between two market snapshots. The data 
    is not stored beyond your current session and is only used for the analysis you see here.
    """)

if __name__ == "__main__":
    main()