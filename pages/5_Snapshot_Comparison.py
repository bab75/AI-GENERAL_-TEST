import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from compare_snapshots import SnapshotAnalyzer
from streamlit_extras.let_it_rain import rain
import zipfile
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Snapshot Comparison - Stock Analysis Platform",
    page_icon="📸",
    layout="wide"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
.stApp {
    background-color: #0e1117;
}
.file-upload-container, .option-card, .analysis-container {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #444;
}
.file-upload-header, .option-title {
    font-size: 18px;
    font-weight: bold;
    color: #4a86e8;
    margin-bottom: 10px;
}
.file-upload-description, .option-description {
    font-size: 14px;
    color: #cccccc;
}
.step-button-container {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 15px;
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
    color: #cccccc;
    margin: 10px 0 15px 0;
}
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
.help-tooltip {
    display: inline-block;
    cursor: help;
    color: #4a86e8;
    margin-left: 5px;
}
.help-tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    background-color: #262730;
    color: #ffffff;
    padding: 5px 10px;
    border-radius: 5px;
    z-index: 10;
    font-size: 12px;
    max-width: 200px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'temp_file_paths' not in st.session_state:
        st.session_state.temp_file_paths = {'previous': None, 'current': None}
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
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'analysis_selections' not in st.session_state:
        st.session_state.analysis_selections = {
            "price": True, "volume": True, "sector": True, "marketcap": True,
            "ipo": True, "correlation": True, "strength": True
        }

    # Output directory
    output_dir = "snapshot_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Tab-based workflow
    st.title("📸 Market Snapshot Comparison")
    st.markdown("Compare two market snapshots to uncover trends and insights.")
    
    tabs = st.tabs(["Upload", "Configure", "Analyze", "Results", "Export"])

    # Tab 1: Upload
    with tabs[0]:
        st.header("Upload Market Snapshots")
        st.info("Upload two CSV or Excel files containing stock data (columns: Symbol, Name, Last Sale, etc.).")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
            st.markdown('<div class="file-upload-header">Previous Snapshot</div>', unsafe_allow_html=True)
            st.markdown('<div class="file-upload-description">Older snapshot as baseline.</div>', unsafe_allow_html=True)
            previous_file = st.file_uploader("Upload previous snapshot", type=['csv', 'xlsx', 'xls'], key="previous_uploader")
            if previous_file and previous_file != st.session_state.previous_file:
                file_path = os.path.join('tmp', previous_file.name)
                os.makedirs('tmp', exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(previous_file.getbuffer())
                st.session_state.temp_file_paths['previous'] = file_path
                st.session_state.previous_file = previous_file
                st.success(f"Loaded: {previous_file.name}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
            st.markdown('<div class="file-upload-header">Current Snapshot</div>', unsafe_allow_html=True)
            st.markdown('<div class="file-upload-description">Recent snapshot for comparison.</div>', unsafe_allow_html=True)
            current_file = st.file_uploader("Upload current snapshot", type=['csv', 'xlsx', 'xls'], key="current_uploader")
            if current_file and current_file != st.session_state.current_file:
                file_path = os.path.join('tmp', current_file.name)
                os.makedirs('tmp', exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(current_file.getbuffer())
                st.session_state.temp_file_paths['current'] = file_path
                st.session_state.current_file = current_file
                st.success(f"Loaded: {current_file.name}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Configure
    with tabs[1]:
        st.header("Configure Filters and Options")
        st.info("Set filters to refine the dataset and customize analysis parameters.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<div class="option-title">Correlation Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="option-description">Set correlation threshold for network analysis.</div>', unsafe_allow_html=True)
            min_correlation = st.slider("Minimum correlation", 0.5, 0.95, 0.7, 0.05, 
                                       help="Higher values show stronger stock relationships.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<div class="option-title">Volume Spikes</div>', unsafe_allow_html=True)
            st.markdown('<div class="option-description">Detect significant volume changes.</div>', unsafe_allow_html=True)
            volume_spike_threshold = st.slider("Volume spike threshold (%)", 50, 500, 100, 10,
                                             help="Minimum volume increase to flag as a spike.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<div class="option-title">Recent IPOs</div>', unsafe_allow_html=True)
            st.markdown('<div class="option-description">Analyze companies post-IPO.</div>', unsafe_allow_html=True)
            recent_ipo_year = st.number_input("IPO year cutoff", 2015, datetime.now().year, 2022,
                                            help="Analyze IPOs from this year onward.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Advanced Filters"):
            st.subheader("Price & Volume Filters")
            col1, col2 = st.columns(2)
            with col1:
                min_price = st.number_input("Min Price ($)", 0.0, 10000.0, 1.0, 0.1, 
                                           help="Exclude stocks below this price.")
                min_gain = st.number_input("Min Gain (%)", -100.0, 1000.0, 0.0, 0.5, 
                                          help="Minimum price increase.")
                min_volume = st.number_input("Min Volume", 0, 1000000000, 10000, 10000, 
                                            help="Minimum trading volume.")
                min_market_cap = st.number_input("Min Market Cap ($M)", 0.0, 5000.0, 10.0, 10.0, 
                                                help="Minimum market capitalization.")
            with col2:
                max_price = st.number_input("Max Price ($)", 0.0, 10000.0, 1000.0, 10.0, 
                                           help="Exclude stocks above this price.")
                max_loss = st.number_input("Max Loss (%)", -100.0, 0.0, -5.0, 0.5, 
                                          help="Maximum price decrease.")
                max_volume_change = st.number_input("Max Volume Change (%)", 0.0, 10000.0, 500.0, 50.0, 
                                                   help="Maximum volume increase.")
                max_market_cap = st.number_input("Max Market Cap ($B)", 0.0, 5000.0, 100.0, 5.0, 
                                                help="Maximum market capitalization.")
            
            st.subheader("Sector Filter")
            selected_sectors = st.multiselect("Sectors", 
                                             ["All", "Technology", "Healthcare", "Financial Services", 
                                              "Consumer Cyclical", "Industrials", "Communication Services",
                                              "Consumer Defensive", "Energy", "Basic Materials", "Real Estate", "Utilities"],
                                             default=["All"],
                                             help="Select sectors to include. 'All' includes every sector.")
            
            st.subheader("Display Options")
            top_n_results = st.slider("Top/bottom results", 5, 100, 20, 5, 
                                     help="Number of top/bottom stocks to show per analysis.")
            show_charts = st.checkbox("Show interactive charts", value=True, 
                                     help="Enable/disable interactive Plotly charts.")

        st.markdown('<div class="step-button-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">1</div> <b>Apply Filters</b>', unsafe_allow_html=True)
        st.markdown('<div class="step-instructions">Prepare your dataset by applying filters.</div>', unsafe_allow_html=True)
        apply_filters_button = st.button(
            "Apply Filters", 
            type="secondary", 
            disabled=(st.session_state.previous_file is None or st.session_state.current_file is None),
            use_container_width=True,
            key="apply_filters_button"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if apply_filters_button and st.session_state.previous_file is not None and st.session_state.current_file is not None:
            with st.spinner("Processing files..."):
                prev_path = st.session_state.temp_file_paths['previous']
                curr_path = st.session_state.temp_file_paths['current']
                if prev_path is None or curr_path is None:
                    prev_path = os.path.join(output_dir, f"previous_{int(time.time())}.csv")
                    with open(prev_path, 'wb') as f:
                        f.write(st.session_state.previous_file.getvalue())
                    curr_path = os.path.join(output_dir, f"current_{int(time.time())}.csv")
                    with open(curr_path, 'wb') as f:
                        f.write(st.session_state.current_file.getvalue())
                    st.session_state.temp_file_paths['previous'] = prev_path
                    st.session_state.temp_file_paths['current'] = curr_path
                
                st.session_state.prev_path = prev_path
                st.session_state.curr_path = curr_path
                st.session_state.files_uploaded = True
            
            with st.spinner("Initializing analyzer..."):
                try:
                    analyzer = SnapshotAnalyzer(prev_path, curr_path, output_dir)
                    st.session_state.analyzer = analyzer
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
                    rain(emoji="❄️", font_size=54, falling_speed=5, animation_length=1)
                    if filtered_count > 0:
                        st.success(f"Filters applied: {filtered_count} stocks remain.")
                        if hasattr(analyzer, 'previous_cleaning_stats'):
                            st.info(f"Cleaned {analyzer.previous_cleaning_stats.get('issues_detected', 0)} issues in previous snapshot.")
                        if hasattr(analyzer, 'current_cleaning_stats'):
                            st.info(f"Cleaned {analyzer.current_cleaning_stats.get('issues_detected', 0)} issues in current snapshot.")
                    else:
                        st.warning("No stocks match filters. Adjust and retry.")
                        st.session_state.filters_applied = False
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.filters_applied = False

    # Tab 3: Analyze
    with tabs[2]:
        st.header("Select Analyses")
        st.info("Choose analyses to run on the filtered dataset. Each analysis provides specific market insights.")
        
        analyses = [
            {"id": "price", "title": "Price Movement", "result_key": "price_movement",
             "icon": "📈", "description": "Shows top gainers and losers by price change.",
             "help": "Identifies stocks with significant price changes, including scatter plots of price movements."},
            {"id": "volume", "title": "Volume Spikes", "result_key": "volume_spikes",
             "icon": "📊", "description": "Detects unusual trading volume changes.",
             "help": "Flags stocks with high volume increases, indicating potential market interest."},
            {"id": "sector", "title": "Sector Performance", "result_key": "sectors",
             "icon": "🏢", "description": "Compares performance across sectors.",
             "help": "Analyzes sector trends with bar charts and treemaps."},
            {"id": "marketcap", "title": "Market Cap", "result_key": "market_cap",
             "icon": "💰", "description": "Tracks changes in company valuations.",
             "help": "Shows market cap shifts with bubble charts and metrics."},
            {"id": "ipo", "title": "Recent IPOs", "result_key": "ipos",
             "icon": "🚀", "description": "Analyzes performance of new public companies.",
             "help": "Focuses on IPOs since the specified year."},
            {"id": "correlation", "title": "Correlation Network", "result_key": "correlation_network",
             "icon": "🕸️", "description": "Visualizes stock price relationships.",
             "help": "Creates network graphs and heatmaps of stock correlations."},
            {"id": "strength", "title": "Relative Strength", "result_key": "relative_strength",
             "icon": "💪", "description": "Finds stocks outperforming their sectors.",
             "help": "Highlights stocks with strong performance relative to their sector."}
        ]
        
        for i in range(0, len(analyses), 2):
            col1, col2 = st.columns(2)
            with col1:
                if i < len(analyses):
                    a = analyses[i]
                    st.markdown(f'<div class="analysis-container">{a["icon"]} {a["title"]}</div>', unsafe_allow_html=True)
                    st.session_state.analysis_selections[a["id"]] = st.checkbox(
                        a["title"], 
                        value=st.session_state.analysis_selections[a["id"]], 
                        key=f"run_{a['id']}",
                        help=f"{a['description']} {a['help']}"
                    )
            with col2:
                if i+1 < len(analyses):
                    a = analyses[i+1]
                    st.markdown(f'<div class="analysis-container">{a["icon"]} {a["title"]}</div>', unsafe_allow_html=True)
                    st.session_state.analysis_selections[a["id"]] = st.checkbox(
                        a["title"], 
                        value=st.session_state.analysis_selections[a["id"]], 
                        key=f"run_{a['id']}",
                        help=f"{a['description']} {a['help']}"
                    )
        
        st.markdown('<div class="step-button-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">2</div> <b>Run Analysis</b>', unsafe_allow_html=True)
        st.markdown('<div class="step-instructions">Generate insights from selected analyses.</div>', unsafe_allow_html=True)
        analyze_button = st.button(
            "Run Analyses", 
            type="primary", 
            disabled=(not st.session_state.filters_applied or st.session_state.filtered_df.empty),
            use_container_width=True,
            key="analyze_button"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_button and st.session_state.filters_applied:
            analyzer = st.session_state.analyzer
            results = {}
            rain(emoji="🎈", font_size=54, falling_speed=5, animation_length=1)
            
            if st.session_state.analysis_selections["price"]:
                with st.spinner("Analyzing price movements..."):
                    results['price_movement'] = analyzer.analyze_price_movement(
                        top_n=top_n_results, min_price=min_price, max_price=max_price,
                        min_gain=min_gain, max_loss=max_loss, min_market_cap=min_market_cap,
                        max_market_cap=max_market_cap, selected_sectors=selected_sectors
                    )
            if st.session_state.analysis_selections["volume"]:
                with st.spinner("Analyzing volume spikes..."):
                    results['volume_spikes'] = analyzer.analyze_volume_spikes(threshold_pct=volume_spike_threshold)
            if st.session_state.analysis_selections["sector"]:
                with st.spinner("Analyzing sectors..."):
                    results['sectors'] = analyzer.analyze_sectors()
            if st.session_state.analysis_selections["marketcap"]:
                with st.spinner("Analyzing market cap..."):
                    results['market_cap'] = analyzer.analyze_market_cap()
            if st.session_state.analysis_selections["ipo"]:
                with st.spinner("Analyzing IPOs..."):
                    results['ipos'] = analyzer.analyze_ipos(recent_year=recent_ipo_year)
            if st.session_state.analysis_selections["strength"]:
                with st.spinner("Analyzing relative strength..."):
                    results['relative_strength'] = analyzer.analyze_relative_strength()
            if st.session_state.analysis_selections["correlation"]:
                with st.spinner("Analyzing correlations..."):
                    try:
                        results['correlation_network'] = analyzer.analyze_correlation_network(min_correlation=min_correlation)
                    except Exception as e:
                        st.error(f"Correlation analysis error: {e}")
            
            st.session_state.analysis_results = results
            st.success("Analysis complete! View results in the 'Results' tab.")

    # Tab 4: Results
    with tabs[3]:
        st.header("Analysis Results")
        if not st.session_state.get('filters_applied', False):
            st.info("Please apply filters and run analyses to view results.")
        elif not st.session_state.get('analysis_results', {}):
            st.info("No analyses run yet. Select and run analyses in the 'Analyze' tab.")
        else:
            results = st.session_state.analysis_results
            # NEW: Map result keys to analysis titles and filter only valid results
            result_keys_to_titles = {a["result_key"]: a["title"] for a in analyses}
            valid_result_keys = [k for k in results.keys() if k in result_keys_to_titles]
            tab_titles = ["Overview"] + [result_keys_to_titles[k] for k in valid_result_keys]
            
            # Debugging: Log tab creation details
            st.session_state['debug_result_keys'] = valid_result_keys
            st.session_state['debug_tab_titles'] = tab_titles
            
            result_tabs = st.tabs(tab_titles)
            
            # Overview tab
            with result_tabs[0]:
                st.subheader("Results Overview")
                st.markdown("Key insights from all analyses.")
                if 'price_movement' in results:
                    st.markdown("**Top Gainer**")
                    gainers = pd.DataFrame(results['price_movement']['gainers'])
                    if not gainers.empty:
                        top_gainer = gainers.iloc[0]
                        st.metric(f"{top_gainer['Symbol']} ({top_gainer['Name']})", 
                                 f"${top_gainer['Current_Last_Sale']:.2f}", 
                                 f"{top_gainer['Price_Change_Pct']:.2f}%")
                if 'market_cap' in results:
                    st.markdown("**Market Cap Change**")
                    summary = results['market_cap']['market_cap_summary']
                    st.metric("Total Market Cap", 
                             f"${summary['current_total']/1e9:.1f}B", 
                             f"{summary['percent_change']:.2f}%")
                if show_charts and 'sectors' in results and 'sector_fig' in results['sectors']:
                    st.markdown("**Sector Performance**")
                    st.plotly_chart(results['sectors']['sector_fig'], use_container_width=True)
            
            # NEW: Iterate over valid result keys to populate tabs
            for tab_idx, result_key in enumerate(valid_result_keys, 1):
                with result_tabs[tab_idx]:
                    analysis_title = result_keys_to_titles[result_key]
                    if result_key == 'price_movement':
                        st.subheader("Price Movement")
                        st.markdown("Shows top gainers and losers based on price change percentage.")
                        if show_charts and 'main_fig' in results['price_movement']:
                            st.plotly_chart(results['price_movement']['main_fig'], use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top Gainers**")
                            st.dataframe(pd.DataFrame(results['price_movement']['gainers'])[['Symbol', 'Name', 'Price_Change_Pct']],
                                        use_container_width=True)
                        with col2:
                            st.markdown("**Top Losers**")
                            st.dataframe(pd.DataFrame(results['price_movement']['losers'])[['Symbol', 'Name', 'Price_Change_Pct']],
                                        use_container_width=True)
                        if show_charts and 'scatter_fig' in results['price_movement']:
                            st.markdown("**Price Comparison**")
                            st.plotly_chart(results['price_movement']['scatter_fig'], use_container_width=True)
                    
                    elif result_key == 'volume_spikes':
                        st.subheader("Volume Spikes")
                        st.markdown("Identifies stocks with significant volume increases.")
                        if show_charts and 'main_fig' in results['volume_spikes']:
                            st.plotly_chart(results['volume_spikes']['main_fig'], use_container_width=True)
                        st.dataframe(pd.DataFrame(results['volume_spikes']['volume_spikes'])[['Symbol', 'Name', 'Volume_Change_Pct']],
                                    use_container_width=True)
                        if show_charts and 'scatter_fig' in results['volume_spikes']:
                            st.markdown("**Volume vs Price**")
                            st.plotly_chart(results['volume_spikes']['scatter_fig'], use_container_width=True)
                    
                    elif result_key == 'sectors':
                        st.subheader("Sector Performance")
                        st.markdown("Compares performance across market sectors.")
                        if show_charts and 'sector_fig' in results['sectors']:
                            st.plotly_chart(results['sectors']['sector_fig'], use_container_width=True)
                        if show_charts and 'treemap_fig' in results['sectors']:
                            st.markdown("**Market Cap by Sector**")
                            st.plotly_chart(results['sectors']['treemap_fig'], use_container_width=True)
                        if show_charts and 'waterfall_fig' in results['sectors']:
                            st.markdown("**Sector Market Cap Changes**")
                            st.plotly_chart(results['sectors']['waterfall_fig'], use_container_width=True)
                    
                    elif result_key == 'market_cap':
                        st.subheader("Market Cap Changes")
                        st.markdown("Tracks shifts in company valuations.")
                        if show_charts and 'main_fig' in results['market_cap']:
                            st.plotly_chart(results['market_cap']['main_fig'], use_container_width=True)
                        if show_charts and 'bubble_fig' in results['market_cap']:
                            st.markdown("**Market Cap Bubble Chart**")
                            st.plotly_chart(results['market_cap']['bubble_fig'], use_container_width=True)
                        summary = results['market_cap']['market_cap_summary']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Change", f"${summary['absolute_change']/1e9:.1f}B", f"{summary['percent_change']:.2f}%")
                        with col2:
                            st.metric("Previous Total", f"${summary['previous_total']/1e9:.1f}B")
                        with col3:
                            st.metric("Current Total", f"${summary['current_total']/1e9:.1f}B")
                    
                    elif result_key == 'ipos' and results['ipos'].get('main_fig'):
                        st.subheader(f"Recent IPOs (Since {recent_ipo_year})")
                        st.markdown("Analyzes performance of newly public companies.")
                        if show_charts:
                            st.plotly_chart(results['ipos']['main_fig'], use_container_width=True)
                        if show_charts and 'performance_fig' in results['ipos']:
                            st.markdown("**IPO Performance**")
                            st.plotly_chart(results['ipos']['performance_fig'], use_container_width=True)
                    
                    elif result_key == 'relative_strength':
                        st.subheader("Relative Strength")
                        st.markdown("Identifies stocks outperforming their sectors.")
                        if show_charts and 'main_fig' in results['relative_strength']:
                            st.plotly_chart(results['relative_strength']['main_fig'], use_container_width=True)
                        if show_charts and 'dist_fig' in results['relative_strength']:
                            st.markdown("**Strength Distribution**")
                            st.plotly_chart(results['relative_strength']['dist_fig'], use_container_width=True)
                    
                    elif result_key == 'correlation_network':
                        st.subheader("Correlation Network")
                        st.markdown("Visualizes relationships between stock price movements.")
                        if show_charts and 'network_fig' in results['correlation_network']:
                            st.plotly_chart(results['correlation_network']['network_fig'], use_container_width=True)
                        if show_charts and 'heatmap_fig' in results['correlation_network']:
                            st.markdown("**Sector Correlation Heatmap**")
                            st.plotly_chart(results['correlation_network']['heatmap_fig'], use_container_width=True)

    # Tab 5: Export
    with tabs[4]:
        st.header("Export and Download")
        if not st.session_state.get('filters_applied', False):
            st.info("Apply filters to view export options.")
        else:
            st.subheader("Filtered Data")
            if st.session_state.filtered_df is not None and not st.session_state.filtered_df.empty:
                preview_df = st.session_state.filtered_df.head(10)[['Symbol', 'Name', 'Price_Change_Pct', 'Sector']]
                st.markdown("**Preview (Top 10 Stocks)**")
                st.dataframe(preview_df, use_container_width=True)
                with st.expander("Full Dataset"):
                    st.dataframe(st.session_state.filtered_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    from utils.export_utils import export_to_excel
                    excel_data, excel_file = export_to_excel(st.session_state.filtered_df, f"filtered_stocks_{int(time.time())}.xlsx")
                    st.download_button("Download Excel", data=excel_data, file_name=excel_file, mime="application/vnd.ms-excel")
                with col2:
                    from utils.export_utils import export_to_csv
                    csv_data, csv_file = export_to_csv(st.session_state.filtered_df, f"filtered_stocks_{int(time.time())}.csv")
                    st.download_button("Download CSV", data=csv_data, file_name=csv_file, mime="text/csv")
            
            st.subheader("Stock Details")
            if st.session_state.filtered_df is not None and not st.session_state.filtered_df.empty:
                stock_symbols = st.session_state.filtered_df['Symbol'].unique().tolist()
                selected_symbol = st.selectbox("Select stock", stock_symbols, key="html_stock_selection", 
                                              disabled=not st.session_state.filters_applied)
                if selected_symbol:
                    stock_data = st.session_state.filtered_df[st.session_state.filtered_df['Symbol'] == selected_symbol].iloc[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Price", f"${stock_data.get('Current_Last_Sale', 0):.2f}", f"{stock_data.get('Price_Change_Pct', 0):.2f}%")
                        st.write(f"**Sector:** {stock_data.get('Sector', 'N/A')}")
                    with col2:
                        st.metric("Volume", f"{stock_data.get('Current_Volume', 0):,.0f}", f"{stock_data.get('Volume_Change_Pct', 0):.2f}%")
                        st.write(f"**Industry:** {stock_data.get('Industry', 'N/A')}")
                    st.dataframe(pd.DataFrame([stock_data]), use_container_width=True)
            
            st.subheader("Analysis Files")
            if st.session_state.get('analysis_results', {}):
                analyzer = st.session_state.analyzer
                file_tabs = st.tabs(["Price", "Volume", "Sector", "Market Cap", "Other"])
                with file_tabs[0]:
                    files = [f for f in analyzer.created_files if 'price' in f.lower()]
                    for i, f in enumerate(files):
                        with open(f, 'rb') as file:
                            st.download_button(f"Download {os.path.basename(f)}", file.read(), os.path.basename(f), key=f"price_{i}")
                with file_tabs[1]:
                    files = [f for f in analyzer.created_files if 'volume' in f.lower()]
                    for i, f in enumerate(files):
                        with open(f, 'rb') as file:
                            st.download_button(f"Download {os.path.basename(f)}", file.read(), os.path.basename(f), key=f"volume_{i}")
                with file_tabs[2]:
                    files = [f for f in analyzer.created_files if 'sector' in f.lower()]
                    for i, f in enumerate(files):
                        with open(f, 'rb') as file:
                            st.download_button(f"Download {os.path.basename(f)}", file.read(), os.path.basename(f), key=f"sector_{i}")
                with file_tabs[3]:
                    files = [f for f in analyzer.created_files if 'cap' in f.lower()]
                    for i, f in enumerate(files):
                        with open(f, 'rb') as file:
                            st.download_button(f"Download {os.path.basename(f)}", file.read(), os.path.basename(f), key=f"cap_{i}")
                with file_tabs[4]:
                    other_files = [f for f in analyzer.created_files if not any(k in f.lower() for k in ['price', 'volume', 'sector', 'cap'])]
                    for i, f in enumerate(other_files):
                        with open(f, 'rb') as file:
                            st.download_button(f"Download {os.path.basename(f)}", file.read(), os.path.basename(f), key=f"other_{i}")
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for f in analyzer.created_files:
                        if os.path.exists(f):
                            zip_file.write(f, os.path.basename(f))
                zip_buffer.seek(0)
                st.download_button("Download All (ZIP)", zip_buffer, f"snapshot_analysis_{int(time.time())}.zip", mime="application/zip")

    # Sidebar: Reset and Help
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Reset Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key in ['previous_file', 'current_file', 'analyzer', 'filtered_df', 
                          'files_uploaded', 'filters_applied', 'prev_path', 'curr_path',
                          'analysis_results', 'temp_file_paths']:
                    del st.session_state[key]
            if os.path.exists('tmp'):
                for f in os.listdir('tmp'):
                    os.remove(os.path.join('tmp', f))
            st.rerun()
        
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to Use:**
            1. **Upload**: Upload two market snapshots (CSV/Excel).
            2. **Configure**: Set filters and analysis options.
            3. **Analyze**: Select and run analyses.
            4. **Results**: View insights and charts.
            5. **Export**: Download data and reports.
            
            **Tips:**
            - Use snapshots at least a week apart.
            - Adjust filters to focus on specific stocks.
            - Interactive charts support hover and zoom.
            """)

if __name__ == "__main__":
    main()
