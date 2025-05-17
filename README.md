# Stock Analysis Platform

## Overview

A comprehensive stock analysis platform that combines advanced technical analysis, machine learning predictive insights, and interactive data visualization for informed investment decision-making.

## Features

### Technical Analysis
- Price chart visualization with customizable timeframes
- Multiple technical indicators (Moving Averages, RSI, MACD, etc.)
- Support and resistance level detection

### Multi-Stock Comparison
- Correlation analysis between multiple stocks
- Performance comparison across different metrics
- Risk assessment and visualization

### Options Analysis
- Options chain visualization
- Strike price analysis
- Expiration date evaluation

### Data Export
- Export to Excel, CSV, and PDF formats
- Customizable report generation
- Data visualization export

### Snapshot Comparison
- Compare market snapshots from different time periods
- Sector performance analysis
- Volume and price movement analysis

### Advanced Analytics
- **Anomaly Detection**: Identify unusual market behavior
- **Pattern Recognition**: Detect technical chart patterns
- **Advanced Visualizations**: Heatmaps, network graphs, and specialized charts
- **Enhanced Reporting**: Generate comprehensive analysis reports

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/stock-analysis-platform.git
cd stock-analysis-platform
pip install -r requirements.txt
```

## Usage

Run the application using Streamlit:

```bash
streamlit run Home.py
```

Navigate to the URL provided in the terminal output to access the application.

## Data Sources

The platform uses the following data sources:

- **Yahoo Finance** (via yfinance): Live market data
- **User-provided CSV/Excel files**: For snapshot comparison analysis

## Advanced Features

### AI-Powered Analysis

The platform offers AI-enhanced features that require API keys:

- **Anomaly Interpretation**: Provides context for detected anomalies
- **Pattern Recognition**: Improves detection of chart patterns

To use these features, you'll need to provide API keys for OpenAI or Anthropic.

### Enhanced Visualization

- **Correlation Heatmaps**: Visualize relationships between multiple stocks
- **Stock Networks**: Graph-based visualization of market relationships
- **Prediction Bands**: Statistical forecasting tools and confidence intervals

## Documentation

For detailed usage instructions, see the [Advanced Analytics Guide](docs/advanced_analytics_guide.md).

## License

MIT

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Visualization powered by [Plotly](https://plotly.com/)