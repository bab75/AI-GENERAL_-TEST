# Advanced Analytics User Guide

## Overview

The Advanced Analytics module extends the Stock Analysis Platform with sophisticated analysis capabilities including:

- **Anomaly Detection**: Identify unusual stock behavior using statistical and AI methods
- **Pattern Recognition**: Detect technical chart patterns with confidence scoring
- **Advanced Visualizations**: Create enhanced visual representations of market data
- **Enhanced Report Generation**: Generate comprehensive reports without external dependencies

This guide explains each feature in detail with instructions for optimal use.

## Accessing Advanced Analytics

Navigate to the Advanced Analytics page from the sidebar or by selecting "6_Advanced_Analytics" from the page dropdown.

## Anomaly Detection

The anomaly detection system identifies irregular price and volume movements that may indicate significant market events.

### Key Features

- **Multi-factor Analysis**: Examines price, volume, volatility, and pattern breaks
- **Customizable Parameters**: Adjust sensitivity to match your trading style
- **AI-powered Interpretations**: Receive human-readable explanations (requires API key)
- **Visual Highlighting**: See anomalies clearly marked on price charts

### How to Use

1. Select a stock ticker symbol
2. Choose your desired timeframe
3. Adjust detection parameters:
   - Z-Score Threshold: Controls sensitivity to price deviations
   - Volume Change Threshold: Sets minimum volume spike percentage
   - Analysis Window Size: Determines lookback period
   - Volatility Threshold: Sets sensitivity to volatility changes
4. Enable AI interpretation if desired (requires API key)
5. Click "Detect Anomalies"

### Interpreting Results

The results page shows:
- Price chart with highlighted anomalies
- Detailed information for each detected anomaly
- Interpretation of what each anomaly might indicate
- AI-enhanced analysis (if enabled)

## Pattern Recognition

The pattern recognition system detects common technical chart patterns with confidence scoring.

### Supported Patterns

- **Reversal Patterns**: Head & Shoulders, Inverse Head & Shoulders, Double Top/Bottom
- **Continuation Patterns**: Flags, Pennants, Triangles (Ascending, Descending, Symmetrical)
- **Complex Patterns**: Cup & Handle, Wedges (Rising, Falling)

### How to Use

1. Select a stock ticker symbol
2. Choose your desired timeframe
3. Adjust recognition parameters:
   - Smoothing Period: Controls noise reduction
   - Threshold Percentage: Sets sensitivity for pattern identification
   - Minimum Peak Distance: Prevents over-detection of local extremes
   - Analysis Window Size: Sets the lookback period
4. Click "Detect Patterns"

### Interpreting Results

The results show:
- Primary detected pattern with confidence score
- Visualization of the pattern on the price chart
- Pattern interpretation and trading considerations
- Additional patterns detected (if any)

## Advanced Visualizations

Enhanced visualization capabilities help identify relationships and patterns across multiple stocks and timeframes.

### Visualization Types

#### Correlation Heatmap
Shows correlation between multiple stocks with customizable timeframes and normalization options.

**Usage:**
1. Select multiple stocks (up to 10)
2. Choose timeframe and correlation column (Close, Open, etc.)
3. Select visualization style (Single or Multi-timeframe)
4. Click "Generate Correlation Heatmap"

#### Sector Performance Heatmap
Displays relative performance across market sectors.

**Usage:**
1. Upload a snapshot file with sector information
2. Click "Generate Sector Performance Heatmap"

#### Stock Relationship Network
Visualizes relationships between stocks as an interactive network.

**Usage:**
1. Select network type (Correlation Network, Correlation Clusters, or Sector Network)
2. Select stocks or upload snapshot data (depending on network type)
3. Adjust network parameters
4. Click "Generate Network Graph"

#### Pattern Visualization
Provides enhanced candlestick charts with pattern annotations.

**Usage:**
1. Select a stock and timeframe
2. Choose visualization style
3. Select technical indicators (for Simple Candlestick)
4. Click "Generate Visualization"

#### Prediction Bands
Shows potential price ranges and support/resistance levels.

**Usage:**
1. Select a stock and timeframe
2. Choose prediction type (Bollinger Bands, Monte Carlo, Support/Resistance)
3. Adjust parameters
4. Click "Generate Prediction Bands"

## Enhanced Report Generation

Create comprehensive PDF reports for various analyses without external dependencies.

### Report Types

#### Technical Analysis Report
Generates a detailed report on a single stock's technical indicators and patterns.

**Usage:**
1. Select a stock and timeframe
2. Choose technical indicators to include
3. Click "Generate Technical Analysis Report"

#### Multi-Stock Comparison Report
Creates a comparative analysis of multiple stocks.

**Usage:**
1. Select multiple stocks (2-5 recommended)
2. Choose timeframe and analysis options
3. Click "Generate Comparison Report"

#### Snapshot Comparison Report
Compares market data between two time periods.

**Usage:**
1. Upload previous and current snapshot files
2. Select analysis options
3. Click "Generate Snapshot Report"

## Using AI Features

The platform offers AI-enhanced features that require API keys:

### Setting Up API Keys

1. For Anomaly Detection with AI interpretation:
   - Get an API key from OpenAI or Anthropic
   - Enter the key when prompted, or
   - Set environment variables (OPENAI_API_KEY or ANTHROPIC_API_KEY)

### AI-Enhanced Capabilities

- **Anomaly Interpretation**: Provides context and potential implications
- **Pattern Recognition Enhancement**: Improves detection accuracy
- **Market Insight Generation**: Offers deeper analysis of trends

## Best Practices

- **For Anomaly Detection**: Start with default parameters and adjust as needed
- **For Pattern Recognition**: Use longer timeframes for more reliable pattern detection
- **For Visualizations**: Compare multiple timeframes to confirm trends
- **For Report Generation**: Include diverse indicators for comprehensive analysis

## Troubleshooting

- **No Patterns Detected**: Try adjusting the threshold or increasing the timeframe
- **Too Many Anomalies**: Increase Z-score threshold or decrease sensitivity
- **AI Features Not Working**: Verify API key is correct and has sufficient quota
- **Report Generation Fails**: Ensure sufficient data is available (at least 30 days recommended)

## Advanced Configuration

Power users can modify advanced settings:

- **Custom Indicators**: Add your own technical indicators
- **Detection Parameters**: Fine-tune detection algorithms
- **Visualization Settings**: Customize chart appearance
- **Report Templates**: Modify report formats

---

*For technical details and API documentation, please see the code comments and module docstrings.*