"""
Network Graph Visualization Module

This module provides network graph visualization capabilities for stock relationships,
sector correlations, and other network-based analyses.
"""

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgba

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_correlation_network(data: Dict[str, pd.DataFrame], 
                              column: str = 'Close',
                              min_correlation: float = 0.7,
                              max_edges: int = 100,
                              period: str = 'all',
                              normalize: bool = True,
                              title: str = 'Stock Correlation Network') -> go.Figure:
    """
    Create a network graph visualizing correlations between stocks.
    
    Args:
        data: Dictionary with ticker symbols as keys and DataFrames as values
        column: Column to use for correlation (default: 'Close')
        min_correlation: Minimum correlation value to show as an edge
        max_edges: Maximum number of edges to display
        period: Time period to analyze ('all' or number of days)
        normalize: Whether to normalize prices before correlation
        title: Plot title
        
    Returns:
        Plotly figure object with correlation network
    """
    # Create a new DataFrame with the closing prices of each stock
    df_combined = pd.DataFrame()
    
    for ticker, df in data.items():
        if df.empty or column not in df.columns:
            continue
            
        # Limit to the specified period
        if period != 'all' and period.isdigit():
            df = df.tail(int(period))
            
        # Use the specified column or fall back to 'Close'
        values = df[column].copy()
        
        # Normalize if requested
        if normalize:
            values = values / values.iloc[0] if not values.empty else values
            
        df_combined[ticker] = values
    
    # Calculate correlation matrix
    correlation_matrix = df_combined.corr()
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (stocks)
    for ticker in correlation_matrix.columns:
        G.add_node(ticker)
    
    # Add edges for correlations above threshold
    edges = []
    for i, ticker1 in enumerate(correlation_matrix.columns):
        for j, ticker2 in enumerate(correlation_matrix.columns):
            if i < j:  # Only process each pair once
                corr = correlation_matrix.loc[ticker1, ticker2]
                if abs(corr) >= min_correlation:
                    edges.append((ticker1, ticker2, abs(corr), corr > 0))
    
    # Sort edges by correlation strength and limit to max_edges
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]
    
    # Add edges to graph
    for ticker1, ticker2, weight, is_positive in edges:
        G.add_edge(ticker1, ticker2, weight=weight, is_positive=is_positive)
    
    # Calculate node positions using a force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces (positive and negative correlations with different colors)
    edge_traces = []
    
    # Positive correlations (green)
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_texts = []
    
    # Negative correlations (red)
    neg_edge_x = []
    neg_edge_y = []
    neg_edge_weights = []
    neg_edge_texts = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        is_positive = edge[2].get('is_positive', True)
        
        if is_positive:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(weight)
            edge_texts.append(f"{edge[0]} - {edge[1]}: {weight:.2f}")
        else:
            neg_edge_x.extend([x0, x1, None])
            neg_edge_y.extend([y0, y1, None])
            neg_edge_weights.append(weight)
            neg_edge_texts.append(f"{edge[0]} - {edge[1]}: {-weight:.2f}")
    
    # Create positive edge trace
    if edge_x:
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(50,200,50,0.8)'),
            hoverinfo='text',
            text=edge_texts,
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create negative edge trace
    if neg_edge_x:
        neg_edge_trace = go.Scatter(
            x=neg_edge_x, y=neg_edge_y,
            line=dict(width=1, color='rgba(200,50,50,0.8)'),
            hoverinfo='text',
            text=neg_edge_texts,
            mode='lines'
        )
        edge_traces.append(neg_edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                title=dict(text='Connections', side='right'),
                thickness=15,
                xanchor='left'
            ),
            line=dict(width=2)
        ),
        hoverinfo='text',
        hovertext=[f"{node}: {len(list(G.neighbors(node)))} connections" for node in G.nodes()]
    )
    
    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=700,
                       width=900
                   ))
    
    # Add a legend for edge colors
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(50,200,50,0.8)', width=2),
        name='Positive Correlation'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(200,50,50,0.8)', width=2),
        name='Negative Correlation'
    ))
    
    fig.update_layout(showlegend=True)
    
    return fig

def create_sector_network(data: pd.DataFrame, 
                         sector_column: str = 'Sector',
                         performance_column: str = 'Price_Change_Pct',
                         size_column: str = 'Market_Cap',
                         min_stocks: int = 3,
                         title: str = 'Sector Performance Network') -> go.Figure:
    """
    Create a network graph showing relationships between sectors and stocks.
    
    Args:
        data: DataFrame with stock data including sector and performance metrics
        sector_column: Column name for sector
        performance_column: Column name for performance metric
        size_column: Column name for node size
        min_stocks: Minimum number of stocks in a sector to include
        title: Plot title
        
    Returns:
        Plotly figure object with sector network
    """
    if sector_column not in data.columns or performance_column not in data.columns:
        logger.error(f"Required columns not found in data")
        return go.Figure()
    
    # Filter out sectors with too few stocks
    sector_counts = data[sector_column].value_counts()
    valid_sectors = sector_counts[sector_counts >= min_stocks].index.tolist()
    filtered_data = data[data[sector_column].isin(valid_sectors)].copy()
    
    if filtered_data.empty:
        logger.error(f"No sectors with at least {min_stocks} stocks")
        return go.Figure()
    
    # Create a graph
    G = nx.Graph()
    
    # Add sector nodes
    for sector in valid_sectors:
        # Calculate average performance for the sector
        sector_perf = filtered_data[filtered_data[sector_column] == sector][performance_column].mean()
        G.add_node(sector, type='sector', performance=sector_perf)
    
    # Add stock nodes and edges to sectors
    for _, row in filtered_data.iterrows():
        stock = row.get('Symbol', row.name)
        sector = row[sector_column]
        performance = row[performance_column]
        size = row.get(size_column, 1)
        
        # Add the stock node
        G.add_node(stock, type='stock', performance=performance, size=size)
        
        # Connect stock to its sector
        G.add_edge(stock, sector)
    
    # Calculate node positions using a force-directed layout with sectors at the center
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Separate nodes by type
    sector_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'sector']
    stock_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'stock']
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create sector node trace
    sector_x = [pos[node][0] for node in sector_nodes]
    sector_y = [pos[node][1] for node in sector_nodes]
    sector_performances = [G.nodes[node].get('performance', 0) for node in sector_nodes]
    
    # Determine color scale for performance
    max_abs_perf = max(abs(min(sector_performances)), abs(max(sector_performances)))
    
    sector_trace = go.Scatter(
        x=sector_x, y=sector_y,
        mode='markers+text',
        text=sector_nodes,
        textposition='middle center',
        marker=dict(
            size=30,
            color=sector_performances,
            colorscale='RdYlGn',
            cmin=-max_abs_perf,
            cmax=max_abs_perf,
            colorbar=dict(
                title=dict(text='Performance %', side='right'),
                thickness=15,
                xanchor='left'
            ),
            line=dict(width=2, color='#000')
        ),
        hoverinfo='text',
        hovertext=[f"{node}: {perf:.2f}%" for node, perf in zip(sector_nodes, sector_performances)],
        name='Sectors'
    )
    
    # Create stock node trace
    stock_x = [pos[node][0] for node in stock_nodes]
    stock_y = [pos[node][1] for node in stock_nodes]
    stock_performances = [G.nodes[node].get('performance', 0) for node in stock_nodes]
    stock_sizes = [min(20, max(5, G.nodes[node].get('size', 1) / 10e9)) for node in stock_nodes]  # Scale market cap to reasonable marker sizes
    
    stock_trace = go.Scatter(
        x=stock_x, y=stock_y,
        mode='markers',
        marker=dict(
            size=stock_sizes,
            color=stock_performances,
            colorscale='RdYlGn',
            cmin=-max_abs_perf,
            cmax=max_abs_perf,
            line=dict(width=1, color='#000')
        ),
        hoverinfo='text',
        hovertext=[f"{node}: {perf:.2f}%" for node, perf in zip(stock_nodes, stock_performances)],
        name='Stocks'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, sector_trace, stock_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=800,
                       width=1000
                   ))
    
    return fig

def create_correlation_cluster_map(data: Dict[str, pd.DataFrame], 
                                  column: str = 'Close',
                                  period: str = 'all',
                                  n_clusters: int = 5,
                                  title: str = 'Stock Correlation Clusters') -> go.Figure:
    """
    Create a network graph with stocks clustered by correlation similarity.
    
    Args:
        data: Dictionary with ticker symbols as keys and DataFrames as values
        column: Column to use for correlation
        period: Time period to analyze ('all' or number of days)
        n_clusters: Number of clusters to identify
        title: Plot title
        
    Returns:
        Plotly figure object with clustered correlation network
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        logger.error("scikit-learn is required for clustering")
        return go.Figure()
    
    # Create a new DataFrame with the closing prices of each stock
    df_combined = pd.DataFrame()
    
    for ticker, df in data.items():
        if df.empty or column not in df.columns:
            continue
            
        # Limit to the specified period
        if period != 'all' and period.isdigit():
            df = df.tail(int(period))
            
        # Use the specified column and normalize
        values = df[column].copy()
        values = values / values.iloc[0] if not values.empty else values
            
        df_combined[ticker] = values
    
    # Calculate correlation matrix
    correlation_matrix = df_combined.corr()
    
    # Convert correlation to distance (1 - correlation for positive correlations)
    distance_matrix = 1 - correlation_matrix.abs()
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(correlation_matrix)),
        affinity='precomputed',
        linkage='average'
    )
    
    # Fit clustering model
    clustering.fit(distance_matrix)
    
    # Add cluster labels to the correlation matrix
    correlation_matrix['cluster'] = clustering.labels_
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (stocks) with cluster information
    for ticker, cluster in zip(correlation_matrix.index, clustering.labels_):
        G.add_node(ticker, cluster=int(cluster))
    
    # Add edges for high correlations
    for i, ticker1 in enumerate(correlation_matrix.index):
        for j, ticker2 in enumerate(correlation_matrix.index):
            if i < j:  # Only process each pair once
                corr = correlation_matrix.loc[ticker1, ticker2]
                # Add edge if correlation is high (positive or negative)
                if abs(corr) > 0.5:
                    G.add_edge(ticker1, ticker2, weight=abs(corr), is_positive=corr > 0)
    
    # Calculate node positions using a force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_widths = []
    edge_texts = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        is_positive = edge[2].get('is_positive', True)
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Set color based on positive/negative correlation
        color = 'rgba(50,200,50,0.4)' if is_positive else 'rgba(200,50,50,0.4)'
        edge_colors.extend([color, color, 'rgba(0,0,0,0)'])
        
        # Set width based on correlation strength
        width = weight * 2
        edge_widths.extend([width, width, 0])
        
        edge_texts.append(f"{edge[0]} - {edge[1]}: {weight:.2f}")
    
    # Create custom edge trace with varying colors and widths
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color=edge_colors, width=edge_widths),
        hoverinfo='text',
        text=edge_texts,
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    
    # Define color map for clusters
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Set color based on cluster
        cluster = attrs.get('cluster', 0)
        color = cluster_colors[cluster % len(cluster_colors)]
        node_colors.append(color)
        
        # Set size based on number of connections
        size = 10 + len(list(G.neighbors(node))) * 2
        node_sizes.append(size)
        
        node_texts.append(f"{node} (Cluster {cluster+1}): {len(list(G.neighbors(node)))} connections")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#000')
        ),
        hoverinfo='text',
        hovertext=node_texts
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=800,
                       width=1000
                   ))
    
    # Add a legend for clusters
    for i in range(min(n_clusters, len(correlation_matrix))):
        color = cluster_colors[i % len(cluster_colors)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=f'Cluster {i+1}'
        ))
    
    # Add a legend for edge types
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(50,200,50,0.8)', width=2),
        name='Positive Correlation'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(200,50,50,0.8)', width=2),
        name='Negative Correlation'
    ))
    
    fig.update_layout(showlegend=True)
    
    return fig

def create_market_map(data: pd.DataFrame,
                     size_column: str = 'Market_Cap',
                     performance_column: str = 'Price_Change_Pct',
                     sector_column: str = 'Sector',
                     title: str = 'Market Map') -> go.Figure:
    """
    Create a treemap visualization of the market landscape.
    
    Args:
        data: DataFrame with stock data
        size_column: Column name for box size
        performance_column: Column name for box color
        sector_column: Column name for grouping
        title: Plot title
        
    Returns:
        Plotly figure object with market map
    """
    if size_column not in data.columns or performance_column not in data.columns or sector_column not in data.columns:
        logger.error(f"Required columns not found in data")
        return go.Figure()
    
    # Remove rows with NaN values in key columns
    clean_data = data.dropna(subset=[size_column, performance_column, sector_column]).copy()
    
    if clean_data.empty:
        logger.error("No clean data available after removing NaN values")
        return go.Figure()
    
    # Get symbol column if available, otherwise use index
    if 'Symbol' in clean_data.columns:
        symbol_col = 'Symbol'
    else:
        clean_data['Symbol'] = clean_data.index
        symbol_col = 'Symbol'
    
    # Add company name if available
    if 'Name' in clean_data.columns:
        name_col = 'Name'
    else:
        clean_data['Name'] = clean_data[symbol_col]
        name_col = 'Name'
    
    # Create labels
    clean_data['label'] = clean_data.apply(
        lambda row: f"{row[symbol_col]}: {row[name_col][:20]}{'...' if len(row[name_col]) > 20 else ''}", 
        axis=1
    )
    
    # Create hover text
    clean_data['hover'] = clean_data.apply(
        lambda row: (
            f"Symbol: {row[symbol_col]}<br>"
            f"Name: {row[name_col]}<br>"
            f"Sector: {row[sector_column]}<br>"
            f"Market Cap: ${row[size_column]:,.0f}<br>"
            f"Performance: {row[performance_column]:.2f}%"
        ),
        axis=1
    )
    
    # Determine color scale range
    max_abs_perf = max(abs(clean_data[performance_column].min()), abs(clean_data[performance_column].max()))
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=clean_data['label'],
        parents=['' for _ in range(len(clean_data))],  # No parent hierarchy
        values=clean_data[size_column],
        textinfo='label',
        hovertext=clean_data['hover'],
        hoverinfo='text',
        marker=dict(
            colors=clean_data[performance_column],
            colorscale='RdYlGn',
            cmin=-max_abs_perf,
            cmax=max_abs_perf,
            colorbar=dict(
                title='Performance %',
                thickness=15,
                lenmode='fraction',
                len=0.8
            )
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig