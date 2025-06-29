import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def generate_urls(symbol):
    base_urls = {
        "zacks": f"https://www.zacks.com/stock/quote/{symbol}?q={symbol}",
        "barchart": f"https://www.barchart.com/stocks/quotes/{symbol}/overview",
        "tipranks": f"https://www.tipranks.com/stocks/{symbol}/forecast",
        "StockAnalysis": f"https://www.stockanalysis.com/stocks/{symbol}/",
        "finviz": f"https://www.finviz.com/quote.ashx?t={symbol}&p=d",
        "stockcharts": f"https://www.stockcharts.com/sc3/ui/?s={symbol}",
        "stockrover": f"https://www.stockrover.com/world/insight/summary/Quotes={symbol}",
        "altindex": f"https://www.altindex.com/ticker/{symbol}",
        "earningshub": f"https://earningshub.com/quote/{symbol}",
        "GuruFocus": f"https://www.gurufocus.com/stock/{symbol}/summary?search",
        "ziggma": f"https://app.ziggma.com/securities/{symbol}",
        "marketchameleon": f"https://marketchameleon.com/Overview/{symbol}",
        "moomoo": f"https://www.moomoo.com/us",
        "finra": f"https://finra-markets.morningstar.com/MarketData/EquityOptions/",
        "wallmine(old data)": f"https://wallmine.com/market/us",
        "stockinvestoriq": f"https://stockinvestoriq.com/stock-analysis/#sa",
        "tickeron": f"https://tickeron.com/ticker/{symbol}/",
        "wallstreetzen": f"https://www.wallstreetzen.com/",       
        "profitviz": f"https://profitviz.com/{symbol}",
        "finchat": f"https://finchat.io/dashboard/",       
        "biztoc": f"https://biztoc.com",      
        "yahoo_finance": f"https://finance.yahoo.com/quote/{symbol}",
        "seeking_alpha": f"https://seekingalpha.com/symbol/{symbol}",
        "marketwatch": f"https://www.marketwatch.com/investing/stock/{symbol}",
        "tradingview": f"https://www.tradingview.com/chart/?symbol={symbol}",
        "fintel": f"https://fintel.io/s/us/{symbol}",
        "benzinga": f"https://www.benzinga.com/quote/{symbol}",
        "cnbc": f"https://www.cnbc.com/quotes/{symbol}",
        "compoundeer": f"https://compoundeer.com/company/{symbol}",
        "nvstly": f"https://nvstly.com/",
        "Penny-stock-screener": f"https://stock-screener.org/",
        "faktia": f"https://faktia.com/stocks/{symbol}",
        "macrotrends": f"https://www.macrotrends.net/",
        "revvinvest": f"https://revvinvest.com/",
        "alphaspread(back-testing option)": f"https://www.alphaspread.com/",
        "moomoo": f"https://www.moomoo.com/us",
        "crowdbullish": f"https://crowdbullish.com/quote/{symbol}",
        "streamlined": f"https://www.streamlined.finance/symbol/{symbol}",  
        "option-visualizer": f"https://www.optionvisualizer.com/home",  
        "chartmill": f"https://www.chartmill.com/stock/quote/{symbol}", 
        "rafa.ai": f"https://www.rafa.ai/", 
        "etf-screener": f"https://www.etf.com/etfanalytics/etf-screener", 
        "usnews.com": f"https://www.usnews.com/",        
        "simfin": f"https://app.simfin.com/", 
        "quiverquant": f"https://www.quiverquant.com/home/",
        "simplywall.st": f"https://simplywall.st/community/narratives",
        "public.com": f"https://public.com/stocks/{symbol}", 
        "whalewisdom.com": f"https://whalewisdom.com/stock/{symbol}",
        "Research-Links": f"https://nftrh.com/links/", 
        "stockinvest.us(AI)": f"https://stockinvest.us/stock/{symbol}",
        "stocktwits.com(sentiment)": f"https://stocktwits.com/symbol/{symbol}", 
        "swingtradebot.com(information)": f"https://swingtradebot.com/equities/{symbol}",
        "simfin": f"https://app.simfin.com/",
        "nexustrade.io(AI)": f"https://nexustrade.io/stock/{symbol}"        
        
      }
    return base_urls
def generate_ytd_stock_data(symbol):
    # Get the current date
    end_date = datetime.now()
    # Get the start of the current year
    start_date = datetime(end_date.year, 1, 1)
    
    # Fetch data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    return df

def plot_ytd_stock_chart(symbol):
    df = generate_ytd_stock_data(symbol)
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol.upper()} YTD Stock Price', 'Volume'),
                        row_width=[0.7, 0.3])

    # Add candlestick
    fig.add_trace(go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='OHLC'),
                row=1, col=1)

    # Add MA20
    ma20 = df['Close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=df['Date'], y=ma20, opacity=0.7, line=dict(color='blue', width=2), name='MA 20'),
                  row=1, col=1)

    # Add volume bar chart
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, showlegend=False), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} YTD Stock Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=12),
        dragmode='zoom',
        hovermode='x unified',
        autosize=True,
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Update x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig


def main():
    st.set_page_config(page_title="Stock Symbol Information Viewer", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .section-header {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
        font-weight: bold;
    }
    .analysis-tools { background-color: #3498db; }
    .charts-analysis { background-color: #2ecc71; }
    .earnings-financials { background-color: #e67e22; }
    .social-investing { background-color: #3498db; }
    .news-opinion { background-color: #e74c3c; }
    .other-resources { background-color: #9b59b6; }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .link-card {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .link-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    # Lottie Animation
    #lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
   # lottie_json = load_lottieurl(lottie_url)
   # st_lottie(lottie_json, height=200)

    st.title("📈 Stock Symbol Information Viewer")
    
    # Sidebar navigation
   # with st.sidebar:
    #    selected = option_menu(
     #       menu_title="Navigation",
     #       options=["Home", "About", "Contact"],
       #     icons=["house", "info-circle", "envelope"],
        #    menu_icon="cast",
         #   default_index=0,
      #  )

    symbol = st.text_input("🔍 Enter Stock Symbol:", "")
    
    if st.button("Submit"):
        if symbol:
            urls = generate_urls(symbol)
            
            st.write(f"## Results for: **{symbol.upper()}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="section-header analysis-tools">📊 Analysis Tools</div>', unsafe_allow_html=True)
                for site in ["profitviz","crowdbullish","streamlined","revvinvest","StockAnalysis","zacks","barchart","marketchameleon","finviz","finra","public.com","whalewisdom.com",
                              "wallstreetzen","tipranks","compoundeer","stockrover","altindex","stockinvest.us(AI)","stocktwits.com(sentiment)","swingtradebot.com(information)","simfin"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.capitalize()}</a></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="section-header charts-analysis">📈 Charts and Technical Analysis</div>', unsafe_allow_html=True)
                for site in ["macrotrends","faktia","alphaspread(back-testing option)","moomoo","simfin","quiverquant","stockcharts","finchat","tradingview","fintel","Penny-stock-screener"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.capitalize()}</a></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-header earnings-financials">💰 Earnings and Financials</div>', unsafe_allow_html=True)
                for site in ["earningshub","marketwatch"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.capitalize()}</a></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header social-investing">👥 Social Investing</div>', unsafe_allow_html=True)
                for site in ["nvstly","rafa.ai","simplywall.st"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.capitalize()}</a></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="section-header news-opinion">📰 News and Opinion</div>', unsafe_allow_html=True)
                for site in ["biztoc", "yahoo_finance","seeking_alpha","benzinga","cnbc","usnews.com"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.replace("_", " ").capitalize()}</a></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section-header other-resources">🔗 Other Resources</div>', unsafe_allow_html=True)
                for site in ["ziggma","option-visualizer","etf-screener","moomoo","stockinvestoriq","chartmill","tickeron","nexustrade.io","GuruFocus","Research-Links","wallmine(old data)"]:
                    st.markdown(f'<div class="link-card"><a href="{urls[site]}" target="_blank">{site.capitalize()}</a></div>', unsafe_allow_html=True)
            
           # Replace the existing sample stock data visualization with this:
            fig = plot_ytd_stock_chart(symbol)
            st.plotly_chart(fig, use_container_width=True)

            # st.info("Click on the links above to open the respective websites in a new tab.")
        else:
            st.error("⚠️ Please enter a stock symbol!")

if __name__ == "__main__":
    main()
    TEST

