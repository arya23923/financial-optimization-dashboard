# Macro-Aware Portfolio Optimization Dashboard (Streamlit Version)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
import datetime

# Configuration
st.set_page_config(layout="wide")
st.title("Macroeconomic-Aware Portfolio Optimization Dashboard")

# API Setup
FRED_API_KEY = "738a82a300b9041f99eba89da37e14c8"  # Replace with your FRED key
fred = Fred(api_key=FRED_API_KEY)

# Sidebar Inputs
st.sidebar.header("Portfolio Configuration")
tickers = [t.strip().upper() for t in st.sidebar.text_input("Enter comma-separated stock tickers", "AAPL, MSFT, GOOGL, AMZN").split(',')]
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch Financial Data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']  # use 'Close' instead of 'Adj Close'
    return data.dropna()


prices = load_data(tickers, start_date, end_date)

if prices.empty or prices.isnull().all().all():
    st.error("No valid price data loaded. Please check the tickers and date range.")
    st.stop()

returns = prices.pct_change().dropna()

# Fetch Macroeconomic Data
def get_macro_data():
    gdp = fred.get_series("GDP", observation_start=start_date)
    cpi = fred.get_series("CPIAUCSL", observation_start=start_date)
    unemployment = fred.get_series("UNRATE", observation_start=start_date)
    rates = fred.get_series("FEDFUNDS", observation_start=start_date)
    df = pd.concat([gdp, cpi, unemployment, rates], axis=1)
    df.columns = ["GDP", "CPI", "Unemployment", "FedFunds"]
    return df.dropna()

macro_df = get_macro_data()

# Display raw data
with st.expander("Raw Financial Data"):
    st.dataframe(prices)

with st.expander("Macroeconomic Indicators"):
    st.dataframe(macro_df)

# Portfolio Optimization
st.subheader("Efficient Frontier Optimization")
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate=0.0)
cleaned_weights = ef.clean_weights()
perf = ef.portfolio_performance(verbose=True)

# Show weights and performance
st.write("### Optimized Portfolio Weights")
st.write(cleaned_weights)
st.write("### Portfolio Performance")
st.write(f"Expected annual return: {perf[0]:.2f}")
st.write(f"Annual volatility: {perf[1]:.2f}")
st.write(f"Sharpe Ratio: {perf[2]:.2f}")

# Visualize weights
fig, ax = plt.subplots()
ax.pie(cleaned_weights.values(), labels=cleaned_weights.keys(), autopct='%1.1f%%')
ax.set_title("Asset Allocation")
st.pyplot(fig)

# Correlation with Macroeconomic Data
st.subheader("Correlation with Macroeconomic Indicators")
macro_monthly = macro_df.resample('M').mean()
returns_monthly = returns.resample('M').mean()
avg_returns = returns_monthly.mean(axis=1)
combined = pd.concat([avg_returns, macro_monthly], axis=1).dropna()
combined.columns = ["Average Returns", "GDP", "CPI", "Unemployment", "FedFunds"]
corr = combined.corr()
st.write(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot()

# Risk Metrics
st.subheader("Risk Analysis")
value_at_risk = np.percentile(returns.sum(axis=1), 5)
st.write(f"5% Value at Risk (VaR): {value_at_risk:.4f}")

# Footer
st.markdown("---")
st.caption("Developed with PyPortfolioOpt, FRED API, and Streamlit")
