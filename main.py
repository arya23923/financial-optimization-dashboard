import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Risk Analysis Dashboard", layout="wide")

# Sidebar Configuration
st.sidebar.header("Portfolio Configuration")

start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date", datetime.today())

tickers_input = st.sidebar.text_area("Enter Stock Tickers (one per line)", "AAPL\nMSFT\nGOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.splitlines() if ticker.strip()]

risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

submit = st.sidebar.button("Submit Tickers")

if submit and len(tickers) > 1:
    st.title("Financial Risk Analysis Dashboard")

    # Fetch price data
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        adj_close_df[ticker] = data['Adj Close']

    # Log Returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Covariance matrix
    cov_matrix = log_returns.cov() * 252

    # Portfolio functions
    def standard_deviation(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def expected_return(weights):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights):
        return (expected_return(weights) - risk_free_rate_input) / standard_deviation(weights)

    def neg_sharpe_ratio(weights):
        return -sharpe_ratio(weights)

    # Optimization
    bounds = [(0, 1.0) for _ in tickers]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    initial_weights = np.array([1 / len(tickers)] * len(tickers))

    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    # Results
    st.subheader("Portfolio Summary")

    df_weights = pd.DataFrame({
        'Ticker': tickers,
        'Weight (%)': optimal_weights * 100
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current Portfolio Allocation")
        fig1, ax1 = plt.subplots()
        ax1.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Portfolio Performance Metrics")
        st.metric("Expected Annual Return", f"{expected_return(optimal_weights):.2%}")
        st.metric("Annual Volatility", f"{standard_deviation(optimal_weights):.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio(optimal_weights):.2f}")

    # Comparison Chart
    st.subheader("Portfolio Weights Comparison")
    fig2, ax2 = plt.subplots(figsize = (10,5))
    bar_width = 0.4
    bars = ax2.bar(tickers, optimal_weights * 100, color='skyblue', width=bar_width)
    ax2.set_ylabel("Weight (%)", fontsize=10)
    ax2.set_title("Optimized Portfolio Allocation", fontsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("Stock Price Trends (Last 5 Years)")
    normalized_prices = adj_close_df / adj_close_df.iloc[0] * 100
    fig_price, ax_price = plt.subplots(figsize=(10, 5))

    
    for ticker in normalized_prices.columns:
        ax_price.plot(normalized_prices.index, normalized_prices[ticker], label=ticker)

    ax_price.set_title("Normalized Adjusted Close Prices (Base = 100)")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Normalized Price")
    ax_price.legend(loc="upper left", fontsize="small")
    ax_price.grid(True)

    # Streamlit chart output
    st.pyplot(fig_price)

else:
    st.warning("Enter at least 2 tickers and click 'Submit Tickers'.")

