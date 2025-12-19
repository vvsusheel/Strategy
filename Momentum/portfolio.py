import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def prepare_price_data(data):
    """
    Converts yfinance multi-index dataframe into
    Close price dataframe: rows=dates, cols=tickers
    """
    close = data["Close"].copy()
    close = close.dropna(axis=1, how="all")
    return close

def momentum_signal(prices, lookback=30):
    momentum = {}

    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < lookback + 1:
            continue

        momentum[col] = (series.iloc[-1] / series.iloc[-lookback]) - 1

    return pd.Series(momentum)


def get_month_end_dates(prices):
    return prices.index.to_series().resample("ME").last()

def construct_portfolio(signal, top_n=10):
    """
    signal: Series (ticker -> score)
    returns: Series (ticker -> weight)
    """
    selected = signal.sort_values(ascending=False).head(top_n)
    weights = pd.Series(1 / len(selected), index=selected.index)
    return weights

def backtest(
    prices,
    lookback=20,
    top_n=10
):
    monthly_dates = get_month_end_dates(prices)

    portfolio_returns = []

    for i in range(1, len(monthly_dates)):
        rebalance_date = monthly_dates.iloc[i - 1]
        next_date = monthly_dates.iloc[i]

        price_slice = prices.loc[:rebalance_date]
        signal = momentum_signal(price_slice, lookback)

        if signal.empty:
            continue

        weights = construct_portfolio(signal, top_n)
        print(f"[{rebalance_date.date()}] selected: {weights.index.tolist()}")

        start_prices = prices.loc[rebalance_date, weights.index]
        end_prices = prices.loc[next_date, weights.index]

        returns = (end_prices / start_prices - 1)
        portfolio_return = (returns * weights).sum()

        portfolio_returns.append(
            {
                "Date": next_date,
                "Return": portfolio_return
            }
        )

    return pd.DataFrame(portfolio_returns).set_index("Date")

def performance_stats(returns):
    cumulative = (1 + returns).cumprod()
    cagr = cumulative.iloc[-1] ** (12 / len(returns)) - 1
    max_dd = (cumulative / cumulative.cummax() - 1).min()

    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Final Value ($)": round(cumulative.iloc[-1], 2)
    }

def calculate_rolling_returns(returns, window=3):
    # (1+r1)*(1+r2)*(1+r3) - 1
    rolling = (1 + returns).rolling(window=window).apply(lambda x: x.prod(), raw=True) - 1
    return rolling.dropna()

# -----------------------------
# Step 1: Get NIFTY 500 symbols
# -----------------------------
import requests
import io

url = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
stocks = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

symbols = [x for x in stocks["Symbol"].tolist() if not str(x).startswith("DUMMY")]
tickers = [s + ".NS" for s in symbols]

print(f"Total stocks: {len(tickers)}")

# -----------------------------
# Step 2: Download price data
# -----------------------------
end_date = datetime.today()
start_date = end_date - timedelta(days=4800)

data = yf.download(
    tickers,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
    threads=True
)

print("Data downloaded")
# -----------------------------
# Prepare prices
# -----------------------------
prices = prepare_price_data(data)

# -----------------------------
# Run backtest
# -----------------------------
returns_df = backtest(
    prices,
    lookback=30,
    top_n=10
)

print("\nMonthly Returns:")
print(returns_df)

# -----------------------------
# Performance
# -----------------------------
stats = performance_stats(returns_df["Return"])
print("\nPerformance Stats:")
print(stats)

print("\n12-Month Rolling Returns:")
rolling_12m = calculate_rolling_returns(returns_df["Return"], window=12)
print(rolling_12m)

import plotly.graph_objects as go

# -----------------------------
# Plotting (Interactive)
# -----------------------------
equity_curve = 100 * (1 + returns_df["Return"]).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=equity_curve.index,
    y=equity_curve.values,
    mode='lines',
    name='Portfolio Value',
    hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title="Portfolio Growth (Initial Investment: $100)",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    template="plotly_dark"
)

fig.write_html("portfolio_growth.html")
print("\nInteractive plot saved to portfolio_growth.html")
