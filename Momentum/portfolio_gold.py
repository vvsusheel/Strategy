import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import io
import plotly.graph_objects as go

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

def construct_portfolio(signal, previous_streaks, top_n=10):
    """
    signal: Series (ticker -> score)
    previous_streaks: Dict (ticker -> consecutive_count)
    returns: 
        weights: Series (ticker -> weight)
        current_streaks: Dict (ticker -> consecutive_count)
    """
    selected = signal.sort_values(ascending=False).head(top_n)
    selected_tickers = selected.index.tolist()
    
    scores = {}
    current_streaks = {}
    
    for ticker in selected_tickers:
        # Get consecutive count from previous month (default 0)
        streak = previous_streaks.get(ticker, 0)
        
        # Calculate score: Base 1.0 + 0.1 per repeat
        bonus = streak * 0.0 # disabled bonus
        raw_score = 1.0 + bonus
        
        # Cap max weightage score to 2
        capped_score = min(2.0, raw_score)
        
        scores[ticker] = capped_score
        current_streaks[ticker] = streak + 1
        
    # Normalize weights
    total_score = sum(scores.values())
    weights_dict = {k: v / total_score for k, v in scores.items()}
    
    weights = pd.Series(weights_dict)
    return weights, current_streaks

def backtest(
    prices,
    gold_prices,
    nifty_prices,
    lookback=20,
    top_n=10,
    equity_weight=0.75,
    gold_weight=0.25,
    enable_shuffle=False,
    shuffle_months=2
):
    monthly_dates = get_month_end_dates(prices)
    # Ensure aligned data
    gold_prices = gold_prices.reindex(prices.index, method='ffill')
    nifty_prices = nifty_prices.reindex(prices.index, method='ffill')

    portfolio_returns = []
    
    # State for Recurrence Bonus
    streaks = {}
    
    # State for Shuffle Strategy
    # 'MOMENTUM' or 'NIFTY'
    current_mode = 'NIFTY' 
    # History of monthly returns: list of (momentum_ret, nifty_ret)
    perf_history = [] 

    for i in range(1, len(monthly_dates)):
        rebalance_date = monthly_dates.iloc[i - 1]
        next_date = monthly_dates.iloc[i]

        # -------------------------------------------
        # 1. Determine Strategy Mode for THIS month
        # -------------------------------------------
        if enable_shuffle and len(perf_history) >= shuffle_months:
            # Check last N months
            last_n = perf_history[-shuffle_months:]
            
            # Condition: Momentum < Nifty for N consecutive months -> Switch to NIFTY
            mom_underperforms = all(m < n for m, n in last_n)
            
            # Condition: Momentum > Nifty for N consecutive months -> Switch to MOMENTUM
            mom_outperforms = all(m > n for m, n in last_n)
            
            if current_mode == 'MOMENTUM' and mom_underperforms:
                current_mode = 'NIFTY'
                print(f"[{rebalance_date.date()}] SWITCHING to NIFTY (Momentum underperformed {shuffle_months}mo)")
            elif current_mode == 'NIFTY' and mom_outperforms:
                current_mode = 'MOMENTUM'
                print(f"[{rebalance_date.date()}] SWITCHING to MOMENTUM (Momentum outperformed {shuffle_months}mo)")

        # -------------------------------------------
        # 2. Calculate MOMENTUM Return (Hypothetical or Real)
        # -------------------------------------------
        price_slice = prices.loc[:rebalance_date]
        signal = momentum_signal(price_slice, lookback)

        if signal.empty:
            # If no signal possible, momentum return is 0 (cash) or Nifty fallback? 
            # Assume 0 or market
            mom_ret = 0.0
            equity_portfolio_return = 0.0 # Placeholder
        else:
            weights, streaks = construct_portfolio(signal, streaks, top_n)
            
            start_prices = prices.loc[rebalance_date, weights.index]
            end_prices = prices.loc[next_date, weights.index]
            equity_returns = (end_prices / start_prices - 1)
            # Calculated momentum return (needed for history in ALL cases)
            mom_ret = (equity_returns * weights).sum()

        # -------------------------------------------
        # 3. Calculate NIFTY Return (Hypothetical or Real)
        # -------------------------------------------
        try:
             n_start = nifty_prices.loc[rebalance_date]
             n_end = nifty_prices.loc[next_date]
             nifty_ret = (n_end / n_start) - 1
        except (KeyError, ValueError):
             nifty_ret = 0.0

        # Store history for next month's decision
        perf_history.append((mom_ret, nifty_ret))
        
        # -------------------------------------------
        # 4. Actual Equity Component Return Selection
        # -------------------------------------------
        if current_mode == 'MOMENTUM':
             selected_equity_return = mom_ret
        else:
             selected_equity_return = nifty_ret

        # -------------------------------------------
        # 5. Gold Component
        # -------------------------------------------
        try:
            gold_start = gold_prices.loc[rebalance_date]
            gold_end = gold_prices.loc[next_date]
            if hasattr(gold_start, "item"): gold_start = gold_start.item()
            if hasattr(gold_end, "item"): gold_end = gold_end.item()
            gold_return = (gold_end / gold_start - 1)
        except (KeyError, ValueError):
             gold_return = 0.0

        # -------------------------------------------
        # 6. Total Return
        # -------------------------------------------
        total_return = equity_weight * selected_equity_return + gold_weight * gold_return

        portfolio_returns.append(
            {
                "Date": next_date,
                "Return": total_return,
                "Mode": current_mode # Log mode for analysis
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
# Step 1: Get NIFTY 200 symbols
# -----------------------------
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
# Step 2: Download price data (Equity + Gold)
# -----------------------------
end_date = datetime.today()
start_date = end_date - timedelta(days=3000)

print("Downloading Equity Data...")
data = yf.download(
    tickers,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
    threads=True
)

print("Downloading Gold Data (GOLDBEES.NS)...")
gold_data = yf.download(
    "GOLDBEES.NS",
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
    threads=True
)

print("Downloading Nifty 50 Data (^NSEI)...")
nifty_data = yf.download(
    "^NSEI",
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
gold_prices = gold_data["Close"]
nifty_prices = nifty_data["Close"]

# Ensure Series (handle yfinance MultiIndex case)
if isinstance(gold_prices, pd.DataFrame):
    gold_prices = gold_prices.iloc[:, 0]
if isinstance(nifty_prices, pd.DataFrame):
    nifty_prices = nifty_prices.iloc[:, 0]

# -----------------------------
# Run backtest
# -----------------------------
# Note: Gold series is passed to backtest
# Default weights
equity_weight = 1.0
gold_weight = 0.0
enable_shuffle = True
shuffle_months = 2

returns_df = backtest(
    prices,
    gold_prices,
    nifty_prices,
    lookback=30,
    top_n=25,
    equity_weight=equity_weight,
    gold_weight=gold_weight,
    enable_shuffle=enable_shuffle,
    shuffle_months=shuffle_months
)

print("\nMonthly Returns:")
print(returns_df)

# -----------------------------
# Performance
# -----------------------------
stats = performance_stats(returns_df["Return"])
print("\nPerformance Stats (75% Equity / 25% Gold):")
print(stats)

print("\n12-Month Rolling Returns:")
rolling_12m = calculate_rolling_returns(returns_df["Return"], window=12)
print(rolling_12m)


# -----------------------------
# Nifty Benchmark Calculation
# -----------------------------
# Calculate on full Nifty data to preserve history and accuracy
# 1. Resample full daily series to monthly
nifty_month_ends = nifty_prices.resample("ME").last()
# 2. Monthly returns
nifty_returns = nifty_month_ends.pct_change().dropna()
# 3. Rolling returns (24m) on full history
nifty_rolling_24m_full = calculate_rolling_returns(nifty_returns, window=24)

# 4. Align to portfolio dates for plotting
# We use reindex with nearest or just matching dates if they are month-ends
# Since returns_df index are month ends, we expect exact matches usually.
# Using reindex ensures we align exactly to the plot's X-axis
nifty_rolling_24m = nifty_rolling_24m_full.reindex(returns_df.index, method='nearest', tolerance=timedelta(days=5))

# 5. Nifty Growth Curve (Cumulative)
# Align returns strictly for cumulative calc
nifty_returns_aligned = nifty_returns.reindex(returns_df.index, method='nearest', tolerance=timedelta(days=5)).fillna(0)
nifty_curve = 100 * (1 + nifty_returns_aligned).cumprod()


from plotly.subplots import make_subplots

# -----------------------------
# Plotting (Interactive)
# -----------------------------
equity_curve = 100 * (1 + returns_df["Return"]).cumprod()
rolling_24m = calculate_rolling_returns(returns_df["Return"], window=24)

# Create subplots: 2 rows, 1 column
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Portfolio Growth", "2-Year Rolling Returns"),
    row_heights=[0.7, 0.3]
)

# Trace 1: Portfolio Growth
fig.add_trace(go.Scatter(
    x=equity_curve.index,
    y=equity_curve.values,
    mode='lines',
    name='Portfolio Value',
    hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: $%{y:.2f}<extra></extra>'
), row=1, col=1)

# Trace 1b: Nifty 50 Growth
fig.add_trace(go.Scatter(
    x=nifty_curve.index,
    y=nifty_curve.values,
    mode='lines',
    name='Nifty 50 Value',
    line=dict(color='gray', dash='dot'),
    hovertemplate='<b>Date</b>: %{x}<br><b>Nifty Value</b>: $%{y:.2f}<extra></extra>'
), row=1, col=1)

# Trace 2: Rolling Returns (Portfolio)
fig.add_trace(go.Scatter(
    x=rolling_24m.index,
    y=rolling_24m.values * 100, # Convert to %
    mode='lines',
    name='Portfolio 2Y Rolling (%)',
    line=dict(color='orange'),
    hovertemplate='<b>Date</b>: %{x}<br><b>Port 2Y</b>: %{y:.2f}%<extra></extra>'
), row=2, col=1)

# Trace 3: Rolling Returns (Nifty 50)
fig.add_trace(go.Scatter(
    x=nifty_rolling_24m.index,
    y=nifty_rolling_24m.values * 100, # Convert to %
    mode='lines',
    name='Nifty 50 2Y Rolling (%)',
    line=dict(color='gray', dash='dot'),
    hovertemplate='<b>Date</b>: %{x}<br><b>Nifty 2Y</b>: %{y:.2f}%<extra></extra>'
), row=2, col=1)

fig.update_layout(
    title=f"Portfolio Analysis ({int(equity_weight * 100)}% Equity / {int(gold_weight * 100)}% Gold, Initial: $100)",
    template="plotly_dark",
    height=800,
    showlegend=True # Enabled legend to distinguish portfolio vs nifty
)

# Update y-axis labels
fig.update_yaxes(title_text="Value ($)", row=1, col=1)
fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

fig.write_html("portfolio_gold_growth.html")
print("\nInteractive plot saved to portfolio_gold_growth.html")
