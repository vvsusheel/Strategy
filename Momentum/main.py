import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# Step 1: Get NIFTY 500 symbols
# -----------------------------
import requests
import io

url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
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
start_date = end_date - timedelta(days=400)

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
# Step 3: Simple momentum calc
# Momentum = price today / price 6 months ago - 1
# -----------------------------
momentum = {}

for ticker in tickers:
    if ticker not in data.columns.levels[1]:
        continue

    close = data["Close"][ticker].dropna()

    if len(close) < 130:   # ~6 months of trading days
        continue

    momentum_score = (close.iloc[-1] / close.iloc[-130]) - 1
    momentum[ticker] = momentum_score

# -----------------------------
# Step 4: Show top 10 stocks
# -----------------------------
momentum_df = (
    pd.Series(momentum)
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Momentum Stocks:")
print(momentum_df)