import yfinance as yf
import pandas as pd

def calculate_rolling_returns(returns, window=3):
    rolling = (1 + returns).rolling(window=window).apply(lambda x: x.prod(), raw=True) - 1
    return rolling.dropna()

print("Downloading Nifty 50...")
data = yf.download("^NSEI", start="2016-01-01", end="2021-01-01", auto_adjust=True, progress=False)
prices = data["Close"]

# Ensure Series
if isinstance(prices, pd.DataFrame):
     prices = prices.iloc[:, 0]

print("\n--- Check Specific Dates ---")
try:
    p_2018 = prices.loc["2018-04-30"]
    p_2020 = prices.loc["2020-04-30"]
    actual_return = (p_2020 / p_2018) - 1
    print(f"Price 2018-04-30: {p_2018:.2f}")
    print(f"Price 2020-04-30: {p_2020:.2f}")
    print(f"Actual 24m Return (Discrete): {actual_return:.2%}")
except KeyError as e:
    print(f"Date missing in daily data: {e}")
    # Try finding nearest
    print("Nearest dates:")
    print(prices.index[prices.index.get_indexer([pd.Timestamp("2018-04-30"), pd.Timestamp("2020-04-30")], method="nearest")])

print("\n--- Monthly Resampling ---")
# Standard Monthly
monthly_prices = prices.resample("ME").last()
print(f"Monthly Price 2018-04-30: {monthly_prices['2018-04-30']:.2f}")
print(f"Monthly Price 2020-04-30: {monthly_prices['2020-04-30']:.2f}")

monthly_returns = monthly_prices.pct_change().dropna()

print("\n--- Rolling Calculation (Correct Way) ---")
# Rolling on full continuous monthly series
rolling_correct = calculate_rolling_returns(monthly_returns, window=24)
try:
    print(f"Rolling 24m on 2020-04-30 (Correct): {rolling_correct.loc['2020-04-30']:.2%}")
except KeyError:
    print("2020-04-30 not found in rolling index")

print("\n--- Rolling Calculation (Flawed Way - Simulating Slicing) ---")
# Simulate if we missed some rows or sliced weirdly
# Let's say we only took even months or something, effectively shrinking the window
sliced_returns = monthly_returns.iloc[::2] # Example of skipping
rolling_sliced = calculate_rolling_returns(sliced_returns, window=24)
# This assumes 24 *steps* (which is now 48 months), checking effect...
# Actually the issue in my code was "loc[common_idx]". If common_idx is full, it should be fine.
# But let's check what my code does.
