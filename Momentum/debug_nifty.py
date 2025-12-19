import yfinance as yf

print("Testing Nifty 50 download (^NSEI)...")
data = yf.download("^NSEI", period="1y", progress=False)

print(f"Shape: {data.shape}")
print(data.head())
print(data.tail())

if data.empty:
    print("Data is empty!")
else:
    print("Data seems okay.")
