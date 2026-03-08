# Returns Projection Tool

## Overview
The Returns Projection Tool is a comprehensive financial calculator that helps to project the future price of a stock based on current valuation metrics. It calculates future EPS, expected share price, Compounded Annual Growth Rate (CAGR), and the required entry/buy price to achieve a desired annualized return. 

It provides an interactive, modern web interface to make adjusting assumptions easy and visualizes the start-to-end price growth linearly.

## Features
- **Calculate Future Share Price:** Based on EPS growth and target PE multiple.
- **CAGR Calculation:** Computes the projected annualized growth rate from the current price.
- **Target Buy Price:** Calculates the maximum initial price to pay to achieve your specifically desired return.
- **Visual Charting:** Displays a 5-year straight-line projection using Chart.js.

## Dependencies
This application requires **Python 3** and utilizes the Flask web framework for the backend server.
The required Python package is:
- `Flask` (version 3.x)

## Installation
1. Ensure you have Python installed. You can check this by running `python3 --version` in your terminal.
2. Navigate to the project directory:
   ```bash
   cd /path/to/Valuation
   ```
3. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## How to Run
1. Start the Flask development server:
   ```bash
   python3 main.py
   ```
2. Open your web browser (e.g., Google Chrome, Safari).
3. Navigate to the local server URL:
   **http://localhost:5001**
   
   *(Note: The app is configured to run on port 5001 to avoid common macOS AirPlay port conflicts on port 5000).*

## Usage
Simply enter your valuation assumptions into the left panel of the web interface:
1. Current Share Price
2. EPS (Trailing Twelve Months)
3. Estimated EPS Growth Rate (%)
4. Target PE Multiple
5. Desired Investment Timeframe (Years)
6. Desired Annual Return (%)

Click **Calculate Projection** to instantly view your projected returns and visualize the chart!
