from flask import Flask, render_template, request, jsonify
import yfinance as yf

app = Flask(__name__)

@app.route('/fetch_stock_data', methods=['GET'])
def fetch_stock_data():
    ticker_symbol = request.args.get('ticker')
    if not ticker_symbol:
        return jsonify({'success': False, 'error': 'No ticker provided'})
        
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # fallback to regularPrice or previousClose if currentPrice is missing
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        eps_ttm = info.get('trailingEps')
        
        if current_price is None or eps_ttm is None:
            return jsonify({'success': False, 'error': 'Could not accurately fetch current price and EPS for this ticker.'})
            
        return jsonify({
            'success': True,
            'current_price': round(float(current_price), 2),
            'eps': round(float(eps_ttm), 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to fetch data: {str(e)}'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    try:
        eps = float(data.get('eps', 0))
        current_price = float(data.get('current_price', 0))
        eps_growth_rate = float(data.get('eps_growth_rate', 0)) / 100.0
        time_period = float(data.get('time_period', 5))
        pe_multiple = float(data.get('pe_multiple', 0))
        desired_return = float(data.get('desired_return', 0)) / 100.0
        
        if time_period <= 0:
            return jsonify({'success': False, 'error': 'Time period must be greater than 0'})

        # Calculate Future EPS
        future_eps = eps * ((1 + eps_growth_rate) ** time_period)
        
        # Calculate Future Share Price
        future_price = future_eps * pe_multiple
        
        # Calculate CAGR Returns
        if current_price > 0:
            cagr = ((future_price / current_price) ** (1 / time_period)) - 1
        else:
            cagr = 0
            
        # Calculate Entry Price / Buy Price for Desired Return
        buy_price = future_price / ((1 + desired_return) ** time_period)
            
        return jsonify({
            'success': True,
            'future_eps': round(future_eps, 2),
            'future_price': round(future_price, 2),
            'cagr': round(cagr * 100, 2),
            'buy_price': round(buy_price, 2),
            'start_price': current_price,
            'end_price': round(future_price, 2),
            'time_period': int(time_period)
        })

    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid input data type. Please enter numbers.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
