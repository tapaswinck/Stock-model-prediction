from flask import Flask, request, jsonify
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

# Load pre-trained model and scaler
model = load_model('lstm_stock_model.h5')  # Ensure this file exists
scaler = MinMaxScaler(feature_range=(0, 1))  # Use the same scaler as during training

# Function to compute RSI
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use exponential weighted moving average
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle potential division by zero
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    rsi.fillna(50, inplace=True)  # Fill NaN values with a neutral RSI value

    return rsi

# Fetch stock data for real-time prediction
def fetch_real_time_data(ticker):
    # Option 1: Use a valid period
    stock_data = yf.download(ticker, period='1y', interval='1d')

    # Option 2: Use a date range (commented out)
    # from datetime import datetime, timedelta
    # end_date = datetime.today()
    # start_date = end_date - timedelta(days=200)
    # stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')

    return stock_data

# Preprocess input data for prediction
def preprocess_data(stock_data):
    stock_data['MA50'] = stock_data['Adj Close'].rolling(window=20).mean()
    stock_data['MA200'] = stock_data['Adj Close'].rolling(window=50).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Adj Close'], 14)

    # Fill any NaN values using backfill or forward fill
    stock_data.bfill(inplace=True)  # Use bfill() instead of fillna(method='bfill')

    feature_cols = ['Adj Close', 'MA50', 'MA200', 'RSI']
    data = stock_data[feature_cols].values

    # Scale data
    scaled_data = scaler.fit_transform(data)

    # Prepare the last 60 days of data
    last_60_days = scaled_data[-60:]
    X_input = np.expand_dims(last_60_days, axis=0)

    return X_input

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', None)

    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        # Fetch and preprocess stock data
        stock_data = fetch_real_time_data(ticker)
        if stock_data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404

        processed_data = preprocess_data(stock_data)

        # Make prediction
        predicted_price = model.predict(processed_data)

        # Inverse transform to get actual price
        dummy = np.zeros((predicted_price.shape[0], scaler.n_features_in_))
        dummy[:, 0] = predicted_price[:, 0]
        predicted_price_actual = scaler.inverse_transform(dummy)[:, 0]

        return jsonify({'predicted_price': float(predicted_price_actual[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
