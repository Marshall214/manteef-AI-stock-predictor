# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from datetime import datetime
import logging

app = Flask(__name__)

# --- Enhanced CORS configuration ---
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://*.netlify.app",
            "https://legendary-kangaroo-ac2fd5.netlify.app",
            "https://manteef-stock-predictor.netlify.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# --- Paths ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_stock_model.json")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "all_stocks_5yr.csv")

# --- Load XGBoost model ---
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Load CSV dataset ---
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['Name', 'date']).reset_index(drop=True)

# --- Feature list ---
feature_cols = ['pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
                'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20']

# --- Helper Functions ---
def compute_technical_indicators(stock_df):
    stock_df = stock_df.copy()
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    stock_df[num_cols] = stock_df[num_cols].ffill().bfill()

    stock_df['pct_change'] = stock_df['close'].pct_change().fillna(0)
    stock_df['ma_7'] = stock_df['close'].rolling(7, min_periods=1).mean()
    stock_df['ma_21'] = stock_df['close'].rolling(21, min_periods=1).mean()
    stock_df['volatility_7'] = stock_df['close'].rolling(7, min_periods=1).std().fillna(0)

    # RSI
    delta = stock_df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    stock_df['RSI_14'] = 100 - (100 / (1 + rs))

    stock_df['momentum_7'] = stock_df['close'] - stock_df['ma_7']
    stock_df['momentum_21'] = stock_df['close'] - stock_df['ma_21']
    stock_df['ma_diff'] = stock_df['ma_7'] - stock_df['ma_21']
    stock_df['vol_ratio_20'] = stock_df['volume'] / stock_df['volume'].rolling(20, min_periods=1).mean()

    return stock_df.fillna(0)

# --- Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/tickers', methods=['GET'])
def ticker_search():
    query = request.args.get('q', '').upper()
    matched = df['Name'].unique()
    if query:
        matched = [t for t in matched if query in t]
    return jsonify({'tickers': matched[:20]})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    ticker = data.get('ticker', '').upper()
    if not ticker:
        return jsonify({'error': 'No ticker provided'}), 400

    stock_df = df[df['Name'] == ticker].copy()
    if stock_df.empty:
        return jsonify({'error': f'Ticker {ticker} not found'}), 404

    # Compute indicators
    stock_df = compute_technical_indicators(stock_df)
    latest_features_dict = stock_df.iloc[-1][feature_cols].to_dict()
    latest_features = np.array([[latest_features_dict[f] for f in feature_cols]])

    # Prediction
    pred_proba = model.predict_proba(latest_features)[0]
    pred_class = int(pred_proba[1] > 0.5)
    confidence = float(max(pred_proba))
    signal = 'BUY' if pred_class == 1 else 'SELL'
    strength = 'STRONG' if confidence >= 0.7 else ('MODERATE' if confidence >= 0.6 else 'WEAK')

    # Get mini price trend (last 30 days)
    mini_trend = stock_df.tail(30).sort_values('date')
    trend_prices = mini_trend['close'].tolist()
    trend_dates = mini_trend['date'].dt.strftime('%Y-%m-%d').tolist()

    # Logging
    logging.info(f"Ticker: {ticker}, Signal: {signal}, Confidence: {confidence:.3f}")

    return jsonify({
        'prediction': pred_class,
        'signal': signal,
        'signal_strength': strength,
        'confidence': round(confidence, 3),
        'features': latest_features_dict,
        'date_range': {
            'start': stock_df['date'].min().strftime('%Y-%m-%d'),
            'end': stock_df['date'].max().strftime('%Y-%m-%d')
        },
        'mini_trend': {
            'prices': trend_prices,
            'dates': trend_dates
        },
        'probabilities': {
            'down': round(float(pred_proba[0]), 3),
            'up': round(float(pred_proba[1]), 3)
        },
        'model_used': type(model).__name__,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features': feature_cols, 'feature_count': len(feature_cols)})

# --- Run App ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"
    logging.info("🚀 Starting CSV-Based Stock Predictor API v1.0...")
    logging.info(f"🌐 Running on port {port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
