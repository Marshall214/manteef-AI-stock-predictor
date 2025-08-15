from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import logging
import psutil
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache configuration
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

# Rate limiter configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://*.netlify.app",
            "https://legendary-kangaroo-ac2fd5.netlify.app",
            "https://manteef-stock-predictor.netlify.app"  # Add your Netlify domain
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Model load
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_stock_model.json")
feature_names = [
    'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
    'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
]

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {os.path.abspath(MODEL_PATH)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0  # Default neutral RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_technical_indicators(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Fetch stock data and calculate technical indicators"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days * 2)  # Request twice the data to ensure enough history
        stock = yf.Ticker(ticker.upper())
        
        # Verify ticker exists and get info
        try:
            info = stock.info
            if not info or 'regularMarketPrice' not in info:
                suggested_symbols = "'AAPL' (Apple), 'MSFT' (Microsoft), 'GOOGL' (Google), 'MIVS' (OTC stock)"
                return {
                    'error': f"Invalid ticker symbol: {ticker}. Please ensure it's a valid stock symbol. Examples: {suggested_symbols}",
                    'success': False
                }
        except Exception as e:
            exchange = getattr(stock, 'exchange', 'Unknown')
            return {
                'error': f"Could not fetch data for {ticker}. This might be because: 1) The stock is not listed, 2) It's on a different exchange ({exchange}), or 3) The symbol is incorrect.",
                'success': False
            }
            
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return {
                'error': f"No trading data available for {ticker}. This might be a newly listed stock or it might be delisted.",
                'success': False
            }
        if len(df) < 30:
            available_days = len(df)
            return {
                'error': f"Insufficient historical data for {ticker}. Found {available_days} days of data, but need at least 30 days. This might be a newly listed stock.",
                'success': False
            }
        
        close_prices = df['Close']
        volumes = df['Volume']
        
        # Calculate indicators
        pct_change = (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2] if len(close_prices) > 1 else 0.0
        ma_7 = close_prices.rolling(window=7).mean().iloc[-1] if len(close_prices) >= 7 else close_prices.iloc[-1]
        ma_21 = close_prices.rolling(window=21).mean().iloc[-1] if len(close_prices) >= 21 else close_prices.iloc[-1]
        returns = close_prices.pct_change().dropna()
        volatility_7 = returns.rolling(window=7).std().iloc[-1] if len(returns) >= 7 else returns.std()
        volatility_7 = volatility_7 if not pd.isna(volatility_7) else 0.01
        volume = float(volumes.iloc[-1])
        RSI_14 = calculate_rsi(close_prices, 14)
        momentum_7 = (close_prices.iloc[-1] - close_prices.iloc[-8]) / close_prices.iloc[-8] if len(close_prices) > 7 else 0.0
        momentum_21 = (close_prices.iloc[-1] - close_prices.iloc[-22]) / close_prices.iloc[-22] if len(close_prices) > 21 else 0.0
        ma_diff = ma_7 - ma_21
        avg_volume_20 = volumes.rolling(window=20).mean().iloc[-1] if len(volumes) >= 20 else volumes.mean()
        vol_ratio_20 = volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        indicators = {
            'pct_change': float(pct_change),
            'ma_7': float(ma_7),
            'ma_21': float(ma_21),
            'volatility_7': float(volatility_7),
            'volume': float(volume),
            'RSI_14': float(RSI_14),
            'momentum_7': float(momentum_7),
            'momentum_21': float(momentum_21),
            'ma_diff': float(ma_diff),
            'vol_ratio_20': float(vol_ratio_20)
        }
        
        metadata = {
            'ticker': ticker.upper(),
            'current_price': float(close_prices.iloc[-1]),
            'previous_close': float(close_prices.iloc[-2]) if len(close_prices) > 1 else float(close_prices.iloc[-1]),
            'data_points': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            }
        }
        
        return {
            'indicators': indicators,
            'metadata': metadata,
            'success': True
        }
    except Exception as e:
        print(f"Error calculating indicators for {ticker}: {e}")
        return {'error': f"Failed to fetch data for {ticker}: {str(e)}", 'success': False}

def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or len(ticker.strip()) == 0:
        return False
    ticker = ticker.strip().upper()
    return 1 <= len(ticker) <= 5 and ticker.isalpha()

def make_prediction(df: pd.DataFrame, input_data: dict) -> tuple:
    """Common prediction logic"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            prediction = 1 if probabilities[1] > 0.5 else 0
        else:
            prediction_prob = model.predict(df)[0]
            if isinstance(prediction_prob, float):
                probabilities = [1 - prediction_prob, prediction_prob]
                prediction = 1 if prediction_prob > 0.5 else 0
            else:
                prediction = int(prediction_prob)
                probabilities = [1 - prediction, prediction]
                
        confidence = float(max(probabilities))
        strength = "STRONG" if confidence >= 0.7 else "MODERATE" if confidence >= 0.6 else "WEAK"
        signal = 'BUY' if prediction == 1 else 'SELL'
        
        return {
            'prediction': int(prediction),
            'signal': signal,
            'confidence': round(confidence, 3),
            'signal_strength': strength,
            'probabilities': {
                'down': round(float(probabilities[0]), 3),
                'up': round(float(probabilities[1]), 3)
            },
            'timestamp': datetime.now().isoformat(),
            'recommendation': f"{strength}_{signal}"
        }, 200
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'error': f'Model prediction failed: {str(e)}'}, 500

# Endpoints
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'environment': os.getenv('FLASK_ENV', 'development'),
        'features': ['manual_prediction', 'ticker_prediction', 'ticker_info']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction with manual feature input"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
        df = pd.DataFrame([data])[feature_names]
        if df.empty or df.isnull().any().any():
            return jsonify({'error': 'Empty or missing values in input data'}), 400
        response, status = make_prediction(df, data)
        return jsonify(response), status
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-ticker', methods=['POST'])
def predict_ticker():
    """Prediction using ticker symbol"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
        ticker = data['ticker'].strip().upper()
        if not validate_ticker(ticker):
            return jsonify({'error': 'Invalid ticker format. Use 1-5 letter stock symbols (e.g., AAPL, TSLA)'}), 400
        result = calculate_technical_indicators(ticker)
        if not result['success']:
            return jsonify({'error': result['error']}), 400
        df = pd.DataFrame([result['indicators']])[feature_names]
        response, status = make_prediction(df, result['indicators'])
        if status == 200:
            response.update({
                'ticker_info': result['metadata'],
                'technical_indicators': result['indicators']
            })
        return jsonify(response), status
    except Exception as e:
        print(f"Ticker prediction error: {e}")
        return jsonify({'error': f'Ticker prediction failed: {str(e)}'}), 500

@app.route('/ticker-info', methods=['POST'])
def get_ticker_info():
    """Get ticker information and technical indicators"""
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
        ticker = data['ticker'].strip().upper()
        if not validate_ticker(ticker):
            return jsonify({'error': 'Invalid ticker format'}), 400
        result = calculate_technical_indicators(ticker)
        if not result['success']:
            return jsonify({'error': result['error']}), 400
        return jsonify({
            'ticker_info': result['metadata'],
            'technical_indicators': result['indicators'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Ticker info error: {e}")
        return jsonify({'error': f'Failed to get ticker info: {str(e)}'}), 500

@app.route('/features', methods=['GET'])
def get_expected_features():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        return jsonify({
            'features': feature_names,
            'feature_count': len(feature_names),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get features: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        model_type = type(model).__name__
        info = {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'environment': os.getenv('FLASK_ENV', 'development'),
            'feature_count': len(feature_names),
            'features': feature_names,
            'key_parameters': {}
        }
        if hasattr(model, 'get_params'):
            params = model.get_params()
            info['key_parameters'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate'),
                'objective': params.get('objective'),
                'subsample': params.get('subsample')
            }
        elif hasattr(model, 'get_xgb_params'):
            params = model.get_xgb_params()
            info['key_parameters'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('eta', params.get('learning_rate')),
                'objective': params.get('objective'),
                'subsample': params.get('subsample')
            }
        elif hasattr(model, 'save_config'):
            info['key_parameters'] = {'note': 'XGBoost 2.0+ model - parameters embedded in model'}
        else:
            print(f"Model type {model_type} - no parameter extraction method found")
            info['key_parameters'] = {'note': f'Parameters not accessible for {model_type}'}
        return jsonify(info)
    except Exception as e:
        print(f"Model info error: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

# Run app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"
    print("🚀 Starting Enhanced Manteef Stock Predictor API...")
    print("📡 Available endpoints:")
    print("   GET  /           - Health check")
    print("   POST /predict    - Make prediction (manual input)")
    print("   POST /predict-ticker - Make prediction from ticker")
    print("   POST /ticker-info - Get ticker technical indicators")
    print("   GET  /features   - Get expected features")
    print("   GET  /model-info - Get model information")
    print(f"🌐 Running on port {port}")
    app.run(debug=debug, host="0.0.0.0", port=port)