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
import requests
from typing import Dict, Optional
import logging
import psutil
import pytz
from functools import wraps
import time

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

# CORS configuration - Enhanced
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

# Model and feature configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_stock_model.json")
feature_names = [
    'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
    'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
]

# Popular tickers for suggestions
POPULAR_TICKERS = {
    'A': ['AAPL', 'AMZN', 'AMD', 'ADBE'],
    'M': ['MSFT', 'META', 'MIVS'],
    'G': ['GOOGL', 'GOOG'],
    'T': ['TSLA', 'TMUS'],
    'N': ['NVDA', 'NFLX', 'NKE'],
    'B': ['BABA', 'BAC'],
    'C': ['CRM', 'COST'],
    'D': ['DIS', 'DDOG'],
    'E': ['EBAY', 'ETSY'],
    'F': ['FB', 'FANG'],
    'H': ['HD', 'HOOD'],
    'I': ['IBM', 'INTC'],
    'J': ['JNJ', 'JPM'],
    'K': ['KO', 'KLAC'],
    'L': ['LOW', 'LYFT'],
    'O': ['ORCL', 'OKTA'],
    'P': ['PYPL', 'PFE'],
    'Q': ['QCOM', 'QQQ'],
    'R': ['ROKU', 'RBLX'],
    'S': ['SHOP', 'SNAP', 'SQ'],
    'U': ['UBER', 'UPST'],
    'V': ['V', 'VZ'],
    'W': ['WMT', 'WORK'],
    'X': ['XOM', 'XLNX'],
    'Y': ['YUM', 'YELP'],
    'Z': ['ZM', 'ZNGA']
}

# Load model
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded from {os.path.abspath(MODEL_PATH)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class FinnhubProvider:
    """Finnhub API data provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({'X-Finnhub-Token': api_key})
    
    def fetch_stock_data(self, ticker: str, days: int = 90) -> dict:
        """Fetch OHLCV data from Finnhub"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Convert to Unix timestamps
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        url = f"{self.base_url}/stock/candle"
        params = {
            'symbol': ticker.upper(),
            'resolution': 'D',  # Daily data
            'from': start_ts,
            'to': end_ts
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('s') != 'ok':
            raise ValueError(f"No data available for {ticker}")
            
        return self.format_data(data, ticker)
    
    def format_data(self, raw_data: dict, ticker: str) -> dict:
        """Convert Finnhub format to pandas DataFrame"""
        df = pd.DataFrame({
            'Open': raw_data['o'],
            'High': raw_data['h'], 
            'Low': raw_data['l'],
            'Close': raw_data['c'],
            'Volume': raw_data['v']
        }, index=pd.to_datetime(raw_data['t'], unit='s'))
        
        return {
            'data': df,
            'ticker': ticker,
            'source': 'finnhub'
        }

def suggest_similar_tickers(ticker: str) -> list:
    """Suggest similar tickers based on first letter"""
    if not ticker:
        return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    first_letter = ticker[0].upper()
    suggestions = POPULAR_TICKERS.get(first_letter, ['AAPL', 'MSFT', 'GOOGL'])
    return suggestions[:3]

def is_market_open():
    """Check if US market is currently open"""
    try:
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Basic market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
        if now_et.weekday() > 4:  # Weekend
            return False
        
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    except Exception:
        return False

def add_market_context(result: dict) -> dict:
    """Add market context to API responses"""
    if result.get('success') and 'metadata' in result:
        try:
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            
            result['metadata']['market_status'] = {
                'is_open': is_market_open(),
                'is_weekend': now_et.weekday() > 4,
                'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
                'data_delay_notice': 'Market data may be delayed up to 15 minutes'
            }
        except Exception as e:
            logger.warning(f"Failed to add market context: {e}")
    return result

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

def calculate_technical_indicators_finnhub(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Calculate indicators using Finnhub data"""
    try:
        # Initialize Finnhub provider
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            raise ValueError("Finnhub API key not configured")
            
        provider = FinnhubProvider(api_key)
        
        # Fetch data
        result = provider.fetch_stock_data(ticker, period_days)
        df = result['data']
        
        if df.empty or len(df) < 30:
            raise ValueError(f"Insufficient data for {ticker}")
            
        # Calculate all indicators (same logic as before)
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
        
        return {
            'indicators': indicators,
            'metadata': {
                'ticker': ticker.upper(),
                'current_price': float(close_prices.iloc[-1]),
                'previous_close': float(close_prices.iloc[-2]) if len(close_prices) > 1 else float(close_prices.iloc[-1]),
                'data_source': 'finnhub',
                'data_points': len(df),
                'date_range': {
                    'start': df.index[0].strftime('%Y-%m-%d'),
                    'end': df.index[-1].strftime('%Y-%m-%d')
                }
            },
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Finnhub error for {ticker}: {e}")
        raise e

def calculate_technical_indicators_yfinance_fallback(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Fallback yfinance implementation"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days * 2)
        stock = yf.Ticker(ticker.upper())
        
        # Verify ticker exists and get info
        try:
            info = stock.info
            if not info or 'regularMarketPrice' not in info:
                suggestions = suggest_similar_tickers(ticker)
                return {
                    'error': f"Invalid ticker symbol: {ticker}",
                    'suggestions': suggestions,
                    'help': f"Try: {', '.join(suggestions)}",
                    'success': False
                }
        except Exception as e:
            suggestions = suggest_similar_tickers(ticker)
            return {
                'error': f"Could not fetch data for {ticker}. Please verify the ticker symbol.",
                'suggestions': suggestions,
                'help': f"Popular symbols: {', '.join(suggestions)}",
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
                'error': f"Not enough trading history for {ticker}. Found {available_days} days, need at least 30. This might be a newly listed stock.",
                'help': "Try a different stock with longer trading history",
                'success': False
            }
        
        close_prices = df['Close']
        volumes = df['Volume']
        
        # Calculate indicators (same as Finnhub)
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
            'data_source': 'yfinance_fallback',
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
        logger.error(f"YFinance fallback error for {ticker}: {e}")
        return {'error': f"Failed to fetch data for {ticker}: {str(e)}", 'success': False}

@cache.memoize(timeout=300)
def calculate_technical_indicators_with_fallback(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Main function with Finnhub primary + yfinance fallback"""
    
    # Try Finnhub first
    try:
        result = calculate_technical_indicators_finnhub(ticker, period_days)
        if result and result.get('success'):
            return add_market_context(result)
    except Exception as finnhub_error:
        logger.warning(f"Finnhub failed for {ticker}: {finnhub_error}")
        
        # Fallback to yfinance
        try:
            logger.info(f"Attempting yfinance fallback for {ticker}")
            result = calculate_technical_indicators_yfinance_fallback(ticker, period_days)
            if result and result.get('success'):
                result['metadata']['fallback_reason'] = str(finnhub_error)
                return add_market_context(result)
            else:
                return result  # Return the error from yfinance
        except Exception as yf_error:
            logger.error(f"Both providers failed for {ticker}: Finnhub({finnhub_error}), YFinance({yf_error})")
            return {
                'error': f"Unable to fetch data for {ticker}. Both primary and backup data sources failed.",
                'details': {
                    'finnhub_error': str(finnhub_error),
                    'yfinance_error': str(yf_error)
                },
                'suggestions': suggest_similar_tickers(ticker),
                'success': False
            }

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
        logger.error(f"Prediction error: {e}")
        return {'error': f'Model prediction failed: {str(e)}'}, 500

# Request timing middleware
@app.before_request
def log_request_info():
    request.start_time = time.time()

@app.after_request
def log_request_response(response):
    duration = time.time() - request.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}s")
    
    # FIXED: Only try to access request.json for appropriate requests
    if request.method == 'POST' and request.content_type and 'application/json' in request.content_type:
        try:
            if hasattr(request, 'json') and request.json:
                ticker = request.json.get('ticker', 'unknown')
                logger.info(f"Ticker request: {ticker} - Status: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not log ticker info: {e}")
    
    return response

# ADDED: Explicit OPTIONS handler for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept")
        response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
        return response

# API Endpoints
@app.route('/', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0',
        'environment': os.getenv('FLASK_ENV', 'development'),
        'features': ['manual_prediction', 'ticker_prediction', 'ticker_info', 'finnhub_integration'],
        'data_sources': ['finnhub_primary', 'yfinance_fallback'],
        'market_open': is_market_open(),
        'cors_enabled': True
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
@limiter.limit("30 per minute")
def predict():
    """Prediction with manual feature input"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
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
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-ticker', methods=['POST', 'OPTIONS'])
@limiter.limit("20 per minute")
def predict_ticker():
    """Enhanced ticker prediction with Finnhub primary"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
            
        ticker = data['ticker'].strip().upper()
        if not validate_ticker(ticker):
            suggestions = suggest_similar_tickers(ticker)
            return jsonify({
                'error': 'Invalid ticker format. Use 1-5 letter stock symbols (e.g., AAPL, TSLA)',
                'suggestions': suggestions,
                'help': f"Try: {', '.join(suggestions)}"
            }), 400
            
        # Use new function with fallback
        result = calculate_technical_indicators_with_fallback(ticker)
        
        if not result.get('success'):
            return jsonify({
                'error': result.get('error', 'Unknown error'),
                'suggestions': result.get('suggestions', []),
                'help': result.get('help', 'Please try a different ticker')
            }), 400
            
        # Rest of existing prediction logic
        df = pd.DataFrame([result['indicators']])[feature_names] 
        response, status = make_prediction(df, result['indicators'])
        
        if status == 200:
            response.update({
                'ticker_info': result['metadata'],
                'technical_indicators': result['indicators']
            })
            
        return jsonify(response), status
        
    except Exception as e:
        logger.error(f"Ticker prediction error: {e}")
        return jsonify({'error': f'Ticker prediction failed: {str(e)}'}), 500

@app.route('/ticker-info', methods=['POST', 'OPTIONS'])
@limiter.limit("30 per minute")
def get_ticker_info():
    """Get ticker information and technical indicators"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
        ticker = data['ticker'].strip().upper()
        if not validate_ticker(ticker):
            suggestions = suggest_similar_tickers(ticker)
            return jsonify({
                'error': 'Invalid ticker format',
                'suggestions': suggestions
            }), 400
        result = calculate_technical_indicators_with_fallback(ticker)
        if not result.get('success'):
            return jsonify({
                'error': result.get('error', 'Unknown error'),
                'suggestions': result.get('suggestions', [])
            }), 400
        return jsonify({
            'ticker_info': result['metadata'],
            'technical_indicators': result['indicators'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Ticker info error: {e}")
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
            'data_sources': {
                'primary': 'Finnhub API',
                'fallback': 'Yahoo Finance',
                'finnhub_configured': bool(os.getenv('FINNHUB_API_KEY'))
            },
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
        else:
            info['key_parameters'] = {'note': 'XGBoost model - parameters embedded'}
        return jsonify(info)
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

# Run app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"
    logger.info("🚀 Starting Enhanced Manteef Stock Predictor API v3.0...")
    logger.info("📡 Available endpoints:")
    logger.info("   GET  /           - Health check")
    logger.info("   POST /predict    - Make prediction (manual input)")
    logger.info("   POST /predict-ticker - Make prediction from ticker")
    logger.info("   POST /ticker-info - Get ticker technical indicators")
    logger.info("   GET  /features   - Get expected features")
    logger.info("   GET  /model-info - Get model information")
    logger.info(f" Finnhub API configured: {bool(os.getenv('FINNHUB_API_KEY'))}")
    logger.info(f"Running on port {port}")
    logger.info("✅ CORS enabled for all origins")
    logger.info("✅ OPTIONS requests handled")
    app.run(debug=debug, host="0.0.0.0", port=port)