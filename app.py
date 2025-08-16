from flask import Flask, request, jsonify, make_response
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

# CORS configuration - FIXED: Use ONLY Flask-CORS, no manual headers
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

def get_stock_data(ticker: str, period_days: int = 90) -> dict:
    """
    Get stock data using yfinance
    """
    try:
        # Clean ticker symbol
        ticker = ticker.upper().strip()
        
        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days * 2)  # Get extra data to ensure we have enough
        
        # Get historical data
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Get company info with error handling
        info = {}
        try:
            stock_info = stock.info
            info = {
                'ticker': ticker,
                'current_price': stock_info.get('currentPrice', hist_data['Close'].iloc[-1]),
                'previous_close': stock_info.get('previousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else hist_data['Close'].iloc[-1]),
                'market_cap': stock_info.get('marketCap', 'N/A'),
                'company_name': stock_info.get('longName', stock_info.get('shortName', ticker)),
                'sector': stock_info.get('sector', 'N/A'),
                'industry': stock_info.get('industry', 'N/A'),
                'description': stock_info.get('longBusinessSummary', f"Stock information for {ticker}")
            }
        except Exception as e:
            logger.warning(f"Could not fetch detailed company info for {ticker}: {e}")
            # Fallback to basic info from historical data
            info = {
                'ticker': ticker,
                'current_price': float(hist_data['Close'].iloc[-1]),
                'previous_close': float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else float(hist_data['Close'].iloc[-1]),
                'market_cap': 'N/A',
                'company_name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'description': f"Stock information for {ticker}"
            }
        
        return {
            'success': True,
            'data': hist_data,
            'info': info,
            'source': 'yfinance'
        }
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return {
            'success': False,
            'error': f"Could not fetch data for {ticker}. Please verify the ticker symbol.",
            'source': 'yfinance'
        }

@cache.memoize(timeout=300)
def calculate_technical_indicators(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Calculate technical indicators using yfinance data"""
    try:
        # Get stock data
        result = get_stock_data(ticker, period_days)
        
        if not result['success']:
            suggestions = suggest_similar_tickers(ticker)
            return {
                'error': result['error'],
                'suggestions': suggestions,
                'help': f"Try: {', '.join(suggestions)}",
                'success': False
            }
        
        df = result['data']
        
        # Validate data
        if df.empty or len(df) < 30:
            available_days = len(df)
            return {
                'error': f"Not enough trading history for {ticker}. Found {available_days} days, need at least 30. This might be a newly listed stock.",
                'help': "Try a different stock with longer trading history",
                'success': False
            }
        
        # Calculate technical indicators
        close_prices = df['Close']
        volumes = df['Volume']
        
        # Basic calculations
        pct_change = (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2] if len(close_prices) > 1 else 0.0
        ma_7 = close_prices.rolling(window=7).mean().iloc[-1] if len(close_prices) >= 7 else close_prices.iloc[-1]
        ma_21 = close_prices.rolling(window=21).mean().iloc[-1] if len(close_prices) >= 21 else close_prices.iloc[-1]
        
        # Volatility calculation
        returns = close_prices.pct_change().dropna()
        volatility_7 = returns.rolling(window=7).std().iloc[-1] if len(returns) >= 7 else returns.std()
        volatility_7 = volatility_7 if not pd.isna(volatility_7) else 0.01
        
        # Volume and other indicators
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
            'current_price': float(result['info']['current_price']),
            'previous_close': float(result['info']['previous_close']),
            'data_source': 'yfinance',
            'data_points': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            },
            'company_info': result['info']
        }
        
        final_result = {
            'indicators': indicators,
            'metadata': metadata,
            'success': True
        }
        
        return add_market_context(final_result)
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {e}")
        suggestions = suggest_similar_tickers(ticker)
        return {
            'error': f"Failed to fetch data for {ticker}: {str(e)}",
            'suggestions': suggestions,
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

# FIXED: Remove manual CORS headers from @app.after_request
@app.after_request
def log_request_response(response):
    duration = time.time() - request.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}s")
    
    # Only try to access request.json for appropriate requests
    if request.method == 'POST' and request.content_type and 'application/json' in request.content_type:
        try:
            if hasattr(request, 'json') and request.json:
                ticker = request.json.get('ticker', 'unknown')
                logger.info(f"Ticker request: {ticker} - Status: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not log ticker info: {e}")
    
    return response

# REMOVED: Manual preflight handler - Flask-CORS handles this automatically

# API Endpoints
@app.route('/', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.1.0',
        'environment': os.getenv('FLASK_ENV', 'development'),
        'features': ['manual_prediction', 'ticker_prediction', 'ticker_info', 'yfinance_integration'],
        'data_sources': ['yfinance_primary'],
        'market_open': is_market_open(),
        'cors_enabled': True
    })

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
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
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-ticker', methods=['POST'])
@limiter.limit("20 per minute")
def predict_ticker():
    """Ticker prediction using yfinance"""
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
            
        # Calculate technical indicators
        result = calculate_technical_indicators(ticker)
        
        if not result.get('success'):
            return jsonify({
                'error': result.get('error', 'Unknown error'),
                'suggestions': result.get('suggestions', []),
                'help': result.get('help', 'Please try a different ticker')
            }), 400
            
        # Make prediction using technical indicators
        df = pd.DataFrame([result['indicators']])[feature_names] 
        prediction_response, status = make_prediction(df, result['indicators'])
        
        if status == 200:
            prediction_response.update({
                'ticker_info': result['metadata'],
                'technical_indicators': result['indicators']
            })
        
        return jsonify(prediction_response), status
        
    except Exception as e:
        logger.error(f"Ticker prediction error: {e}")
        return jsonify({'error': f'Ticker prediction failed: {str(e)}'}), 500

@app.route('/ticker-info', methods=['POST'])
@limiter.limit("30 per minute")
def get_ticker_info():
    """Get ticker information and technical indicators"""
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
            
        result = calculate_technical_indicators(ticker)
        
        if not result.get('success'):
            return jsonify({
                'error': result.get('error', 'Unknown error'),
                'suggestions': result.get('suggestions', [])
            }), 400
        
        return jsonify({
            'ticker_info': result['metadata'],
            'technical_indicators': result['indicators'],
            'timestamp': datetime.now().isoformat()
        }), 200
        
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
                'primary': 'Yahoo Finance (yfinance)',
                'fallback': 'None (yfinance is primary and only source)',
                'yfinance_available': True
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
    logger.info("🚀 Starting Enhanced Manteef Stock Predictor API v3.1...")
    logger.info("📡 Available endpoints:")
    logger.info("   GET  /           - Health check")
    logger.info("   POST /predict    - Make prediction (manual input)")
    logger.info("   POST /predict-ticker - Make prediction from ticker")
    logger.info("   POST /ticker-info - Get ticker technical indicators")
    logger.info("   GET  /features   - Get expected features")
    logger.info("   GET  /model-info - Get model information")
    logger.info("📊 Data Source: Yahoo Finance (yfinance) - No API key required")
    logger.info(f"🌐 Running on port {port}")
    logger.info("✅ CORS enabled for all origins")
    logger.info("✅ OPTIONS requests handled")
    app.run(debug=debug, host="0.0.0.0", port=port)