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
import requests
from typing import Dict, Optional
import logging
import pytz
import time
import json
from urllib.parse import urlencode
import random

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
    'CACHE_DEFAULT_TIMEOUT': 600  # 10 minutes for stock data
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
            "https://manteef-stock-predictor.netlify.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')  # Set this in your environment
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', 'demo')

# Model and feature configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_stock_model.json")
feature_names = [
    'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
    'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
]

# Popular tickers for suggestions
POPULAR_TICKERS = {
    'A': ['AAPL', 'AMZN', 'AMD', 'ADBE'],
    'M': ['MSFT', 'META'],
    'G': ['GOOGL', 'GOOG'],
    'T': ['TSLA', 'TMUS'],
    'N': ['NVDA', 'NFLX', 'NKE'],
    'B': ['BABA', 'BAC'],
    'C': ['CRM', 'COST'],
    'D': ['DIS'],
    'I': ['IBM', 'INTC'],
    'J': ['JNJ', 'JPM'],
    'K': ['KO'],
    'P': ['PYPL', 'PFE'],
    'S': ['SHOP', 'SNAP'],
    'U': ['UBER'],
    'V': ['V', 'VZ'],
    'W': ['WMT'],
}

# Load model
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info(f"✅ Model loaded from {os.path.abspath(MODEL_PATH)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def create_robust_session():
    """Create a robust requests session"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

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
        
        if now_et.weekday() > 4:  # Weekend
            return False
        
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    except Exception:
        return False

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def fetch_alpha_vantage_data(ticker: str) -> dict:
    """Fetch data from Alpha Vantage API"""
    try:
        logger.info(f"🔍 Trying Alpha Vantage for {ticker}")
        
        if ALPHA_VANTAGE_API_KEY == 'demo':
            logger.warning("⚠️  Using demo API key for Alpha Vantage - limited functionality")
        
        session = create_robust_session()
        
        # Get daily adjusted data
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'compact',  # Last 100 days
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = session.get(url, params=params, timeout=30)
        data = response.json()
        
        # Check for errors
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError("Alpha Vantage rate limit exceeded")
        
        if 'Time Series (Daily)' not in data:
            raise ValueError("No time series data in Alpha Vantage response")
        
        # Convert to DataFrame
        time_series = data['Time Series (Daily)']
        df_data = []
        
        for date_str, values in time_series.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['6. volume'])
            })
        
        df = pd.DataFrame(df_data).set_index('Date').sort_index()
        
        if len(df) < 30:
            raise ValueError(f"Insufficient data: only {len(df)} days available")
        
        # Get company info
        info = {
            'ticker': ticker,
            'current_price': float(df['Close'].iloc[-1]),
            'previous_close': float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Close'].iloc[-1]),
            'market_cap': 'N/A',
            'company_name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'description': f"Stock data for {ticker} from Alpha Vantage"
        }
        
        logger.info(f"✅ Alpha Vantage: Got {len(df)} days of data for {ticker}")
        return {
            'success': True,
            'data': df,
            'info': info,
            'source': 'alpha_vantage'
        }
        
    except Exception as e:
        logger.warning(f"❌ Alpha Vantage failed for {ticker}: {e}")
        return {'success': False, 'error': str(e)}

def fetch_twelve_data(ticker: str) -> dict:
    """Fetch data from Twelve Data API"""
    try:
        logger.info(f"🔍 Trying Twelve Data for {ticker}")
        
        session = create_robust_session()
        
        # Get time series data
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker,
            'interval': '1day',
            'outputsize': 100,
            'apikey': TWELVE_DATA_API_KEY if TWELVE_DATA_API_KEY != 'demo' else None
        }
        
        # Remove apikey if demo
        if params['apikey'] is None:
            del params['apikey']
        
        response = session.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'status' in data and data['status'] == 'error':
            raise ValueError(f"Twelve Data error: {data.get('message', 'Unknown error')}")
        
        if 'values' not in data:
            raise ValueError("No time series data in Twelve Data response")
        
        # Convert to DataFrame
        df_data = []
        for item in data['values']:
            df_data.append({
                'Date': pd.to_datetime(item['datetime']),
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': int(item['volume'])
            })
        
        df = pd.DataFrame(df_data).set_index('Date').sort_index()
        
        if len(df) < 30:
            raise ValueError(f"Insufficient data: only {len(df)} days available")
        
        info = {
            'ticker': ticker,
            'current_price': float(df['Close'].iloc[-1]),
            'previous_close': float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Close'].iloc[-1]),
            'market_cap': 'N/A',
            'company_name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'description': f"Stock data for {ticker} from Twelve Data"
        }
        
        logger.info(f"✅ Twelve Data: Got {len(df)} days of data for {ticker}")
        return {
            'success': True,
            'data': df,
            'info': info,
            'source': 'twelve_data'
        }
        
    except Exception as e:
        logger.warning(f"❌ Twelve Data failed for {ticker}: {e}")
        return {'success': False, 'error': str(e)}

def fetch_yahoo_finance_direct(ticker: str) -> dict:
    """Fetch data directly from Yahoo Finance (unofficial)"""
    try:
        logger.info(f"🔍 Trying Yahoo Finance direct for {ticker}")
        
        session = create_robust_session()
        
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (90 * 24 * 60 * 60)  # 90 days ago
        
        # Yahoo Finance direct API
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': '1d',
            'events': 'history'
        }
        
        response = session.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Yahoo Finance returned status {response.status_code}")
        
        # Parse CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if df.empty or len(df) < 30:
            raise ValueError(f"Insufficient data: only {len(df)} days available")
        
        # Clean and format data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # Handle null values
        df = df.dropna()
        
        if len(df) < 30:
            raise ValueError(f"After cleaning: only {len(df)} days available")
        
        info = {
            'ticker': ticker,
            'current_price': float(df['Close'].iloc[-1]),
            'previous_close': float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Close'].iloc[-1]),
            'market_cap': 'N/A',
            'company_name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'description': f"Stock data for {ticker} from Yahoo Finance"
        }
        
        logger.info(f"✅ Yahoo Finance Direct: Got {len(df)} days of data for {ticker}")
        return {
            'success': True,
            'data': df,
            'info': info,
            'source': 'yahoo_direct'
        }
        
    except Exception as e:
        logger.warning(f"❌ Yahoo Finance direct failed for {ticker}: {e}")
        return {'success': False, 'error': str(e)}

def generate_realistic_mock_data(ticker: str) -> dict:
    """Generate realistic mock data as final fallback"""
    logger.warning(f"🎭 Generating realistic mock data for {ticker}")
    
    # Seed for consistency
    np.random.seed(hash(ticker) % 2**32)
    
    # Base price based on ticker
    base_prices = {
        'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'TSLA': 200, 'NVDA': 400,
        'AMZN': 3000, 'META': 300, 'NFLX': 400, 'AMD': 80, 'INTC': 50
    }
    base_price = base_prices.get(ticker, 50 + (hash(ticker) % 200))
    
    # Generate 90 days of realistic data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    dates = dates[dates.dayofweek < 5]  # Business days only
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Small daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        # Add some bounds to keep it realistic
        if new_price < base_price * 0.7:
            new_price = base_price * 0.7
        elif new_price > base_price * 1.3:
            new_price = base_price * 1.3
        prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate OHLCV from close price
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(2000000, 800000))
        volume = max(volume, 100000)  # Minimum volume
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    info = {
        'ticker': ticker,
        'current_price': float(df['Close'].iloc[-1]),
        'previous_close': float(df['Close'].iloc[-2]),
        'market_cap': 'N/A (Mock Data)',
        'company_name': f'{ticker} Corporation (Demo)',
        'sector': 'Technology (Demo)',
        'industry': 'Software (Demo)',
        'description': f"Mock data for demonstration purposes. Real-time data unavailable for {ticker}."
    }
    
    logger.info(f"✅ Generated {len(df)} days of mock data for {ticker}")
    return {
        'success': True,
        'data': df,
        'info': info,
        'source': 'mock_data'
    }

@cache.memoize(timeout=600)  # Cache for 10 minutes
def get_stock_data_multi_source(ticker: str) -> dict:
    """Try multiple data sources in order of preference"""
    logger.info(f"🎯 Multi-source data fetch for {ticker}")
    
    # List of data sources to try in order
    sources = [
        fetch_alpha_vantage_data,
        fetch_twelve_data,
        fetch_yahoo_finance_direct,
        generate_realistic_mock_data  # Final fallback
    ]
    
    for source_func in sources:
        try:
            result = source_func(ticker)
            if result.get('success'):
                logger.info(f"✅ Successfully fetched {ticker} data from {result.get('source')}")
                return result
        except Exception as e:
            logger.warning(f"Source {source_func.__name__} failed: {e}")
            continue
    
    # If all sources fail
    logger.error(f"❌ All data sources failed for {ticker}")
    return {
        'success': False,
        'error': f"Unable to fetch data for {ticker} from any source. Please verify the ticker symbol.",
        'source': 'none'
    }

@cache.memoize(timeout=600)
def calculate_technical_indicators(ticker: str) -> Optional[Dict]:
    """Calculate technical indicators using multi-source data"""
    try:
        # Get stock data from multiple sources
        result = get_stock_data_multi_source(ticker)
        
        if not result.get('success'):
            suggestions = suggest_similar_tickers(ticker)
            return {
                'error': result.get('error', 'Data fetch failed'),
                'suggestions': suggestions,
                'help': f"Try: {', '.join(suggestions)}",
                'success': False
            }
        
        df = result['data']
        
        # Validate data
        if df.empty or len(df) < 30:
            return {
                'error': f"Insufficient data for {ticker}. Need at least 30 days.",
                'help': "Try a different stock symbol",
                'success': False
            }
        
        logger.info(f"📊 Calculating indicators for {ticker} using {len(df)} data points")
        
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
            'data_source': result['source'],
            'data_points': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            },
            'company_info': result['info'],
            'market_status': {
                'is_open': is_market_open(),
                'current_time_et': datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')
            }
        }
        
        logger.info(f"✅ Calculated indicators for {ticker}: RSI={RSI_14:.1f}, Price=${result['info']['current_price']:.2f}")
        
        return {
            'indicators': indicators,
            'metadata': metadata,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating indicators for {ticker}: {e}")
        suggestions = suggest_similar_tickers(ticker)
        return {
            'error': f"Failed to process data for {ticker}: {str(e)}",
            'suggestions': suggestions,
            'success': False
        }

def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or len(ticker.strip()) == 0:
        return False
    ticker = ticker.strip().upper()
    return 1 <= len(ticker) <= 5 and ticker.replace('.', '').isalpha()

def make_prediction(df: pd.DataFrame, input_data: dict) -> tuple:
    """Make prediction using the ML model"""
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

# Middleware
@app.before_request
def log_request_info():
    request.start_time = time.time()

@app.after_request
def log_request_response(response):
    duration = time.time() - request.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}s")
    return response

# API Endpoints
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_keys_available = {
        'alpha_vantage': ALPHA_VANTAGE_API_KEY != 'demo',
        'twelve_data': TWELVE_DATA_API_KEY != 'demo'
    }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '4.0.0',
        'data_sources': ['alpha_vantage', 'twelve_data', 'yahoo_direct', 'mock_fallback'],
        'api_keys_configured': api_keys_available,
        'market_open': is_market_open(),
        'cors_enabled': True
    })

@app.route('/predict-ticker', methods=['POST'])
@limiter.limit("15 per minute")  # Lower limit due to API calls
def predict_ticker():
    """Predict stock movement for a ticker"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
            
        ticker = data['ticker'].strip().upper()
        logger.info(f"🎯 Prediction request for {ticker}")
        
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
            
        # Make prediction
        df = pd.DataFrame([result['indicators']])[feature_names]
        prediction_response, status = make_prediction(df, result['indicators'])
        
        if status == 200:
            prediction_response.update({
                'ticker_info': result['metadata'],
                'technical_indicators': result['indicators']
            })
            logger.info(f"✅ Prediction complete for {ticker}: {prediction_response['signal']}")
        
        return jsonify(prediction_response), status
        
    except Exception as e:
        logger.error(f"❌ Ticker prediction error: {e}")
        return jsonify({'error': f'Ticker prediction failed: {str(e)}'}), 500

@app.route('/ticker-info', methods=['POST'])
@limiter.limit("20 per minute")
def get_ticker_info():
    """Get ticker information and technical indicators"""
    try:
        data = request.json
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker symbol required'}), 400
            
        ticker = data['ticker'].strip().upper()
        logger.info(f"📊 Info request for {ticker}")
        
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
        logger.error(f"❌ Ticker info error: {e}")
        return jsonify({'error': f'Failed to get ticker info: {str(e)}'}), 500

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

@app.route('/features', methods=['GET'])
def get_expected_features():
    """Get expected feature names"""
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
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        api_status = {
            'alpha_vantage': 'configured' if ALPHA_VANTAGE_API_KEY != 'demo' else 'demo_key',
            'twelve_data': 'configured' if TWELVE_DATA_API_KEY != 'demo' else 'demo_key',
            'yahoo_direct': 'available',
            'mock_fallback': 'available'
        }
        
        return jsonify({
            'model_type': type(model).__name__,
            'timestamp': datetime.now().isoformat(),
            'version': '4.0.0',
            'feature_count': len(feature_names),
            'features': feature_names,
            'data_sources': api_status,
            'environment': os.getenv('FLASK_ENV', 'development')
        })
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/test-data-sources/<ticker>', methods=['GET'])
def test_data_sources(ticker):
    """Test endpoint to check all data sources for a ticker"""
    try:
        logger.info(f"🧪 Testing all data sources for {ticker}")
        
        results = {}
        sources = [
            ('alpha_vantage', fetch_alpha_vantage_data),
            ('twelve_data', fetch_twelve_data),
            ('yahoo_direct', fetch_yahoo_finance_direct),
            ('mock_data', generate_realistic_mock_data)
        ]
        
        for source_name, source_func in sources:
            try:
                result = source_func(ticker)
                results[source_name] = {
                    'success': result.get('success', False),
                    'data_points': len(result.get('data', [])) if result.get('success') else 0,
                    'error': result.get('error', None),
                    'source': result.get('source', source_name)
                }
                if result.get('success'):
                    logger.info(f"✅ {source_name}: {results[source_name]['data_points']} data points")
                else:
                    logger.warning(f"❌ {source_name}: {results[source_name]['error']}")
            except Exception as e:
                results[source_name] = {
                    'success': False,
                    'error': str(e),
                    'data_points': 0
                }
                logger.error(f"❌ {source_name} exception: {e}")
        
        return jsonify({
            'ticker': ticker,
            'test_results': results,
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'Use the first successful source in order of preference'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api-status', methods=['GET'])
def api_status():
    """Check the status of external APIs"""
    status = {}
    
    # Test Alpha Vantage
    try:
        if ALPHA_VANTAGE_API_KEY != 'demo':
            session = create_robust_session()
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': 'AAPL',
                'interval': '1min',
                'apikey': ALPHA_VANTAGE_API_KEY,
                'outputsize': 'compact'
            }
            response = session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Error Message' in data:
                status['alpha_vantage'] = {'status': 'error', 'message': data['Error Message']}
            elif 'Note' in data:
                status['alpha_vantage'] = {'status': 'rate_limited', 'message': 'API rate limit exceeded'}
            else:
                status['alpha_vantage'] = {'status': 'ok', 'message': 'API key working'}
        else:
            status['alpha_vantage'] = {'status': 'demo', 'message': 'Using demo API key'}
    except Exception as e:
        status['alpha_vantage'] = {'status': 'error', 'message': str(e)}
    
    # Test Twelve Data
    try:
        if TWELVE_DATA_API_KEY != 'demo':
            session = create_robust_session()
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'AAPL',
                'interval': '1min',
                'outputsize': 1,
                'apikey': TWELVE_DATA_API_KEY
            }
            response = session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'status' in data and data['status'] == 'error':
                status['twelve_data'] = {'status': 'error', 'message': data.get('message', 'Unknown error')}
            else:
                status['twelve_data'] = {'status': 'ok', 'message': 'API key working'}
        else:
            status['twelve_data'] = {'status': 'demo', 'message': 'Using demo/free tier'}
    except Exception as e:
        status['twelve_data'] = {'status': 'error', 'message': str(e)}
    
    # Test Yahoo Finance Direct
    try:
        session = create_robust_session()
        end_time = int(time.time())
        start_time = end_time - (7 * 24 * 60 * 60)  # 7 days ago
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/AAPL"
        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': '1d',
            'events': 'history'
        }
        
        response = session.get(url, params=params, timeout=10)
        
        if response.status_code == 200 and 'Date,Open,High,Low,Close' in response.text:
            status['yahoo_direct'] = {'status': 'ok', 'message': 'Yahoo Finance direct access working'}
        else:
            status['yahoo_direct'] = {'status': 'error', 'message': f'Status code: {response.status_code}'}
    except Exception as e:
        status['yahoo_direct'] = {'status': 'error', 'message': str(e)}
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'api_status': status,
        'overall_status': 'healthy' if any(s.get('status') == 'ok' for s in status.values()) else 'degraded'
    })

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

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"
    
    logger.info("🚀 Starting Multi-Source Stock Predictor API v4.0...")
    logger.info("📡 Available endpoints:")
    logger.info("   GET  /                    - Health check")
    logger.info("   POST /predict-ticker     - Predict ticker movement")
    logger.info("   POST /ticker-info        - Get ticker information")
    logger.info("   POST /predict            - Manual prediction")
    logger.info("   GET  /features           - Get expected features")
    logger.info("   GET  /model-info         - Get model information")
    logger.info("   GET  /test-data-sources/<ticker> - Test all data sources")
    logger.info("   GET  /api-status         - Check API status")
    logger.info("📊 Data Sources:")
    logger.info(f"   1. Alpha Vantage: {'✅ Configured' if ALPHA_VANTAGE_API_KEY != 'demo' else '⚠️  Demo key'}")
    logger.info(f"   2. Twelve Data: {'✅ Configured' if TWELVE_DATA_API_KEY != 'demo' else '⚠️  Demo key'}")
    logger.info("   3. Yahoo Finance Direct: ✅ Available")
    logger.info("   4. Mock Data: ✅ Fallback available")
    logger.info(f"🌐 Running on port {port}")
    
    app.run(debug=debug, host="0.0.0.0", port=port)