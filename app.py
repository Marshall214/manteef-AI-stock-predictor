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
import requests
from typing import Dict, Optional
import logging
import psutil
import pytz
from functools import wraps
import time
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

# Multiple User Agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
]

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
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def create_enhanced_session():
    """Create a robust session with rotation and retry logic"""
    session = requests.Session()
    
    # Random user agent
    user_agent = random.choice(USER_AGENTS)
    session.headers.update({
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    })
    
    # Add retry strategy
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_stock_data_enhanced(ticker: str, period_days: int = 90) -> dict:
    """
    Enhanced stock data fetching with multiple strategies and fallbacks
    """
    ticker = ticker.upper().strip()
    logger.info(f"🔍 Fetching data for {ticker} using enhanced method")
    
    # Strategy 1: Standard yfinance with enhanced session
    try:
        logger.info(f"📊 Strategy 1: Enhanced yfinance for {ticker}")
        session = create_enhanced_session()
        stock = yf.Ticker(ticker, session=session)
        
        # Try different period formats
        periods_to_try = ['3mo', '6mo', '1y', '2y']
        hist_data = None
        
        for period in periods_to_try:
            try:
                logger.info(f"  Trying period: {period}")
                hist_data = stock.history(period=period, auto_adjust=True, prepost=True)
                if not hist_data.empty and len(hist_data) >= 30:
                    logger.info(f"  ✅ Success with {period}: {len(hist_data)} days")
                    break
                else:
                    logger.info(f"  ⚠️ {period} returned {len(hist_data)} days")
            except Exception as e:
                logger.info(f"  ❌ {period} failed: {e}")
                continue
        
        if hist_data is None or hist_data.empty or len(hist_data) < 30:
            raise ValueError(f"All period attempts failed for {ticker}")
        
        # Try to get company info with timeout
        info = {}
        try:
            logger.info(f"  Getting company info for {ticker}")
            stock_info = stock.info
            if stock_info and isinstance(stock_info, dict) and len(stock_info) > 5:
                info = {
                    'ticker': ticker,
                    'current_price': stock_info.get('currentPrice') or stock_info.get('regularMarketPrice') or float(hist_data['Close'].iloc[-1]),
                    'previous_close': stock_info.get('previousClose') or stock_info.get('regularMarketPreviousClose') or (float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else float(hist_data['Close'].iloc[-1])),
                    'market_cap': stock_info.get('marketCap', 'N/A'),
                    'company_name': stock_info.get('longName') or stock_info.get('shortName') or ticker,
                    'sector': stock_info.get('sector', 'N/A'),
                    'industry': stock_info.get('industry', 'N/A'),
                    'description': stock_info.get('longBusinessSummary', f"Stock information for {ticker}")[:200] + "..." if stock_info.get('longBusinessSummary') else f"Stock information for {ticker}"
                }
                logger.info(f"  ✅ Company info retrieved: {info.get('company_name', 'N/A')}")
            else:
                raise ValueError("Info data insufficient")
        except Exception as e:
            logger.warning(f"  ⚠️ Company info failed, using basic info: {e}")
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
            'source': 'yfinance_enhanced'
        }
        
    except Exception as e:
        logger.warning(f"❌ Strategy 1 failed for {ticker}: {e}")
        
    # Strategy 2: Simplified yfinance without session
    try:
        logger.info(f"📊 Strategy 2: Simple yfinance for {ticker}")
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='6mo')
        
        if not hist_data.empty and len(hist_data) >= 20:
            logger.info(f"  ✅ Simple method success: {len(hist_data)} days")
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
                'source': 'yfinance_simple'
            }
        else:
            raise ValueError("Insufficient data from simple method")
            
    except Exception as e:
        logger.warning(f"❌ Strategy 2 failed for {ticker}: {e}")
    
    # Strategy 3: Alternative API approach (direct Yahoo Finance API)
    try:
        logger.info(f"📊 Strategy 3: Direct Yahoo API for {ticker}")
        return get_yahoo_direct_data(ticker)
    except Exception as e:
        logger.warning(f"❌ Strategy 3 failed for {ticker}: {e}")
    
    # All strategies failed
    logger.error(f"❌ All strategies failed for {ticker}")
    suggestions = suggest_similar_tickers(ticker)
    return {
        'success': False,
        'error': f"Could not fetch data for {ticker}. This could be due to network restrictions or invalid ticker symbol.",
        'suggestions': suggestions,
        'help': f"Try: {', '.join(suggestions)} or check if {ticker} is a valid stock symbol",
        'source': 'none'
    }

def get_yahoo_direct_data(ticker: str) -> dict:
    """
    Direct Yahoo Finance API approach as fallback
    """
    session = create_enhanced_session()
    
    # Get historical data directly from Yahoo Finance API
    end_date = int(datetime.now().timestamp())
    start_date = int((datetime.now() - timedelta(days=365)).timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        'period1': start_date,
        'period2': end_date,
        'interval': '1d',
        'includePrePost': 'true'
    }
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/json'
    }
    
    response = session.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    
    data = response.json()
    
    if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
        raise ValueError(f"No data returned from Yahoo API for {ticker}")
    
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quote = result['indicators']['quote'][0]
    
    # Create DataFrame
    df_data = {
        'Open': quote.get('open', []),
        'High': quote.get('high', []),
        'Low': quote.get('low', []),
        'Close': quote.get('close', []),
        'Volume': quote.get('volume', [])
    }
    
    # Remove None values
    valid_indices = [i for i, close in enumerate(df_data['Close']) if close is not None]
    
    if len(valid_indices) < 30:
        raise ValueError(f"Insufficient valid data points for {ticker}")
    
    # Filter data
    filtered_data = {}
    filtered_timestamps = []
    
    for i in valid_indices:
        filtered_timestamps.append(timestamps[i])
        for key in df_data:
            if key not in filtered_data:
                filtered_data[key] = []
            filtered_data[key].append(df_data[key][i] or 0)
    
    # Create DataFrame with proper timestamps
    hist_data = pd.DataFrame(filtered_data)
    hist_data.index = pd.to_datetime(filtered_timestamps, unit='s')
    
    # Basic info
    meta = result.get('meta', {})
    info = {
        'ticker': ticker,
        'current_price': float(meta.get('regularMarketPrice', hist_data['Close'].iloc[-1])),
        'previous_close': float(meta.get('previousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else hist_data['Close'].iloc[-1])),
        'market_cap': 'N/A',
        'company_name': meta.get('longName', ticker),
        'sector': 'N/A',
        'industry': 'N/A',
        'description': f"Stock information for {ticker}"
    }
    
    return {
        'success': True,
        'data': hist_data,
        'info': info,
        'source': 'yahoo_direct'
    }

@cache.memoize(timeout=300)
def calculate_technical_indicators(ticker: str, period_days: int = 90) -> Optional[Dict]:
    """Calculate technical indicators using enhanced data fetching"""
    try:
        # Get stock data using enhanced method
        result = get_stock_data_enhanced(ticker, period_days)
        
        if not result['success']:
            return {
                'error': result['error'],
                'suggestions': result.get('suggestions', []),
                'help': result.get('help', 'Please try a different ticker'),
                'success': False
            }
        
        df = result['data']
        
        # Validate data
        if df.empty or len(df) < 30:
            available_days = len(df)
            return {
                'error': f"Not enough trading history for {ticker}. Found {available_days} days, need at least 30. This might be a newly listed stock.",
                'suggestions': suggest_similar_tickers(ticker),
                'help': "Try a different stock with longer trading history",
                'success': False
            }
        
        logger.info(f"✅ Processing {len(df)} days of data for {ticker} from {result['source']}")
        
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
        volume = float(volumes.iloc[-1]) if not pd.isna(volumes.iloc[-1]) else 1000000
        RSI_14 = calculate_rsi(close_prices, 14)
        momentum_7 = (close_prices.iloc[-1] - close_prices.iloc[-8]) / close_prices.iloc[-8] if len(close_prices) > 7 else 0.0
        momentum_21 = (close_prices.iloc[-1] - close_prices.iloc[-22]) / close_prices.iloc[-22] if len(close_prices) > 21 else 0.0
        ma_diff = ma_7 - ma_21
        avg_volume_20 = volumes.rolling(window=20).mean().iloc[-1] if len(volumes) >= 20 else volumes.mean()
        vol_ratio_20 = volume / avg_volume_20 if avg_volume_20 > 0 and not pd.isna(avg_volume_20) else 1.0
        
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
            'company_info': result['info']
        }
        
        final_result = {
            'indicators': indicators,
            'metadata': metadata,
            'success': True
        }
        
        logger.info(f"✅ Technical indicators calculated for {ticker}")
        return add_market_context(final_result)
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {e}")
        suggestions = suggest_similar_tickers(ticker)
        return {
            'error': f"Failed to process data for {ticker}: {str(e)}",
            'suggestions': suggestions,
            'help': f"Try: {', '.join(suggestions)}",
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
    
    if request.method == 'POST' and request.content_type and 'application/json' in request.content_type:
        try:
            if hasattr(request, 'json') and request.json:
                ticker = request.json.get('ticker', 'unknown')
                logger.info(f"Ticker request: {ticker} - Status: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not log ticker info: {e}")
    
    return response

# Add diagnostic endpoint for testing
@app.route('/test-yfinance', methods=['GET'])
def test_yfinance_endpoint():
    """Test yfinance functionality"""
    results = []
    
    def log_result(message, success=True):
        results.append({
            'message': message,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    # Test basic yfinance
    try:
        import yfinance as yf
        log_result(f"✅ YFinance imported, version: {yf.__version__}")
    except Exception as e:
        log_result(f"❌ YFinance import failed: {e}", False)
        return jsonify({'results': results})
    
    # Test enhanced method
    test_tickers = ['AAPL', 'MSFT', 'TSLA']
    for ticker in test_tickers:
        try:
            result = get_stock_data_enhanced(ticker)
            if result['success']:
                log_result(f"✅ {ticker}: {len(result['data'])} days via {result['source']}")
            else:
                log_result(f"❌ {ticker}: {result.get('error', 'Failed')}", False)
        except Exception as e:
            log_result(f"❌ {ticker}: Exception - {e}", False)
    
    return jsonify({
        'results': results,
        'summary': {
            'total_tests': len(results),
            'passed': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']])
        }
    })

# API Endpoints
@app.route('/', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.2.0',
        'environment': os.getenv('FLASK_ENV', 'production'),
        'features': ['manual_prediction', 'ticker_prediction', 'ticker_info', 'enhanced_yfinance'],
        'data_sources': ['yfinance_enhanced', 'yfinance_simple', 'yahoo_direct'],
        'market_open': is_market_open(),
        'cors_enabled': True,
        'enhancements': ['multiple_user_agents', 'retry_logic', 'fallback_strategies']
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
    """Ticker prediction using enhanced yfinance"""
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
            
        # Calculate technical indicators using enhanced method
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
            'environment': os.getenv('FLASK_ENV', 'production'),
            'feature_count': len(feature_names),
            'features': feature_names,
            'data_sources': {
                'primary': 'Enhanced Yahoo Finance (yfinance)',
                'fallback_1': 'Simple yfinance',
                'fallback_2': 'Direct Yahoo API',
                'strategies': 3
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
    logger.info("🚀 Starting Enhanced Manteef Stock Predictor API v3.2...")
    logger.info("📡 Available endpoints:")
    logger.info("   GET  /           - Health check")
    logger.info("   GET  /test-yfinance - Diagnostic test")
    logger.info("   POST /predict    - Make prediction (manual input)")
    logger.info("   POST /predict-ticker - Make prediction from ticker")
    logger.info("   POST /ticker-info - Get ticker technical indicators")
    logger.info("   GET  /features   - Get expected features")
    logger.info("   GET  /model-info - Get model information")
    logger.info("📊 Data Sources: Enhanced YFinance with 3 fallback strategies")
    logger.info(f"🌐 Running on port {port}")
    logger.info("✅ CORS enabled for specified origins")
    app.run(debug=debug, host="0.0.0.0", port=port)