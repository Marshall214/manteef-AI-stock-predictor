import pytest
import os
import sys
from app import app, calculate_features, prepare_features, calculate_rsi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import xgboost as xgb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configurations
TEST_TICKER = "AAPL"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_stock_model.json")

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_predict_ticker_missing_ticker(client):
    """Test prediction endpoint with missing ticker"""
    response = client.post('/predict-ticker', json={})
    assert response.status_code == 400
    assert "error" in response.json

def test_predict_ticker_invalid_ticker(client):
    """Test prediction endpoint with invalid ticker"""
    response = client.post('/predict-ticker', json={"ticker": "INVALID_TICKER_123"})
    assert response.status_code == 404
    assert "error" in response.json

def test_predict_ticker_valid(client):
    """Test prediction endpoint with valid ticker"""
    response = client.post('/predict-ticker', json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "confidence" in response.json
    assert "features" in response.json

def test_calculate_features():
    """Test feature calculation function"""
    # Get sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    stock = yf.Ticker("AAPL")
    hist = stock.history(start=start_date, end=end_date)
    
    # Calculate features
    result = calculate_features(hist)
    
    # Check if all required features are present
    required_features = [
        'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
        'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
    ]
    
    for feature in required_features:
        assert feature in result.columns

def test_prepare_features():
    """Test feature preparation function"""
    # Create sample data
    data = pd.DataFrame({
        'pct_change': [0.01],
        'ma_7': [100],
        'ma_21': [101],
        'volatility_7': [0.02],
        'volume': [1000000],
        'RSI_14': [55],
        'momentum_7': [0.03],
        'momentum_21': [0.04],
        'ma_diff': [-1],
        'vol_ratio_20': [1.1]
    })
    
    # Prepare features
    result = prepare_features(data)
    
    # Check if all features are present in the result
    assert len(result) == 10
    assert all(key in result for key in [
        'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
        'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
    ])
