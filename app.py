from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load the model from the 'model' folder
model_path = os.path.join('model', 'xgb_stock_model.pkl')
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {os.path.abspath(model_path)}")
    
    # Get feature names if available
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"Expected features: {feature_names}")
    else:
        print("Feature names not available in model")
        
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict stock movement"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Expect JSON with feature values
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Basic validation
        if df.empty:
            return jsonify({'error': 'Empty data provided'}), 400
            
        # Check for missing values
        if df.isnull().any().any():
            return jsonify({'error': 'Missing values in input data'}), 400
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Calculate confidence (max probability)
        confidence = float(max(probabilities))
        
        # Determine signal strength based on confidence
        if confidence >= 0.7:
            strength = "STRONG"
        elif confidence >= 0.6:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        signal = 'BUY' if prediction == 1 else 'SELL'
        
        return jsonify({
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
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/features', methods=['GET'])
def get_expected_features():
    """Get the expected feature names for the model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
        else:
            # If feature names not available, return common expected features
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'pct_change', 'ma_7', 'ma_21', 'volatility_70',
                'rsi', 'momentum', 'difference', 'average'
            ]
        
        return jsonify({
            'features': features,
            'feature_count': len(features),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get features: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': type(model).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feature information if available
        if hasattr(model, 'feature_names_in_'):
            info['feature_count'] = len(model.feature_names_in_)
            info['features'] = model.feature_names_in_.tolist()
        
        # Add XGBoost specific info if it's an XGBoost model
        if hasattr(model, 'get_params'):
            params = model.get_params()
            info['key_parameters'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate')
            }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    print("🚀 Starting Flask Stock Prediction API...")
    print("📡 Available endpoints:")
    print("   GET  /           - Health check")
    print("   POST /predict    - Make prediction")
    print("   GET  /features   - Get expected features")
    print("   GET  /model-info - Get model information")
    
    app.run(debug=True, host='0.0.0.0', port=5000)