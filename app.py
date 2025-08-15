from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# CORS configuration
CORS(app, origins=[
    "http://localhost:3000",  # Local dev
    "https://*.netlify.app",  # Any Netlify subdomain
    "https://*.vercel.app",   # Any Vercel subdomain
    "https://legendary-kangaroo-ac2fd5.netlify.app"  # Your specific frontend
])

# model load
model_path = os.path.join('model', 'xgb_stock_model.pkl')
try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {os.path.abspath(model_path)}")
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"Expected features: {feature_names}")
    else:
        feature_names = [
            'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
            'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
        ]
        print("Feature names not found in model. Using default features.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []

# endpoints
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'environment': os.getenv('FLASK_ENV', 'development')
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Ensure all expected features are present
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Create DataFrame with correct feature order
        df = pd.DataFrame([data])
        df = df[feature_names]  # Ensure correct column order
        
        if df.empty or df.isnull().any().any():
            return jsonify({'error': 'Empty or missing values in input data'}), 400

        # Make prediction - XGBoost returns probabilities for binary classification
        try:
            # For XGBoost binary classification, predict returns class probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)[0]
                prediction = 1 if probabilities[1] > 0.5 else 0
            else:
                # XGBoost classifier with predict method
                prediction_prob = model.predict(df)[0]
                if isinstance(prediction_prob, float):
                    # Raw probability output
                    probabilities = [1 - prediction_prob, prediction_prob]
                    prediction = 1 if prediction_prob > 0.5 else 0
                else:
                    # Direct class prediction
                    prediction = int(prediction_prob)
                    probabilities = [1 - prediction, prediction]
                    
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

        confidence = float(max(probabilities))
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
        print(f"Prediction error: {e}")  # Add logging
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


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

        # Get XGBoost-specific parameters
        try:
            if hasattr(model, 'get_params'):
                # Scikit-learn wrapper (XGBClassifier)
                params = model.get_params()
                info['key_parameters'] = {
                    'n_estimators': params.get('n_estimators'),
                    'max_depth': params.get('max_depth'),
                    'learning_rate': params.get('learning_rate'),
                    'objective': params.get('objective'),
                    'subsample': params.get('subsample')
                }
            elif hasattr(model, 'get_xgb_params'):
                # Native XGBoost model
                params = model.get_xgb_params()
                info['key_parameters'] = {
                    'n_estimators': params.get('n_estimators'),
                    'max_depth': params.get('max_depth'),
                    'learning_rate': params.get('eta', params.get('learning_rate')),
                    'objective': params.get('objective'),
                    'subsample': params.get('subsample')
                }
            elif hasattr(model, 'save_config'):
                # XGBoost 2.0+ format
                info['key_parameters'] = {'note': 'XGBoost 2.0+ model - parameters embedded in model'}
            else:
                print(f"Model type {model_type} - no parameter extraction method found")
                info['key_parameters'] = {'note': f'Parameters not accessible for {model_type}'}
                
        except Exception as e:
            print(f"Warning: Could not get model parameters: {e}")
            info['key_parameters'] = {'error': f'Parameter extraction failed: {str(e)}'}

        return jsonify(info)
    except Exception as e:
        print(f"Model info error: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500


# error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


# run app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"

    print("🚀 Starting Manteef Stock Predictor API...")
    print("📡 Available endpoints:")
    print("   GET  /           - Health check")
    print("   POST /predict    - Make prediction")
    print("   GET  /features   - Get expected features")
    print("   GET  /model-info - Get model information")
    print(f"🌐 Running on port {port}")

    app.run(debug=debug, host="0.0.0.0", port=port)