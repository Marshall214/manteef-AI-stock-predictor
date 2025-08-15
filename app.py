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

        df = pd.DataFrame([data])
        if df.empty or df.isnull().any().any():
            return jsonify({'error': 'Empty or missing values in input data'}), 400

        # safe predictions
        pred_model = model
        try:
            # If pipeline, extract final estimator
            if hasattr(model, 'named_steps') and 'xgb' in model.named_steps:
                pred_model = model.named_steps['xgb']
        except Exception:
            pred_model = model

        prediction = model.predict(df)[0]  # pipeline ensures preprocessing
        try:
            probabilities = pred_model.predict_proba(df)[0]
        except AttributeError:
            probabilities = [1 - prediction, prediction] if prediction in [0, 1] else [0.5, 0.5]

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
        info = {
            'model_type': type(model).__name__,
            'timestamp': datetime.now().isoformat(),
            'environment': os.getenv('FLASK_ENV', 'development'),
            'feature_count': len(feature_names),
            'features': feature_names
        }

        # Safely add key parameters if available
        try:
            params = model.get_params()
            info['key_parameters'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate')
            }
        except Exception:
            info['key_parameters'] = {}

        return jsonify(info)
    except Exception as e:
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
