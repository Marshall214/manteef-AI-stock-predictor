from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Configure CORS properly for deployed frontend
CORS(app, origins=[
    "http://localhost:3000",  # Dev
    "https://*.netlify.app",  # Netlify
    "https://*.vercel.app",   # Vercel
    "https://legendary-kangaroo-ac2fd5.netlify.app",  # Your actual URL
])

# Load model
MODEL_PATH = os.path.join('model', 'xgb_stock_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {os.path.abspath(MODEL_PATH)}")
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"Expected features: {feature_names}")
    else:
        feature_names = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = None

# -------------------------
# Routes
# -------------------------
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

        if df.isnull().any().any():
            return jsonify({'error': 'Missing values in input data'}), 400

        # Make prediction safely
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = float(max(probabilities))

        # Determine signal
        signal = 'BUY' if prediction == 1 else 'SELL'
        if confidence >= 0.7:
            strength = "STRONG"
        elif confidence >= 0.6:
            strength = "MODERATE"
        else:
            strength = "WEAK"

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

    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500


@app.route('/features', methods=['GET'])
def get_features():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if feature_names:
            features = feature_names
        else:
            # fallback
            features = [
                'pct_change', 'ma_7', 'ma_21', 'volatility_7', 'volume',
                'RSI_14', 'momentum_7', 'momentum_21', 'ma_diff', 'vol_ratio_20'
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
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        info = {
            'model_type': type(model).__name__,
            'timestamp': datetime.now().isoformat(),
            'environment': os.getenv('FLASK_ENV', 'development')
        }

        if feature_names:
            info['feature_count'] = len(feature_names)
            info['features'] = feature_names

        # Safe handling for pipeline or single estimator
        estimator = model
        if hasattr(model, 'steps'):  # pipeline
            estimator = model.steps[-1][1]

        if hasattr(estimator, 'get_params'):
            params = estimator.get_params()
            info['key_parameters'] = {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate')
            }

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': f'Model info failed: {str(e)}'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

# -------------------------
# Run the app
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") != "production"

    print("🚀 Starting Manteef Stock Predictor API...")
    print(f"🌐 Running on port {port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
