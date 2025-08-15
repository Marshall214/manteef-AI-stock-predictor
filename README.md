# 📈 Manteef Stock Predictor API

A Flask-based REST API for making stock buy/sell predictions using an **XGBoost machine learning model**.  
Built to be consumed by a React (or any) frontend and deployed on cloud services like Railway.

---

## Features
- **Health Check Endpoint** to verify deployment status.
- **Prediction Endpoint** to return buy/sell signal, probability, and signal strength.
- **Model Info Endpoint** to inspect the loaded model’s parameters.
- **Expected Features Endpoint** to list the model's required inputs.
- Cross-Origin Resource Sharing (**CORS**) enabled for multiple frontend domains.

---

## 🗂 Project Structure
.
├── app.py # Flask backend
├── requirements.txt # Dependencies
├── model/ # Stored trained model (.json)
├── data/ # CSV datasets (if any)
├── notebooks/ # Jupyter notebooks for training
├── static/ # Frontend assets (if applicable)
└── README.md # Documentation


---

## 🛠 Tech Stack
- **Backend:** Flask, Flask-CORS
- **ML Framework:** XGBoost
- **Data Processing:** Pandas, NumPy
- **Deployment:** Railway (Backend), Netlify (Frontend)
- **Serving:** Gunicorn (production WSGI server)

---

## ⚙Installation & Setup

### 1️Clone the repository
```bash
git clone https://github.com/yourusername/manteef-stock-predictor.git
cd manteef-stock-predictor


model/xgb_stock_model.json

Deployment Notes
Railway Deployment

Push code to GitHub.

Connect Railway to your repository.

Set PORT environment variable to 8080 (Gunicorn default).

Ensure your requirements.txt has exact XGBoost version used in training.

Model path in app.py
