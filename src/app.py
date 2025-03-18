from flask import Flask, render_template, request, jsonify
import pandas as pd
from predict_fraud import predict_fraud

app = Flask(__name__)

# Home Page - Transaction Input Form
@app.route('/')
def home():
    return render_template('index.html')

# API Endpoint for Fraud Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data from form
        transaction_data = {key: float(request.form[key]) for key in request.form.keys()}
        
        # Run fraud detection model
        result = predict_fraud(transaction_data)
        
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
