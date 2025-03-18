import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap

# Load pre-trained models
print("📥 Loading trained models...")
scaler = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/RandomForest.pkl")
svm_model = joblib.load("models/SVM.pkl")
lr_model = joblib.load("models/LogisticRegression.pkl")
qsvc_model = joblib.load("models/QuantumSVM.pkl")

# Quantum Kernel Setup (Same as training)
feature_map = ZFeatureMap(feature_dimension=30, reps=2)  # Assuming dataset has 30 features
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=BasicAer.get_backend("statevector_simulator"))

print("✅ Models Loaded Successfully!")

# Define Fraud Detection Function
def predict_fraud(transaction):
    """
    Predicts whether a transaction is fraud or not.
    Input: transaction (dictionary of features)
    Output: Fraud (1) or Not Fraud (0)
    """
    # Convert input transaction to DataFrame
    input_df = pd.DataFrame([transaction])
    
    # Standardize input
    input_scaled = scaler.transform(input_df)
    
    # Classical Model Predictions
    rf_pred = rf_model.predict_proba(input_scaled)[:, 1]  # Probability of fraud
    svm_pred = svm_model.predict_proba(input_scaled)[:, 1]
    lr_pred = lr_model.predict_proba(input_scaled)[:, 1]
    
    # Quantum Model Prediction
    input_qk = quantum_kernel.evaluate(x_vec=input_scaled)
    qsvc_pred = qsvc_model.predict(input_qk)
    
    # Calculate Final Fraud Score (Weighted Average)
    fraud_score = (0.3 * rf_pred + 0.3 * svm_pred + 0.2 * lr_pred + 0.2 * qsvc_pred)
    
    # Decision Logic
    if fraud_score > 0.75:
        return "❌ Transaction Declined: Fraud Detected (Risk Score: {:.2f})".format(fraud_score[0])
    elif fraud_score > 0.50:
        return "⚠️ Transaction Flagged: Requires Manual Review (Risk Score: {:.2f})".format(fraud_score[0])
    else:
        return "✅ Transaction Approved: No Fraud Detected (Risk Score: {:.2f})".format(fraud_score[0])

# Example Test Transaction
test_transaction = {
    "V1": -1.23, "V2": 2.34, "V3": -0.56, "V4": 1.12, "V5": -2.67, "V6": 0.88, "V7": -1.22, "V8": 1.34, "V9": -2.11, "V10": 1.99,
    "V11": -1.67, "V12": 2.45, "V13": -0.89, "V14": 1.78, "V15": -2.55, "V16": 0.91, "V17": -1.45, "V18": 1.32, "V19": -2.44, "V20": 1.87,
    "V21": -1.99, "V22": 2.55, "V23": -0.67, "V24": 1.49, "V25": -2.21, "V26": 0.79, "V27": -1.33, "V28": 1.88, "Amount": 250.00
}

# Run Prediction
print("\n🔍 Testing Sample Transaction...")
result = predict_fraud(test_transaction)
print(result)

