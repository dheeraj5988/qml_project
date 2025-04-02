import os
import numpy as np
import pandas as pd
import joblib
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZFeatureMap
from quantum_utils import get_quantum_instance

# Load the scaler
scaler = joblib.load("models/scaler.pkl")

# Load the trained models
models = {
    "RandomForest": joblib.load("models/RandomForest.pkl"),
    "SVM": joblib.load("models/SVM.pkl"),
    "LogisticRegression": joblib.load("models/LogisticRegression.pkl"),
    "QuantumVQC": joblib.load("models/qml_fraud_model.pkl"),
}

# Function to preprocess new data
def preprocess_input(data):
    """Preprocess input data using the saved scaler."""
    return scaler.transform(data)

# Function to make predictions using classical models
def predict_classical(data):
    """Make predictions using classical models."""
    predictions = {}
    for name, model in models.items():
        if name != "QuantumVQC":
            predictions[name] = model.predict(data)
    return predictions

# Function to make quantum predictions
def predict_quantum(data):
    """Make predictions using the quantum model."""
    feature_map = ZFeatureMap(feature_dimension=data.shape[1], reps=2)
    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=get_quantum_instance())

    # Transform input using the quantum kernel
    data_qk = quantum_kernel.evaluate(x_vec=data)

    # Predict using QuantumVQC
    return models["QuantumVQC"].predict(data_qk)

# Function to run inference
def run_inference(new_data):
    """Run inference for classical and quantum models."""
    new_data = preprocess_input(new_data)

    print("\n🔍 Running inference on new data...")

    classical_predictions = predict_classical(new_data)
    quantum_prediction = predict_quantum(new_data)

    # Display results
    for name, pred in classical_predictions.items():
        print(f"{name} Prediction: {pred}")

    print(f"QuantumVQC Prediction: {quantum_prediction}")

# Example new transaction data (Replace with actual inputs)
new_transaction = np.array([[5000, -1.3, 0.8, 2.1, -0.5, 1.4, -0.8, 3.5, 0.9]])  # Example 9 feature values

# Run inference
run_inference(new_transaction)
