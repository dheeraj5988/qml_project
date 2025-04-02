from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZFeatureMap
from qiskit import BasicAer

# Initialize FastAPI
app = FastAPI(title="Quantum Fraud Detection API", version="1.0")

# Load trained models
scaler = joblib.load("models/scaler.pkl")
models = {
    "RandomForest": joblib.load("models/RandomForest.pkl"),
    "SVM": joblib.load("models/SVM.pkl"),
    "LogisticRegression": joblib.load("models/LogisticRegression.pkl"),
    "QuantumSVM": joblib.load("models/QuantumSVM.pkl")
}

# Define Quantum Kernel (same as in training)
feature_map = ZFeatureMap(feature_dimension=30, reps=2)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=BasicAer.get_backend("statevector_simulator"))

# Request body model
class TransactionData(BaseModel):
    features: list[float]
    model: str  # Model choice: "RandomForest", "SVM", "LogisticRegression", "QuantumSVM"

# Root route
@app.get("/")
def home():
    return {"message": "Quantum Fraud Detection API is Running!"}

# Prediction route
@app.post("/predict")
def predict_fraud(data: TransactionData):
    try:
        # Check model validity
        if data.model not in models:
            raise HTTPException(status_code=400, detail="Invalid model name. Choose from: RandomForest, SVM, LogisticRegression, QuantumSVM.")

        # Preprocess input
        input_data = np.array(data.features).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Handle Quantum SVM separately
        if data.model == "QuantumSVM":
            input_data = quantum_kernel.evaluate(x_vec=input_data)

        # Make prediction
        prediction = models[data.model].predict(input_data)
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"

        return {"model": data.model, "prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API using: `uvicorn api:app --reload`
