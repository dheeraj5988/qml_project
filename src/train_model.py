import os
import numpy as np
import pandas as pd
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from qiskit import Aer
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from quantum_utils import get_quantum_instance

# Ensure Kaggle API is configured
os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")

# Load dataset dynamically from Kaggle
print("📥 Loading dataset from Kaggle...")
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv"
)

print("✅ Dataset Loaded Successfully!")
print(df.head())

# Select features and target
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "models/scaler.pkl")

# Train Classical Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"models/{name}.pkl")

# Quantum Feature Map and Kernel
print("\n⚛️ Implementing Quantum Model...")

feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
variational_circuit = RealAmplitudes(num_qubits=X_train.shape[1], reps=2)

quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=get_quantum_instance())

# Train Quantum Classifier
vqc = VQC(quantum_kernel=quantum_kernel, var_form=variational_circuit)
vqc.fit(X_train, y_train)

# Evaluate Quantum Model
y_pred_vqc = vqc.predict(X_test)
print("\nQuantum VQC Model Accuracy:", accuracy_score(y_test, y_pred_vqc))
print(classification_report(y_test, y_pred_vqc))

# Save the Quantum Model
joblib.dump(vqc, "models/qml_fraud_model.pkl")

print("\n✅ Training complete! All models saved successfully.")
