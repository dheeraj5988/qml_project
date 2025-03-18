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
from qiskit import BasicAer
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZFeatureMap

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
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target (0 = Normal, 1 = Fraud)

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

# Quantum Kernel Method (Qiskit)
print("\n⚛️ Implementing Quantum Kernel Method...")
feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=BasicAer.get_backend("statevector_simulator"))

# Train Quantum Model
X_train_qk = quantum_kernel.evaluate(x_vec=X_train)
X_test_qk = quantum_kernel.evaluate(x_vec=X_test)

qsvc = SVC(kernel="precomputed")
qsvc.fit(X_train_qk, y_train)

# Evaluate Quantum Model
y_pred_qsvc = qsvc.predict(X_test_qk)
print("\nQuantum SVM Model Accuracy:", accuracy_score(y_test, y_pred_qsvc))
print(classification_report(y_test, y_pred_qsvc))

# Save the Quantum Model
joblib.dump(qsvc, "models/QuantumSVM.pkl")

print("\n✅ Training complete! Models saved successfully.")
