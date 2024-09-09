import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from qiskit import Aer, QuantumCircuit, execute
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Set the NAB dataset path
dataset_path = "/home/bilz/datasets/q/NAB/data/"  # Path to the NAB dataset
results_path = "/home/bilz/results/q/NAB/CE/"  # Directory to save results

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering plots without a display

# Preprocessing function for NAB dataset (handles both labeled and unlabeled data)
def preprocess_data(data, window_size=20):
    print(f"Preprocessing NAB data with window size {window_size}...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))

    # Process the NAB dataset, usually based on 'value' column for anomaly detection
    if 'value' in data.columns:
        print("Processing NAB data from 'value' column...")
        data['value'] = scaler.fit_transform(data['value'].values.reshape(-1, 1))
        X = np.array([data['value'].values[i:i + window_size] for i in range(len(data) - window_size)])
    else:
        raise ValueError("The dataset does not contain a 'value' column.")
    
    # Check if there's a label column (e.g., 'label', 'anomaly', etc.)
    if 'label' in data.columns:
        print("Using 'label' column for ground truth labels...")
        y_true = np.array([1 if label == 1 else 0 for label in data['label'][window_size:]])  # 1 for anomaly, 0 for normal
    elif 'anomaly' in data.columns:
        print("Using 'anomaly' column for ground truth labels...")
        y_true = np.array([1 if label == 1 else 0 for label in data['anomaly'][window_size:]])
    else:
        # If no label is found, default to an unsupervised approach
        print("No label column found. Treating as unsupervised dataset.")
        y_true = None  # No ground truth labels available
    
    print("Preprocessing complete.")
    return X, y_true

# Function to load NAB datasets from the local CSV files
def load_nab_datasets():
    print("Loading NAB datasets from local CSV files...")
    datasets = {}

    # Loop over all CSV files in the dataset path
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Loading dataset: {file}")
                data_nab = pd.read_csv(file_path)
                
                # Check column names
                print(f"Columns in the dataset: {data_nab.columns}")
                
                if data_nab.empty:
                    raise ValueError(f"Failed to load dataset {file}")
                
                datasets[file] = data_nab
    
    print(f"Loaded {len(datasets)} NAB datasets.")
    return datasets

# Function to save results to a text file (appending results one by one)
def save_results_to_file(dataset_name, result, file_path):
    with open(file_path, 'a') as f:
        f.write(f"Dataset: {dataset_name}\n")
        
        if result['quantum_accuracy'] is not None:
            f.write(f"Quantum Accuracy: {result['quantum_accuracy']:.4f}\n")
        else:
            f.write(f"Quantum Accuracy: No ground truth (unsupervised)\n")
        
        for model_name, acc in result['classical_accuracies'].items():
            f.write(f"{model_name} Accuracy: {acc:.4f}\n")
        
        f.write("\n")
    print(f"Saved results for {dataset_name} to {file_path}")

# Quantum encoding function
def encode_data(X):
    n_qubits = len(X)
    qc = QuantumCircuit(n_qubits)
    for i, x in enumerate(X):
        qc.ry(x, i)
    return qc

# Variational circuit function
def variational_circuit(params, n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i, param in enumerate(params):
        qc.rx(param, i)
        qc.rz(param, i)
    return qc

# Objective function for quantum method
loss_history = []
def objective_function(params, X):
    n_qubits = len(X)
    qc = encode_data(X)
    var_circuit = variational_circuit(params, n_qubits)
    qc.compose(var_circuit, inplace=True)
    
    backend = Aer.get_backend('statevector_simulator')
    result = execute(qc, backend).result()
    statevector = result.get_statevector(qc)
    
    normal_state = np.zeros_like(statevector)
    normal_state[0] = 1
    loss = 1 - np.abs(np.dot(statevector.conj(), normal_state)) ** 2
    
    loss_history.append(loss)
    return loss

# Optimize quantum circuit parameters
def optimize_params(X, initial_params):
    print("Optimizing quantum circuit parameters...")
    result = minimize(objective_function, initial_params, args=(X,), method='COBYLA')
    print("Optimization complete.")
    return result.x

# Classical anomaly detection models
def classical_methods(X_train, y_train, X_test, y_test):
    models = {
        'IsolationForest': IsolationForest(contamination=0.1),
        'OneClassSVM': OneClassSVM(nu=0.1),
        'LocalOutlierFactor': LocalOutlierFactor(novelty=True),
    }
    
    results = {}
    for name, model in models.items():
        print(f"Running {name}...")
        if name in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor']:
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)  # Convert 1 -> 0 (normal), -1 -> 1 (anomaly)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy:.2f}")
        results[name] = accuracy
    
    return results

# Smart threshold using IQR method
def calculate_smart_threshold(scores):
    q25, q75 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    return upper_bound

# Main function to run the comparison on all NAB datasets
def run_comparison(datasets, window_size=20):
    results = {}
    total_datasets = len(datasets)
    processed_datasets = 0

    for dataset_name, data in datasets.items():
        processed_datasets += 1
        print(f"\nProcessing dataset: {dataset_name} ({processed_datasets}/{total_datasets} - {processed_datasets/total_datasets*100:.2f}%)")
        
        X, y_true = preprocess_data(data, window_size)

        # Split the data into training and testing
        if y_true is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
        else:
            # Handle the case where there are no labels (unsupervised)
            X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
            y_train, y_test = None, None

        # Quantum anomaly detection
        anomaly_scores = []
        print(f"Running quantum anomaly detection for {dataset_name}...")
        for sample_X in tqdm(X_test, desc=f"Quantum Anomaly Detection for {dataset_name}"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores.append(score)

        # Smart threshold based on IQR (for unsupervised approach, this threshold will be used directly)
        smart_threshold = calculate_smart_threshold(anomaly_scores)
        y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]

        # Quantum method accuracy (if y_test is available)
        if y_test is not None:
            quantum_accuracy = accuracy_score(y_test[:len(y_pred_quantum)], y_pred_quantum)
            print(f"Quantum Method Accuracy for {dataset_name}: {quantum_accuracy:.2f}")
        else:
            quantum_accuracy = None  # No ground truth to calculate accuracy
        
        # Classical anomaly detection (only if labels are available)
        if y_test is not None:
            classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)
        else:
            classical_accuracies = {}

        # Store results for visualization and saving
        results[dataset_name] = {
            'quantum_accuracy': quantum_accuracy,
            'classical_accuracies': classical_accuracies,
            'y_test': y_test,
            'y_pred_quantum': y_pred_quantum,
            'anomaly_scores': anomaly_scores,
            'loss_history': loss_history.copy(),
            'threshold': smart_threshold
        }

        # Save individual dataset results to the file after processing
        save_results_to_file(dataset_name, results[dataset_name], results_path + "nab_results.txt")

        # Clear loss history for the next dataset
        loss_history.clear()

    return results

# Main script execution
def main():
    datasets = load_nab_datasets()
    results = run_comparison(datasets, window_size=4)
    print(f"Results saved in nab_results.txt")

if __name__ == "__main__":
    main()
