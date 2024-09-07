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

# Set dataset path and file name for the WAADM dataset
dataset_path = "/home/bilz/Datasets/q/"
dataset_name = "WAADM.csv"  # Updated to WAADM dataset
results_path = "/home/bilz/results/q/WAADM/CE/"  # Directory to save results

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering plots without a display

# Preprocessing function for WAADM dataset
def preprocess_data(mac_data, window_size=20):
    print(f"Preprocessing data with window size {window_size}...")

    if mac_data.empty:
        raise ValueError("Dataset is empty.")
    
    # Ensure signal_strength is present and can be processed
    if 'signal_strength' not in mac_data.columns or 'anomaly' not in mac_data.columns:
        raise ValueError("Expected 'signal_strength' and 'anomaly' columns not found in dataset.")
    
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    mac_data['signal_strength'] = scaler.fit_transform(mac_data['signal_strength'].values.reshape(-1, 1))

    # Prepare rolling window for signal strength data
    X = np.array([mac_data['signal_strength'].values[i:i + window_size] for i in range(len(mac_data) - window_size)])
    y_true = np.array(mac_data['anomaly'][window_size:])

    print("Preprocessing complete.")
    return X, y_true

# Function to load the WAADM dataset from the local CSV file
def load_datasets():
    print("Loading WAADM dataset from local CSV file...")
    file_path = dataset_path + dataset_name
    data_waadm = pd.read_csv(file_path)

    if data_waadm.empty:
        raise ValueError("Failed to load dataset.")
    
    print(f"Columns in the dataset: {data_waadm.columns}")
    return data_waadm

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

# Run comparison for each MAC address
def run_comparison_for_mac(data, mac_column, window_size=20):
    print(f"Processing MAC address: {mac_column}")
    
    mac_data = data[['signal_strength', 'anomaly']][data['mac_address'] == mac_column]  # Filter for the current MAC address
    X, y_true = preprocess_data(mac_data, window_size)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
    
    # Quantum anomaly detection
    anomaly_scores = []
    print(f"Running quantum anomaly detection for {mac_column}...")
    for sample_X in tqdm(X_test, desc=f"Quantum Anomaly Detection for {mac_column}"):
        initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
        optimized_params = optimize_params(sample_X, initial_params)
        score = objective_function(optimized_params, sample_X)
        anomaly_scores.append(score)

    # Smart threshold based on IQR
    smart_threshold = calculate_smart_threshold(anomaly_scores)
    y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]

    # Quantum method accuracy
    quantum_accuracy = accuracy_score(y_test[:len(y_pred_quantum)], y_pred_quantum)
    print(f"Quantum Method Accuracy for {mac_column}: {quantum_accuracy:.2f}")

    # Classical anomaly detection
    classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)

    return {
        'quantum_accuracy': quantum_accuracy,
        'classical_accuracies': classical_accuracies
    }

# Main function to run the comparison across all MAC addresses and save results
def main():
    data = load_datasets()
    
    mac_columns = data['mac_address'].unique()  # Get unique MAC addresses
    results = []

    # Open a text file to save the accuracy results
    with open(results_path + 'accuracy_results.txt', 'w') as f:
        for i, mac_column in enumerate(mac_columns):
            res = run_comparison_for_mac(data, mac_column, window_size=4)
            results.append(res)

            # Save the result in a text file
            f.write(f"MAC {i + 1} - {mac_column}\n")
            f.write(f"Quantum Accuracy: {res['quantum_accuracy']:.2f}\n")
            for model, acc in res['classical_accuracies'].items():
                f.write(f"{model} Accuracy: {acc:.2f}\n")
            f.write("\n")

    # Generate comparison plots after processing all MAC addresses
    quantum_accuracies = [res['quantum_accuracy'] for res in results]
    classical_accuracies = [list(res['classical_accuracies'].values()) for res in results]
    classical_means = np.mean(classical_accuracies, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(quantum_accuracies)), quantum_accuracies, alpha=0.6, label='Quantum')
    plt.bar(range(len(classical_means)), classical_means, alpha=0.6, label='Classical (Mean)')
    plt.title("Quantum vs Classical Mean Accuracy Across MAC Addresses")
    plt.xlabel('MAC Address Index')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(results_path + 'comparison_plot.png')
    print(f"Saved accuracy comparison plot at {results_path + 'comparison_plot.png'}")

if __name__ == "__main__":
    main()
