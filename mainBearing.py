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

# Set dataset path and file name for the bearing dataset
dataset_path = "/home/bilz/Datasets/q/"
dataset_name = "Bearing.csv"

# Use a non-interactive backend to avoid "Wayland" issues
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering plots without a display

# Preprocessing function for bearing dataset
def preprocess_data(data, window_size=20):
    print(f"Preprocessing bearing data with window size {window_size}...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    
    # Process the bearing data
    if 'Bearing 1' in data.columns:
        print("Processing bearing data from 'Bearing 1'...")
        data['Bearing 1'] = scaler.fit_transform(data['Bearing 1'].values.reshape(-1, 1))
        X = np.array([data['Bearing 1'].values[i:i + window_size] for i in range(len(data) - window_size)])
        y_true = np.array([1 if val > 0.8 else 0 for val in data['Bearing 1'][window_size:]])
    else:
        raise ValueError("Unknown data format in dataset.")
    
    print("Preprocessing complete.")
    return X, y_true



# Function to load the bearing dataset from the local CSV file
def load_datasets():
    print("Loading bearing dataset from local CSV file...")
    datasets = {}

    # Combine the path and name into a full file path
    file_path = dataset_path + dataset_name
    data_bearing = pd.read_csv(file_path)
    
    # Check column names
    print(f"Columns in the dataset: {data_bearing.columns}")
    
    if data_bearing.empty:
        raise ValueError("Failed to load dataset.")
    
    datasets['Bearing'] = data_bearing
    print("Loaded bearing dataset.")
    visualize_dataset(data_bearing, 'Bearing')
    
    return datasets


# Function to visualize a dataset
def visualize_dataset(data, dataset_name):
    plt.figure(figsize=(14, 6))
    
    # Replace 'Bearing1' with the correct column name
    if 'Vibration' in data.columns:
        plt.plot(data.index, data['Vibration'], label='Vibration')
    
    plt.title(f"{dataset_name} - Data Visualization")
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    
    # Add label if it exists, else skip legend
    if 'Vibration' in data.columns:
        plt.legend()
    
    plt.savefig(f"{dataset_name}_data_visualization.png")  # Save plot as PNG


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

# Main function to run the comparison
def run_comparison(datasets, window_size=20):
    results = {}

    for name, data in datasets.items():
        print(f"\nProcessing dataset: {name}")
        X, y_true = preprocess_data(data, window_size)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)
        
        # Quantum anomaly detection
        anomaly_scores = []
        print(f"Running quantum anomaly detection for {name}...")
        for sample_X in tqdm(X_test, desc="Quantum Anomaly Detection"):
            initial_params = np.random.rand(len(sample_X)) * 2 * np.pi
            optimized_params = optimize_params(sample_X, initial_params)
            score = objective_function(optimized_params, sample_X)
            anomaly_scores.append(score)

        # Smart threshold based on IQR
        smart_threshold = calculate_smart_threshold(anomaly_scores)
        y_pred_quantum = [1 if score > smart_threshold else 0 for score in anomaly_scores]

        # Quantum method accuracy
        quantum_accuracy = accuracy_score(y_test[:len(y_pred_quantum)], y_pred_quantum)
        print(f"Quantum Method Accuracy for {name}: {quantum_accuracy:.2f}")

        # Classical anomaly detection
        classical_accuracies = classical_methods(X_train, y_train, X_test, y_test)

        # Store results for visualization
        results[name] = {
            'quantum_accuracy': quantum_accuracy,
            'classical_accuracies': classical_accuracies,
            'y_test': y_test,
            'y_pred_quantum': y_pred_quantum,
            'anomaly_scores': anomaly_scores,
            'loss_history': loss_history.copy(),
            'threshold': smart_threshold
        }

        # Clear loss history for the next dataset
        loss_history.clear()

    return results

# Visualize results
def visualize_results(datasets, results):
    for name, data in datasets.items():
        print(f"\nVisualizing results for dataset: {name}")
        res = results[name]

        # Update this to match the actual column names in your dataset
        if 'Bearing 1' in data.columns:
            value_column = 'Bearing 1'
        elif 'Bearing 2' in data.columns:
            value_column = 'Bearing 2'
        elif 'Bearing 3' in data.columns:
            value_column = 'Bearing 3'
        elif 'Bearing 4' in data.columns:
            value_column = 'Bearing 4'
        else:
            raise ValueError("No valid bearing column found in dataset.")

        test_indices = data.index[-len(res['y_test']):]

        # Data Visualization of Normal and Anomaly
        plt.figure(figsize=(14, 6))
        plt.plot(data.index, data[value_column], label=value_column)
        y_test_array = np.array(res['y_test'])
        y_pred_quantum_array = np.array(res['y_pred_quantum'])

        plt.scatter(test_indices[y_test_array == 1], data[value_column].iloc[-len(y_test_array):].to_numpy()[y_test_array == 1], color='red', label='True Anomalies')
        plt.scatter(test_indices[y_pred_quantum_array == 1], data[value_column].iloc[-len(y_pred_quantum_array):].to_numpy()[y_pred_quantum_array == 1], color='blue', label='Detected Anomalies (Quantum)', marker='x')
        plt.xlabel('Timestamp')
        plt.ylabel('Normalized Value')
        plt.title(f'{name} - Time Series Data with Anomalies')
        plt.legend()
        plt.savefig(f"{name}_data_visualization.png")
        plt.show()

        # Training Loss (0-100 iterations)
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(res['loss_history'][:100])), res['loss_history'][:100], label='Training Loss (0-100 iterations)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{name} - Quantum Training Loss over 0-100 Iterations')
        plt.legend()
        plt.savefig(f"{name}_training_loss.png")
        plt.show()

        # Anomaly Score
        plt.figure(figsize=(14, 6))
        plt.plot(test_indices, res['anomaly_scores'], label='Anomaly Score')
        plt.axhline(y=res['threshold'], color='r', linestyle='--', label='Threshold')
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        plt.title(f'{name} - Anomaly Scores with Threshold')
        plt.legend()
        plt.savefig(f"{name}_anomaly_scores.png")
        plt.show()


# Main script execution
def main():
    datasets = load_datasets()
    results = run_comparison(datasets, window_size=4)
    visualize_results(datasets, results)

if __name__ == "__main__":
    main()
