import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle
import os
import json

class MLPNeuralNetwork:
    def __init__(self, input_size: int, output_size: int, params_file=None):
        self.model = 'MLP - Multi-Layer Perceptron'
        self.type = 'Regression'
        self.purpose = 'Housing Prices Prediction'
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.best_params = {}
        self.hidden_layers = []  # Ensure hidden layers are initialized
        self.output = None
        
        # Check for existing parameter files
        if params_file:
            # User specified a parameter file
            if self.load_parameters(params_file):
                print(f"Loaded parameters from {params_file}")
        else:
            # Try to load from default locations
            possible_files = ["best_params.pkl", "best_params.json", "best_params.txt"]
            loaded = False
            for file in possible_files:
                if os.path.exists(file):
                    if self.load_parameters(file):
                        print(f"Automatically loaded parameters from {file}")
                        loaded = True
                        break
            
            if not loaded:
                print("No parameter file found. You will need to run grid_search before fitting the model.")
                print("Or load parameters manually using load_parameters(filename).")
        
    def get_info(self):
        """Display comprehensive information about the neural network model."""
        print(f"Model: {self.model}")
        print(f"Type: {self.type}")
        print(f"Purpose: {self.purpose}")
        
        print("\nArchitecture:")
        print(f"  Input size: {self.input_size}")
        print(f"  Output size: {self.output_size}")
        
        if self.best_params:
            print(f"  Hidden layers: {self.best_params.get('layer', 'Not specified')}")
            print(f"  Neurons per layer: {self.best_params.get('npl', 'Not specified')}")
            
            print("\nHyperparameters:")
            print(f"  Learning rate (alpha): {self.best_params.get('alpha', 'Not specified')}")
            print(f"  Regularization (lambda): {self.best_params.get('lambda', 'Not specified')}")
            print(f"  Convergence threshold (epsilon): {self.best_params.get('epsilon', 'Not specified')}")
            
            if 'mean_mae' in self.best_params:
                print(f"\nPerformance: MAE = {self.best_params['mean_mae']:.6f}")
                
            if 'timestamp' in self.best_params:
                print(f"\nLast trained: {self.best_params['timestamp']}")
        else:
            print("\nNote: No parameters loaded. Run grid_search or load_parameters first.")
            
        # Show status of model readiness
        if hasattr(self, 'weights') and self.weights:
            print("\nStatus: Model is initialized with weights")
        else:
            print("\nStatus: Model needs initialization before training/prediction")
        
        print()
        
    def generate_weights_values(self, x, y):
        """Generate weight values with careful initialization to avoid NaN/Inf values"""
        try:
            # Start with small constant values as a fallback
            weights = np.ones((x, y)) * 0.01
            
            # Try He initialization (good for ReLU but also works for sigmoid)
            temp_weights = np.random.randn(x, y) * np.sqrt(2.0 / x)
            if not np.isnan(temp_weights).any() and not np.isinf(temp_weights).any():
                weights = temp_weights
                
            return weights
        except Exception as e:
            print(f"Error in weight initialization: {e}. Using safe fallback.")
            return np.ones((x, y)) * 0.01
        
    def generate_weights_matrix(self, neurons_per_layer: int, hidden_sizes: int):
        """Generate weight matrices for all layers with safety checks"""
        try:
            # Generate initial weights
            weight_matrix = []
            
            # Input to first hidden layer
            first_layer = self.generate_weights_values(self.input_size + 1, neurons_per_layer)
            weight_matrix.append(first_layer)
            
            # Hidden to hidden layers
            for _ in range(1, hidden_sizes):
                hidden_layer = self.generate_weights_values(neurons_per_layer + 1, neurons_per_layer)
                weight_matrix.append(hidden_layer)
                
            # Last hidden to output layer
            output_layer = self.generate_weights_values(neurons_per_layer + 1, self.output_size)
            weight_matrix.append(output_layer)
            
            # Safety check - verify no NaN or Inf values
            for i, w in enumerate(weight_matrix):
                if np.isnan(w).any() or np.isinf(w).any():
                    print(f"Safety check: Replacing invalid weights in layer {i}")
                    if i == 0:
                        weight_matrix[i] = np.ones((self.input_size + 1, neurons_per_layer)) * 0.01
                    elif i < len(weight_matrix) - 1:
                        weight_matrix[i] = np.ones((neurons_per_layer + 1, neurons_per_layer)) * 0.01
                    else:
                        weight_matrix[i] = np.ones((neurons_per_layer + 1, self.output_size)) * 0.01
            
            return weight_matrix
        except Exception as e:
            print(f"Error in weight matrix generation: {e}. Using safe fallback.")
            # Generate safe fallback weights
            weight_matrix = []
            weight_matrix.append(np.ones((self.input_size + 1, neurons_per_layer)) * 0.01)
            for _ in range(1, hidden_sizes):
                weight_matrix.append(np.ones((neurons_per_layer + 1, neurons_per_layer)) * 0.01)
            weight_matrix.append(np.ones((neurons_per_layer + 1, self.output_size)) * 0.01)
            return weight_matrix
    
    def generate_delta(self):
        """Generate delta arrays for backpropagation"""
        return [np.zeros_like(w) for w in self.weights]
    
    def generate_gradient(self):
        """Generate gradient arrays for weight updates"""
        return [np.zeros_like(w) for w in self.weights]
    
    def sigmoid(self, z):
        """Apply sigmoid activation with safety clipping to prevent overflow"""
        z = np.clip(z, -50, 50)  # More conservative clipping to prevent issues
        return 1.0 / (1.0 + np.exp(-z))
    
    def initialize_hidden_layers(self, batch_size, hidden_sizes, neurons_per_layer):
        """Initialize hidden layers with correct dimensions"""
        self.hidden_layers = []
        for _ in range(hidden_sizes):
            self.hidden_layers.append(np.zeros((batch_size, neurons_per_layer)))
    
    def forward_propagation(self, data):
        """Forward pass through the network with robust error checks"""
        # Validate input data
        if data.size == 0:
            raise ValueError("Input data is empty")
        
        # Check for NaN or Inf in input
        if np.isnan(data).any() or np.isinf(data).any():
            # Handle NaN/Inf by replacing with zeros (safe approach)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: NaN or Inf values in input data replaced with zeros")
        
        # Add bias term
        a = np.insert(data, 0, 1, axis=1)
        
        # Initialize or update hidden layers for current batch size
        batch_size = data.shape[0]
        for i in range(len(self.hidden_layers)):
            if self.hidden_layers[i] is None or self.hidden_layers[i].shape[0] != batch_size:
                neurons_per_layer = self.weights[i].shape[1]
                self.hidden_layers[i] = np.zeros((batch_size, neurons_per_layer))
        
        # Forward pass through each layer
        for i in range(len(self.hidden_layers)):
            # Validate matrix dimensions
            if a.shape[1] != self.weights[i].shape[0]:
                raise ValueError(f"Matrix dimensions mismatch at layer {i}: a.shape={a.shape}, weights.shape={self.weights[i].shape}")
            
            # Compute layer output with safety checks
            z = a.dot(self.weights[i])
            
            # Check for and handle NaN values
            if np.isnan(z).any():
                z = np.nan_to_num(z, nan=0.0)
                print(f"Warning: NaN values in layer {i} output replaced with zeros")
            
            # Apply activation and add bias for next layer
            self.hidden_layers[i] = self.sigmoid(z)
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)
        
        # Output layer
        z = a.dot(self.weights[-1])
        
        # Final safety check
        if np.isnan(z).any():
            z = np.nan_to_num(z, nan=0.0)
            print("Warning: NaN values in final output replaced with zeros")
        
        self.output = z
        return self.output
    
    def backward_propagation(self, data, actual):
        """Backward pass with gradient computation and safety checks"""
        # Reshape target values if needed
        if len(actual.shape) == 1:
            actual = actual.reshape(-1, 1)
            
        # Compute initial error
        delta = self.output - actual
        batch_size = data.shape[0]

        # Store output layer delta
        self.delta[-1] = delta
        
        # Backpropagate error through hidden layers
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            # Compute sigmoid derivative
            sigmoid_derivative = self.hidden_layers[i] * (1 - self.hidden_layers[i])
            
            # Compute delta for current layer
            delta = delta.dot(self.weights[i + 1].T)[:, 1:] * sigmoid_derivative
            
            # Check for NaN/Inf values (can occur due to numerical instability)
            if np.isnan(delta).any() or np.isinf(delta).any():
                delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: NaN/Inf values in delta at layer {i} replaced with zeros")
                
            self.delta[i] = delta
        
        # Compute gradients for weight updates
        # Input layer gradient
        a = np.insert(data, 0, 1, axis=1)
        self.gradient[0] = a.T.dot(self.delta[0]) / batch_size
        
        # Hidden layer gradients
        for i in range(len(self.weights) - 1):
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)
            self.gradient[i + 1] = a.T.dot(self.delta[i + 1]) / batch_size
        
        # Clip gradients to prevent exploding gradients
        for i in range(len(self.gradient)):
            self.gradient[i] = np.clip(self.gradient[i], -1.0, 1.0)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, alpha: float, epsilon: float, lamb: float, max_epochs: int, batch_size: int):
        """Train the neural network with robust error handling"""
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data is empty")
            
        previous_cost = float('inf')
        num_samples = X_train.shape[0]
        
        # Ensure y_train is properly shaped
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        for epoch in range(max_epochs):
            # Shuffle data for each epoch
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Forward and backward passes
                self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch)

                # Update weights with regularization
                for j in range(len(self.weights)):
                    weight_update = alpha * (self.gradient[j] + (lamb / num_samples) * self.weights[j])
                    # Clip update to prevent large changes
                    weight_update = np.clip(weight_update, -0.1, 0.1)
                    self.weights[j] -= weight_update

            # Evaluate on full training set
            self.forward_propagation(X_train)
            
            # Check for NaN in output or weights
            if np.isnan(self.output).any():
                print("Warning: NaN detected in output. Reinitializing weights.")
                # Could add recovery logic here
                raise ValueError("Training failed due to NaN values in output")
            
            for i, w in enumerate(self.weights):
                if np.isnan(w).any():
                    print(f"Warning: NaN detected in weights[{i}]. Reinitializing weights.")
                    raise ValueError("Training failed due to NaN values in weights")
            
            # Compute cost with regularization
            cost = mean_absolute_error(self.output, y_train) + (lamb / (2 * num_samples)) * sum(np.sum(w ** 2) for w in self.weights)
            
            # Check for convergence
            if abs(cost - previous_cost) <= epsilon:
                print(f"Converged at epoch {epoch}")
                return cost
            
            previous_cost = cost

        print(f"Training stopped after reaching max_epochs ({max_epochs}).")
        return previous_cost
    
    def test(self, X_test: np.ndarray, y_test: np.ndarray):
        """Test the neural network on validation data"""
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            raise ValueError("Test data is empty")
            
        # Forward pass on test data
        self.forward_propagation(X_test)
        
        # Reshape y_test if necessary
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            
        # Return mean absolute error
        return mean_absolute_error(self.output, y_test)
    
    def grid_search(self, features: np.ndarray, labels: np.ndarray, params: dict, k_fold_index: object):
        """Perform grid search with cross-validation for hyperparameter tuning"""
        if features.shape[0] == 0 or labels.shape[0] == 0:
            raise ValueError("Features or labels data is empty")
            
        alphas, lambdas, epsilons = params['alphas'], params['lambdas'], params['epsilons']
        hidden_sizes, neurons_per_layer = params['hidden_sizes'], params['neurons_per_layer']
        
        best_mae, best_params = float('inf'), {}
        
        # Early stopping threshold - if MAE is worse than this after early_stopping_folds,
        # we'll skip the rest of the folds for this parameter set
        early_stopping_threshold = params.get('early_stopping_threshold', 80000)
        early_stopping_folds = params.get('early_stopping_folds', 5)  # Check after this many folds
        
        # Show early stopping settings if provided
        if 'early_stopping_threshold' in params:
            print(f"Early stopping enabled: Will skip parameter sets with MAE > {early_stopping_threshold} after {early_stopping_folds} folds")

        # Grid search through all parameter combinations
        for alpha in alphas:
            for lamb in lambdas:
                for epsilon in epsilons:
                    for layer in hidden_sizes:
                        for npl in neurons_per_layer:
                            print(f"Training with alpha={alpha}, lambda={lamb}, epsilon={epsilon}, layer={layer}, npl={npl}")
                            print()
                            
                            # Initialize network for this parameter set
                            self.hidden_layers = [None] * layer
                            self.weights = self.generate_weights_matrix(npl, layer)
                            self.delta = self.generate_delta()
                            self.gradient = self.generate_gradient()
                            mean_abs_errors = []
                            
                            # Check for NaN/Inf in initial weights
                            valid_weights = True
                            for i, w in enumerate(self.weights):
                                if np.isnan(w).any() or np.isinf(w).any():
                                    print(f"Warning: Invalid weights in layer {i}. Skipping this parameter set.")
                                    valid_weights = False
                                    break
                                    
                            if not valid_weights:
                                continue
                                
                            print('Weights are valid.')
                            print()
                            print('Starting cross-validation...')
                            print()
                            
                            # Cross-validation
                            try:
                                fold_num = 0
                                skip_current_param_set = False
                                
                                for train_index, test_index in k_fold_index:
                                    fold_num += 1
                                    print(f"Processing fold {fold_num}...")
                                    
                                    # Split data for this fold
                                    X_train, X_test = features[train_index], features[test_index]
                                    y_train, y_test = labels[train_index], labels[test_index]
                                    
                                    # Train model
                                    try:
                                        self.train(X_train, y_train, alpha, epsilon, lamb, max_epochs=1000, batch_size=64)
                                        mae = self.test(X_test, y_test)
                                        
                                        if np.isnan(mae):
                                            print(f"Warning: NaN MAE in fold {fold_num}. Skipping this fold.")
                                            continue
                                            
                                        print(f"MAE for fold {fold_num}: {mae}")
                                        mean_abs_errors.append(mae)
                                        
                                        # Early stopping check - if we have enough folds to evaluate
                                        if fold_num >= early_stopping_folds:
                                            current_mean_mae = sum(mean_abs_errors) / len(mean_abs_errors)
                                            # If current mean MAE is much worse than best so far or threshold, skip further folds
                                            if (best_mae < float('inf') and current_mean_mae > best_mae * 1.5) or \
                                               (current_mean_mae > early_stopping_threshold):
                                                print(f"Early stopping: Current MAE ({current_mean_mae:.4f}) is significantly worse than best ({best_mae:.4f})")
                                                print("Skipping remaining folds for this parameter set.")
                                                skip_current_param_set = True
                                                break
                                    except Exception as e:
                                        print(f"Error in fold {fold_num}: {e}. Skipping this fold.")
                                        print()
                                        continue
                                    
                                    if skip_current_param_set:
                                        break
                                
                                # Calculate mean MAE if we have results
                                if mean_abs_errors:
                                    mean_mae = sum(mean_abs_errors) / len(mean_abs_errors)
                                    
                                    # Only consider complete runs or runs with enough folds for early stopping
                                    if not skip_current_param_set or len(mean_abs_errors) >= early_stopping_folds:
                                        print(f"Mean MAE: {mean_mae}")
                                        print()
                                        
                                        # Update best parameters if this is better
                                        if mean_mae < best_mae:
                                            best_mae = mean_mae
                                            best_params = {
                                                'alpha': alpha,
                                                'lambda': lamb,
                                                'epsilon': epsilon,
                                                'layer': layer,
                                                'npl': npl,
                                                'mean_mae': mean_mae,  # Save the performance metric
                                                'folds_completed': len(mean_abs_errors)  # Track how many folds were actually used
                                            }
                                            print(f"New best parameters found with MAE: {best_mae:.4f}")
                                else:
                                    print("No valid results for this parameter set.")
                                    print()
                            except Exception as e:
                                print(f"Error during cross-validation: {e}. Skipping this parameter set.")
                                print()
                                continue

        # Store best parameters
        if best_params:
            self.best_params = best_params
            print("Best parameters found:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")
            # Save parameters to file
            self.save_parameters("best_params.pkl")
        else:
            print("No valid parameters found during grid search!")
        
        return best_params
    
    def save_parameters(self, filename="best_params.pkl"):
        """Save best parameters to a file
        
        Args:
            filename: Name of the file to save parameters to. Default is "best_params.pkl".
                     The extension determines the format:
                     - .pkl: pickle format (binary)
                     - .json: JSON format (text)
                     - .txt: Text format (text)
        """
        if not self.best_params:
            print("No parameters to save. Run grid_search first.")
            return False
            
        try:
            # Add model structure information
            params_to_save = self.best_params.copy()
            params_to_save['input_size'] = self.input_size
            params_to_save['output_size'] = self.output_size
            params_to_save['timestamp'] = pd.Timestamp.now().isoformat()
            
            # Determine the format based on file extension
            _, ext = os.path.splitext(filename)
            
            if ext.lower() == '.pkl':
                # Save as pickle (binary)
                with open(filename, 'wb') as f:
                    pickle.dump(params_to_save, f)
            elif ext.lower() == '.json':
                # Save as JSON (text)
                with open(filename, 'w') as f:
                    json.dump(params_to_save, f, indent=4)
            else:
                # Save as plain text
                with open(filename, 'w') as f:
                    f.write("# Neural Network Best Parameters\n")
                    f.write(f"# Saved on: {params_to_save['timestamp']}\n\n")
                    for key, value in params_to_save.items():
                        f.write(f"{key}: {value}\n")
                        
            print(f"Parameters successfully saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
            
    def load_parameters(self, filename="best_params.pkl"):
        """Load best parameters from a file
        
        Args:
            filename: Name of the file to load parameters from. Default is "best_params.pkl".
                     The extension determines the format:
                     - .pkl: pickle format (binary)
                     - .json: JSON format (text)
                     - .txt: Text format (text, simple key-value format only)
        
        Returns:
            bool: True if parameters were loaded successfully, False otherwise
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return False
            
        try:
            # Determine the format based on file extension
            _, ext = os.path.splitext(filename)
            
            if ext.lower() == '.pkl':
                # Load from pickle (binary)
                with open(filename, 'rb') as f:
                    self.best_params = pickle.load(f)
            elif ext.lower() == '.json':
                # Load from JSON (text)
                with open(filename, 'r') as f:
                    self.best_params = json.load(f)
            else:
                # Load from plain text (simple format)
                self.best_params = {}
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line.startswith('#') or not line:
                            continue
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            # Try to convert to appropriate type
                            try:
                                # Try as number
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                # Keep as string
                                pass
                            self.best_params[key] = value
                            
            print(f"Parameters successfully loaded from {filename}")
            print("Loaded parameters:", self.best_params)
            return True
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False
            
    def fit(self, features: np.ndarray, labels: np.ndarray, params_file=None):
        """Train the model with the best parameters
        
        Args:
            features: Input features for training
            labels: Target labels for training
            params_file: Optional file to load parameters from before training
        """
        # Try to load parameters if file is provided
        if params_file:
            if not self.load_parameters(params_file):
                print("Could not load parameters. Checking if grid search was run...")
        
        if not self.best_params:
            raise ValueError("No best parameters available. Run grid_search first or provide a valid parameters file.")
            
        # Extract parameters
        alpha = self.best_params.get('alpha', 0.01)
        lamb = self.best_params.get('lambda', 0.0)
        epsilon = self.best_params.get('epsilon', 1e-6)
        layer = self.best_params.get('layer', 1)
        npl = self.best_params.get('npl', 10)
        
        print(f"Training with parameters: alpha={alpha}, lambda={lamb}, epsilon={epsilon}, layers={layer}, neurons={npl}")
        
        # Initialize model with best parameters
        self.hidden_layers = [None] * layer
        self.weights = self.generate_weights_matrix(npl, layer)
        self.delta = self.generate_delta()
        self.gradient = self.generate_gradient()
        
        # Train with best parameters
        cost = self.train(features, labels, alpha, epsilon, lamb, max_epochs=5000, batch_size=64)
        print(f"Final training cost: {cost}")
        return cost
        
    def predict(self, test_id: np.ndarray, X_test: np.ndarray):
        """Generate predictions for test data"""
        if X_test.shape[0] == 0:
            raise ValueError("Test data is empty")
            
        # Forward pass to get predictions
        self.forward_propagation(X_test)
        
        # Combine IDs with predictions for submission
        predictions = pd.DataFrame({'Id': test_id, 'SalePrice': self.output.flatten()})
        return predictions

    @staticmethod
    def create_from_parameters(params_file, input_size=None, output_size=None):
        """
        Create a model from saved parameters
        
        Args:
            params_file: The file containing the saved parameters
            input_size: Input dimension (if not specified, will try to infer from parameters)
            output_size: Output dimension (if not specified, will try to infer from parameters)
        
        Returns:
            A neural network model initialized with the saved parameters
        """
        try:
            # First try to extract input_size and output_size from the parameter file
            temp_model = MLPNeuralNetwork(1, 1)  # Temporary model to load params
            
            if not temp_model.load_parameters(params_file):
                print(f"Failed to load parameters from {params_file}")
                return None
            
            # Get the model architecture from parameters
            if input_size is None:
                # Try to infer from the weights in the first layer
                if 'input_size' in temp_model.best_params:
                    input_size = temp_model.best_params['input_size']
                else:
                    print("Could not determine input_size from parameters. Please specify manually.")
                    return None
                
            if output_size is None:
                # Try to infer from the weights in the last layer
                if 'output_size' in temp_model.best_params:
                    output_size = temp_model.best_params['output_size']
                else:
                    # Default to 1 for regression
                    output_size = 1
                    print("Using default output_size=1. Specify manually if needed.")
            
            # Create a new model with the correct dimensions
            model = MLPNeuralNetwork(input_size, output_size, params_file)
            
            # Setup the model structure according to parameters
            if model.best_params:
                # Initialize the model structure
                layer = model.best_params.get('layer', 1)
                npl = model.best_params.get('npl', 10)
                
                model.hidden_layers = [None] * layer
                model.weights = model.generate_weights_matrix(npl, layer)
                model.delta = model.generate_delta()
                model.gradient = model.generate_gradient()
                
                print(f"Created model with {layer} hidden layers, {npl} neurons per layer")
                print("The model is ready for making predictions with predict()")
                
                return model
            else:
                print("Failed to create model from parameters")
                return None
        except Exception as e:
            print(f"Error creating model: {e}")
            return None

    @staticmethod
    def check_saved_parameters(custom_path=None):
        """
        Check if parameter files exist in the current directory or custom path
        
        Args:
            custom_path: Optional path to check for parameter files
        
        Returns:
            list: List of found parameter files
        """
        possible_files = ["best_params.pkl", "best_params.json", "best_params.txt"]
        found_files = []
        
        search_path = custom_path if custom_path else "."
        for file in possible_files:
            full_path = os.path.join(search_path, file)
            if os.path.exists(full_path):
                found_files.append(full_path)
        
        if found_files:
            print(f"Found {len(found_files)} parameter files:")
            for file in found_files:
                # Check file size and modification time
                size = os.path.getsize(file)
                modified = os.path.getmtime(file)
                modified_str = pd.Timestamp.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {file} ({size/1024:.1f} KB, modified: {modified_str})")
        else:
            print("No parameter files found.")
            if custom_path:
                print(f"Searched in: {custom_path}")
        
        return found_files

    def summarize_parameters(self):
        """
        Print a summary of the loaded parameters
        """
        if not self.best_params:
            print("No parameters loaded. Use load_parameters() or run grid_search().")
            return
        
        print("="*50)
        print("Model Parameters Summary")
        print("="*50)
        
        # Separate hyperparameters from other parameters
        hyperparams = {k: v for k, v in self.best_params.items() if k in ['alpha', 'lambda', 'epsilon', 'layer', 'npl']}
        other_params = {k: v for k, v in self.best_params.items() if k not in hyperparams}
        
        print("Hyperparameters:")
        for k, v in hyperparams.items():
            print(f"  {k}: {v}")
        
        if 'mean_mae' in other_params:
            print(f"\nPerformance: MAE = {other_params['mean_mae']:.6f}")
        
        if 'timestamp' in other_params:
            print(f"\nSaved on: {other_params['timestamp']}")
        
        print("\nModel Architecture:")
        print(f"  Input size: {self.input_size}")
        print(f"  Hidden layers: {hyperparams.get('layer', 'Unknown')}")
        print(f"  Neurons per layer: {hyperparams.get('npl', 'Unknown')}")
        print(f"  Output size: {self.output_size}")
        
        print("="*50)