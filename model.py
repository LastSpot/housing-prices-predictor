import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import pickle
import os
import json
import math
from sklearn.model_selection import KFold

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
        
        # Adam optimizer parameters
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep
        
        # Batch normalization parameters
        self.use_batch_norm = False
        self.gamma = []  # Scale parameters
        self.beta = []   # Shift parameters
        self.running_mean = []  # Running mean for inference
        self.running_var = []   # Running variance for inference
        self.z_norm = []  # Normalized z values
        
        # Check for existing parameter files
        if params_file:
            # User specified a parameter file
            if self.load_parameters(params_file):
                print(f"Loaded parameters from {params_file}")
        else:
            # Try to load from default locations
            possible_files = ["best_params.pkl", "best_params.json", "best_params.txt",
                             "best_params_rmse.pkl", "best_params_huber.pkl", 
                             "best_params_weighted_rmse.pkl", "best_params_log_cosh.pkl"]
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
            
            if 'mean_rmse' in self.best_params:
                print(f"\nPerformance: RMSE = {self.best_params['mean_rmse']:.6f}")
            elif 'mean_mae' in self.best_params:
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
    
    def sigmoid(self, x):
        """Apply sigmoid activation function with safety clipping"""
        # Clip values to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        """
        ReLU activation function for better regression performance
        f(x) = max(0, x)
        """
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        """
        Leaky ReLU activation to prevent dead neurons
        f(x) = x if x > 0 else alpha * x
        """
        return np.maximum(alpha * x, x)

    def linear(self, x):
        """
        Linear activation function for output layer in regression tasks
        f(x) = x
        """
        return x

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function with safety clipping"""
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def relu_derivative(self, x):
        """
        Derivative of ReLU function
        f'(x) = 1 if x > 0 else 0
        """
        return np.where(x > 0, 1, 0)

    def leaky_relu_derivative(self, x, alpha=0.01):
        """
        Derivative of Leaky ReLU function
        f'(x) = 1 if x > 0 else alpha
        """
        return np.where(x > 0, 1, alpha)

    def initialize_hidden_layers(self, batch_size, hidden_sizes, neurons_per_layer):
        """Initialize hidden layers with correct dimensions"""
        self.hidden_layers = []
        for _ in range(hidden_sizes):
            self.hidden_layers.append(np.zeros((batch_size, neurons_per_layer)))
    
    def initialize_batch_norm(self, neurons_per_layer, layers):
        """Initialize batch normalization parameters"""
        self.gamma = [np.ones(neurons_per_layer) for _ in range(layers)]
        self.beta = [np.zeros(neurons_per_layer) for _ in range(layers)]
        self.running_mean = [np.zeros(neurons_per_layer) for _ in range(layers)]
        self.running_var = [np.ones(neurons_per_layer) for _ in range(layers)]
        self.z_norm = [None for _ in range(layers)]
        self.use_batch_norm = True
    
    def batch_normalize(self, z, layer_idx, training=True, momentum=0.9, epsilon=1e-8):
        """
        Apply batch normalization to pre-activations
        
        Args:
            z: Pre-activation values
            layer_idx: Layer index
            training: Whether in training or inference mode
            momentum: Momentum for running statistics
            epsilon: Small constant to prevent division by zero
            
        Returns:
            Normalized pre-activations
        """
        if training:
            batch_mean = np.mean(z, axis=0)
            batch_var = np.var(z, axis=0)
            
            # Update running statistics
            self.running_mean[layer_idx] = momentum * self.running_mean[layer_idx] + (1 - momentum) * batch_mean
            self.running_var[layer_idx] = momentum * self.running_var[layer_idx] + (1 - momentum) * batch_var
            
            # Normalize
            z_norm = (z - batch_mean) / np.sqrt(batch_var + epsilon)
            self.z_norm[layer_idx] = z_norm
        else:
            # Use running statistics for inference
            z_norm = (z - self.running_mean[layer_idx]) / np.sqrt(self.running_var[layer_idx] + epsilon)
        
        # Scale and shift
        return self.gamma[layer_idx] * z_norm + self.beta[layer_idx]

    def forward_propagation(self, data, training=True):
        """Forward pass through the network with robust error checks and batch normalization"""
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
        
        # Initialize batch norm params if needed and not already initialized
        if self.use_batch_norm and (not self.gamma or len(self.gamma) < len(self.hidden_layers)):
            neurons_per_layer = self.weights[0].shape[1]
            self.initialize_batch_norm(neurons_per_layer, len(self.hidden_layers))
        
        # Forward pass through each hidden layer
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
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                z = self.batch_normalize(z, i, training=training)
            
            # Apply ReLU activation for hidden layers
            self.hidden_layers[i] = self.relu(z)
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)
        
        # Output layer - use linear activation for regression
        z = a.dot(self.weights[-1])
        
        # Final safety check
        if np.isnan(z).any():
            z = np.nan_to_num(z, nan=0.0)
            print("Warning: NaN values in final output replaced with zeros")
        
        self.output = z  # Linear activation for regression
        return self.output
    
    def backward_propagation(self, data, actual):
        """
        Backward pass through the network with ReLU derivative and batch normalization
        
        Args:
            data: Input features
            actual: True target values
        
        Returns:
            gradients: Weight gradients for parameter updates
        """
        # Reshape target to match output if one-dimensional
        if len(actual.shape) == 1:
            actual = actual.reshape(-1, 1)
        
        # Forward pass to get activations (needed for gradient calculations)
        self.forward_propagation(data, training=True)
        
        # Create list to store gradients for each layer
        gradients = [None] * len(self.weights)
        batch_size = data.shape[0]
        
        # Add bias term to input
        a = np.insert(data, 0, 1, axis=1)
        
        # Calculate output layer delta (error)
        # For regression with linear output: delta = (output - actual)
        delta = self.output - actual
        
        # Safety check for NaN/Inf
        if np.isnan(delta).any() or np.isinf(delta).any():
            delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: NaN or Inf values in delta replaced with zeros")
        
        # Calculate activations for each layer (needed for batch norm backward pass)
        activations = [a]
        for i in range(len(self.hidden_layers)):
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)
            activations.append(a)
        
        # Backward pass for output layer
        gradients[-1] = (1/batch_size) * activations[-1].T.dot(delta)
        
        # Backward pass for hidden layers
        for i in range(len(self.hidden_layers)-1, -1, -1):
            # Calculate delta for current hidden layer
            if i == len(self.hidden_layers)-1:
                # Delta from output layer (remove bias term from weights)
                delta = delta.dot(self.weights[-1][1:].T)
            else:
                # Delta from next hidden layer (remove bias term from weights)
                delta = delta.dot(self.weights[i+1][1:].T)
            
            # Apply ReLU derivative
            # For hidden layers with ReLU activation
            delta = delta * self.relu_derivative(self.hidden_layers[i])
            
            # Apply batch normalization backward pass if enabled
            if self.use_batch_norm:
                # Gradient through batch normalization 
                # This is a simplified version - full implementation would involve
                # more complex computations for gamma and beta gradients
                if self.z_norm[i] is not None:
                    # Scale delta by gamma
                    delta = delta * self.gamma[i]
                    
                    # Normalization gradients calculation omitted for simplicity
                    # In a full implementation, you would compute dgamma, dbeta, etc.
            
            # Safety check for NaN/Inf
            if np.isnan(delta).any() or np.isinf(delta).any():
                delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: NaN or Inf values in layer {i} delta replaced with zeros")
            
            # Calculate gradients for current layer
            if i == 0:
                # For input layer
                gradients[i] = (1/batch_size) * activations[i].T.dot(delta)
            else:
                # For hidden layers
                gradients[i] = (1/batch_size) * activations[i].T.dot(delta)
            
            # Safety check and gradient clipping
            if np.isnan(gradients[i]).any() or np.isinf(gradients[i]).any():
                gradients[i] = np.nan_to_num(gradients[i], nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: NaN or Inf values in layer {i} gradients replaced with zeros")
            
            # Gradient clipping to prevent exploding gradients
            gradients[i] = np.clip(gradients[i], -1.0, 1.0)
        
        return gradients
    
    def huber_loss(self, y_true, y_pred, delta=100000):
        """
        Huber loss - combines benefits of MSE and MAE by being less sensitive to outliers
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            delta: Threshold where the loss changes from MSE to MAE
            
        Returns:
            The Huber loss value
        """
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * np.square(quadratic) + delta * linear)
    
    def weighted_rmse(self, y_true, y_pred, weights=None):
        """
        Weighted RMSE - gives more importance to errors on high-value properties
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            weights: Optional weights for each sample (if None, weights based on true values)
            
        Returns:
            The weighted RMSE value
        """
        # If no weights provided, weight by the true values (higher value = higher weight)
        if weights is None:
            # Normalize weights to have mean of 1
            weights = y_true / np.mean(y_true)
        
        squared_errors = np.square(y_true - y_pred)
        weighted_squared_errors = squared_errors * weights
        mean_weighted_squared_error = np.mean(weighted_squared_errors)
        return np.sqrt(mean_weighted_squared_error)
    
    def log_cosh_loss(self, y_true, y_pred):
        """
        Log-cosh loss - approximates Huber loss but is twice differentiable
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            The log-cosh loss value
        """
        error = y_true - y_pred
        return np.mean(np.log(np.cosh(error)))
    
    def calculate_loss(self, y_true, y_pred, loss_type='rmse', delta=100000, weights=None):
        """
        Calculate the specified loss function
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_type: Type of loss function to use ('rmse', 'mae', 'huber', 'weighted_rmse', 'log_cosh')
            delta: Threshold for Huber loss
            weights: Optional weights for weighted_rmse
            
        Returns:
            The calculated loss value
        """
        if loss_type == 'rmse':
            mse = np.mean(np.square(y_true - y_pred))
            return np.sqrt(mse)
        elif loss_type == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif loss_type == 'huber':
            return self.huber_loss(y_true, y_pred, delta)
        elif loss_type == 'weighted_rmse':
            return self.weighted_rmse(y_true, y_pred, weights)
        elif loss_type == 'log_cosh':
            return self.log_cosh_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train(self, data, target, learning_rate=0.01, convergence_threshold=0.001, 
              regularization=0.0001, max_epochs=1000, batch_size=32, loss_type='rmse', 
              use_adam=True, use_batch_norm=False, beta1=0.9, beta2=0.999, 
              epsilon=1e-8, verbose=True):
        """
        Train the neural network with Adam optimizer and batch normalization.
        
        Args:
            data: Training data features
            target: Training data targets
            learning_rate: Learning rate for gradient descent
            convergence_threshold: Stop when improvements are below this threshold
            regularization: L2 regularization strength
            max_epochs: Maximum number of training epochs
            batch_size: Size of mini-batches
            loss_type: Type of loss function to use
            use_adam: Whether to use Adam optimizer
            use_batch_norm: Whether to use batch normalization
            beta1: Adam parameter - exponential decay rate for first moment
            beta2: Adam parameter - exponential decay rate for second moment
            epsilon: Small value to prevent division by zero
            verbose: Whether to print progress
            
        Returns:
            loss_history: History of loss values during training
        """
        # Input validation
        if data.size == 0 or target.size == 0:
            raise ValueError("Empty training data or targets")
        
        # Reshape target if necessary
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        
        # Enable batch normalization if requested
        self.use_batch_norm = use_batch_norm
        if use_batch_norm and (not self.gamma or len(self.gamma) < len(self.weights) - 1):
            neurons_per_layer = self.weights[0].shape[1]
            self.initialize_batch_norm(neurons_per_layer, len(self.weights) - 1)
        
        # Initialize Adam optimizer if requested
        if use_adam:
            self.initialize_adam()
            self.t = 0  # Reset timestep counter
        
        loss_history = []
        prev_loss = float('inf')
        n_samples = data.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        # Training loop
        for epoch in range(max_epochs):
            # Shuffle data for stochastic gradient descent
            indices = np.random.permutation(n_samples)
            shuffled_data = data[indices]
            shuffled_target = target[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                batch_data = shuffled_data[start_idx:end_idx]
                batch_target = shuffled_target[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward_propagation(batch_data, training=True)
                
                # Check for NaN values in predictions
                if np.isnan(predictions).any():
                    # Handle NaN predictions
                    print("Warning: NaN predictions detected. Taking corrective action...")
                    predictions = np.nan_to_num(predictions, nan=0.0)
                
                # Backward pass
                gradients = self.backward_propagation(batch_data, batch_target)
                
                # Update weights using Adam or standard gradient descent
                for i in range(len(self.weights)):
                    # Add L2 regularization term to gradient
                    regularized_gradient = gradients[i] + regularization * self.weights[i]
                    
                    # Avoid NaN/Inf in gradients
                    if np.isnan(regularized_gradient).any() or np.isinf(regularized_gradient).any():
                        regularized_gradient = np.nan_to_num(regularized_gradient, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if use_adam:
                        # Update with Adam optimizer
                        self.t += 1
                        self.weights[i] = self.adam_update(i, regularized_gradient, learning_rate, 
                                                         beta1, beta2, epsilon)
                    else:
                        # Standard gradient descent
                        self.weights[i] -= learning_rate * regularized_gradient
                    
                    # Check for NaN values in weights
                    if np.isnan(self.weights[i]).any():
                        # Handle NaN weights (critical error)
                        raise ValueError(f"NaN weights detected in layer {i}. Training failed.")
                
                # Calculate batch loss
                batch_loss = self.calculate_loss(predictions, batch_target, loss_type)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
            
            # Record loss history
            loss_history.append(epoch_loss)
            
            # Check convergence
            if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
                print(f"Epoch {epoch}: {loss_type.upper()} = {epoch_loss:.6f}")
            
            # Check for convergence
            improvement = prev_loss - epoch_loss
            if improvement < convergence_threshold and epoch > 50:
                if verbose:
                    print(f"Converged after {epoch+1} epochs with {loss_type.upper()} = {epoch_loss:.6f}")
                break
            
            prev_loss = epoch_loss
        
        return loss_history

    def initialize_adam(self):
        """Initialize Adam optimizer parameters"""
        # First moment vector (momentum)
        self.m = [np.zeros_like(w) for w in self.weights]
        # Second moment vector (RMSprop)
        self.v = [np.zeros_like(w) for w in self.weights]

    def adam_update(self, layer, gradient, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Update weights using Adam optimizer.
        
        Args:
            layer: Layer index
            gradient: Gradient for current layer
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant to prevent division by zero
            
        Returns:
            Updated weights for the layer
        """
        # Update biased first moment estimate
        self.m[layer] = beta1 * self.m[layer] + (1 - beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v[layer] = beta2 * self.v[layer] + (1 - beta2) * (gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        m_corrected = self.m[layer] / (1 - beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_corrected = self.v[layer] / (1 - beta2 ** self.t)
        
        # Update weights
        weights_update = learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        
        # Safety check for NaN/Inf
        if np.isnan(weights_update).any() or np.isinf(weights_update).any():
            weights_update = np.nan_to_num(weights_update, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Warning: NaN or Inf values in weight updates for layer {layer} replaced with zeros")
        
        # Return updated weights
        return self.weights[layer] - weights_update

    def test(self, X_test: np.ndarray, y_test: np.ndarray, loss_type='rmse'):
        """Test the neural network on validation data
        
        Args:
            X_test: Test features
            y_test: Test labels
            loss_type: Loss function to use for evaluation
            
        Returns:
            The calculated loss value
        """
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            raise ValueError("Test data is empty")
            
        # Forward pass on test data
        self.forward_propagation(X_test)
        
        # Reshape y_test if necessary
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            
        # Return the specified loss
        return self.calculate_loss(y_test, self.output, loss_type)
    
    def grid_search(self, data, target, param_grid, k_fold=5, loss_type='rmse', use_adam=True, use_batch_norm=False, verbose=True):
        """
        Perform grid search with cross-validation to find optimal hyperparameters.
        
        Args:
            data: Training data
            target: Target values
            param_grid: Dictionary of parameters to search
            k_fold: Number of folds for cross-validation
            loss_type: Type of loss function to use ('rmse', 'mae', 'huber', 'weighted_rmse', 'log_cosh')
            use_adam: Whether to use Adam optimizer
            use_batch_norm: Whether to use batch normalization
            verbose: Whether to print progress
            
        Returns:
            best_params: Best parameters found
        """
        if data.size == 0 or target.size == 0:
            raise ValueError("Empty training data or targets")
        
        # Reshape target if necessary
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        
        # Extract parameters from grid
        hidden_neurons_grid = param_grid.get('hidden_neurons', [10])
        hidden_layers_grid = param_grid.get('hidden_layers', [1])
        learning_rates = param_grid.get('learning_rate', [0.01])
        reg_strengths = param_grid.get('regularization', [0.0001])
        batch_sizes = param_grid.get('batch_size', [32])
        
        best_score = float('inf')
        best_params = {}
        total_combinations = (len(hidden_neurons_grid) * len(hidden_layers_grid) * 
                              len(learning_rates) * len(reg_strengths) * len(batch_sizes))
        
        print(f"Grid search with {total_combinations} parameter combinations")
        print(f"Loss function: {loss_type}, Optimizer: {'Adam' if use_adam else 'SGD'}, Batch normalization: {'Yes' if use_batch_norm else 'No'}")
        
        combination_count = 0
        
        # Split data into k folds
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # Iterate through all parameter combinations
        for hidden_neurons in hidden_neurons_grid:
            for hidden_layers in hidden_layers_grid:
                for learning_rate in learning_rates:
                    for reg_strength in reg_strengths:
                        for batch_size in batch_sizes:
                            combination_count += 1
                            if verbose:
                                print(f"\nCombination {combination_count}/{total_combinations}:")
                                print(f"Hidden layers: {hidden_layers}, Neurons: {hidden_neurons}, Learning rate: {learning_rate}, L2: {reg_strength}, Batch size: {batch_size}")
                            
                            # Initialize weights for this configuration
                            self.initialize_weights(hidden_layers, hidden_neurons)
                            
                            fold_scores = []
                            
                            # Cross-validation
                            for train_idx, val_idx in kf.split(data):
                                # Split data
                                train_data, val_data = data[train_idx], data[val_idx]
                                train_target, val_target = target[train_idx], target[val_idx]
                                
                                # Train model
                                self.train(train_data, train_target, learning_rate=learning_rate,
                                          regularization=reg_strength, batch_size=batch_size,
                                          loss_type=loss_type, use_adam=use_adam, 
                                          use_batch_norm=use_batch_norm, verbose=False)
                                
                                # Validate model
                                val_score = self.test(val_data, val_target, loss_type=loss_type)
                                fold_scores.append(val_score)
                                
                                # Early stopping if model is performing extremely poorly
                                if np.isnan(val_score) or val_score > 10 * best_score:
                                    print("    Early stopping due to poor performance")
                                    fold_scores = [float('inf')] * k_fold
                                    break
                            
                            # Calculate average score across folds
                            avg_score = np.mean(fold_scores)
                            if verbose:
                                print(f"    Average {loss_type.upper()}: {avg_score:.6f}")
                            
                            # Check if best so far
                            if avg_score < best_score:
                                best_score = avg_score
                                best_params = {
                                    'hidden_layers': hidden_layers,
                                    'hidden_neurons': hidden_neurons,
                                    'learning_rate': learning_rate,
                                    'regularization': reg_strength,
                                    'batch_size': batch_size,
                                    'loss_type': loss_type,
                                    'use_adam': use_adam,
                                    'use_batch_norm': use_batch_norm
                                }
                                
                                if verbose:
                                    print(f"    New best {loss_type.upper()}: {best_score:.6f}")
                                    
                                # Save best parameters so far with the loss type in filename
                                self.best_params = best_params
                                self.save_parameters(f"best_params_{loss_type}.pkl")
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best {loss_type.upper()}: {best_score:.6f}")
        
        # Initialize weights with best parameters
        self.initialize_weights(best_params['hidden_layers'], best_params['hidden_neurons'])
        self.best_params = best_params
        
        # Save final best parameters
        self.save_parameters(f"best_params_{loss_type}.pkl")
        
        return best_params

    def fit(self, data, target, params_file=None, loss_type=None, use_adam=None, use_batch_norm=None):
        """
        Train the model with the best parameters found by grid search.
        
        Args:
            data: Training data
            target: Target values
            params_file: File with saved parameters
            loss_type: Loss function to use (overrides saved parameter)
            use_adam: Whether to use Adam optimizer (overrides saved parameter)
            use_batch_norm: Whether to use batch normalization (overrides saved parameter)
            
        Returns:
            loss_history: History of loss values during training
        """
        if data.size == 0 or target.size == 0:
            raise ValueError("Empty training data or targets")
        
        # Reshape target if necessary
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        
        # Load parameters if specified
        if params_file:
            self.load_parameters(params_file)
        
        # Check if we have best parameters
        if not self.best_params:
            raise ValueError("No best parameters available. Run grid_search first or load parameters.")
        
        # Override parameters if specified
        if loss_type is not None:
            self.best_params['loss_type'] = loss_type
        if use_adam is not None:
            self.best_params['use_adam'] = use_adam
        if use_batch_norm is not None:
            self.best_params['use_batch_norm'] = use_batch_norm
        
        # Get best parameters
        hidden_layers = self.best_params.get('hidden_layers', 1)
        hidden_neurons = self.best_params.get('hidden_neurons', 10)
        learning_rate = self.best_params.get('learning_rate', 0.01)
        regularization = self.best_params.get('regularization', 0.0001)
        batch_size = self.best_params.get('batch_size', 32)
        loss_type = self.best_params.get('loss_type', 'rmse')
        use_adam = self.best_params.get('use_adam', True)
        use_batch_norm = self.best_params.get('use_batch_norm', False)
        
        # Initialize weights if not already done
        if not hasattr(self, 'weights') or len(self.weights) == 0:
            self.initialize_weights(hidden_layers, hidden_neurons)
        
        # Train with best parameters
        cost = self.train(data, target, learning_rate, max_epochs=10000, batch_size=batch_size, loss_type=loss_type, use_adam=use_adam)
        print(f"Final training cost ({loss_type}): {cost}")
        return cost

    def initialize_weights(self, hidden_layers, neurons_per_layer):
        """
        Initialize weights using He initialization for ReLU activations.
        
        Args:
            hidden_layers: Number of hidden layers
            neurons_per_layer: Number of neurons per hidden layer
        """
        self.hidden_layers = [None] * hidden_layers
        self.weights = []
        
        # Input layer to first hidden layer
        fan_in = self.input_size
        fan_out = neurons_per_layer
        limit = np.sqrt(2 / fan_in)  # He initialization for ReLU
        
        # Input layer to first hidden layer (with bias)
        w1 = np.random.normal(0, limit, size=(self.input_size + 1, neurons_per_layer))
        self.weights.append(w1)
        
        # Hidden layers to hidden layers
        for i in range(1, hidden_layers):
            fan_in = neurons_per_layer
            fan_out = neurons_per_layer
            limit = np.sqrt(2 / fan_in)  # He initialization for ReLU
            
            wi = np.random.normal(0, limit, size=(neurons_per_layer + 1, neurons_per_layer))
            self.weights.append(wi)
        
        # Last hidden layer to output layer
        fan_in = neurons_per_layer
        fan_out = self.output_size
        limit = np.sqrt(2 / fan_in)  # He initialization
        
        # Last hidden layer to output (with bias)
        wo = np.random.normal(0, limit, size=(neurons_per_layer + 1, self.output_size))
        self.weights.append(wo)
        
        # Initialize batch normalization parameters if needed
        if getattr(self, 'use_batch_norm', False):
            self.initialize_batch_norm(neurons_per_layer, hidden_layers)
        
        # Initialize Adam optimizer parameters if needed
        if hasattr(self, 'm') and self.m is not None:
            self.initialize_adam()

    def save_parameters(self, filename='best_params.pkl'):
        """
        Save parameters to file
        
        Args:
            filename: File to save parameters to
        """
        if not self.best_params:
            print("No parameters to save. Run grid_search first.")
            return False
        
        try:
            # Add model structure information
            self.best_params['input_size'] = self.input_size
            self.best_params['output_size'] = self.output_size
            self.best_params['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Determine file format based on extension
            if filename.endswith('.pkl'):
                with open(filename, 'wb') as f:
                    pickle.dump(self.best_params, f)
                
            elif filename.endswith('.json'):
                with open(filename, 'w') as f:
                    # Convert numpy types to Python types for JSON serialization
                    params_copy = {}
                    for k, v in self.best_params.items():
                        if isinstance(v, np.integer):
                            params_copy[k] = int(v)
                        elif isinstance(v, np.floating):
                            params_copy[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            params_copy[k] = v.tolist()
                        else:
                            params_copy[k] = v
                        
                    json.dump(params_copy, f, indent=4)
                
            else:  # Default to text format
                with open(filename, 'w') as f:
                    f.write("# Neural Network Best Parameters\n")
                    f.write(f"# Saved on: {self.best_params['timestamp']}\n\n")
                    
                    for k, v in self.best_params.items():
                        f.write(f"{k}: {v}\n")
                    
            print(f"Parameters saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False

    def load_parameters(self, filename='best_params.pkl'):
        """
        Load parameters from file
        
        Args:
            filename: File to load parameters from
        
        Returns:
            True if parameters were loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filename):
                print(f"File {filename} does not exist.")
                return False
            
            # Determine file format based on extension
            if filename.endswith('.pkl'):
                with open(filename, 'rb') as f:
                    self.best_params = pickle.load(f)
                
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    self.best_params = json.load(f)
                
            else:  # Try to parse as text
                params = {}
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Try to convert to appropriate type
                            try:
                                # Try as float first
                                value = float(value)
                                # Convert to int if it's a whole number
                                if value.is_integer():
                                    value = int(value)
                            except ValueError:
                                # Keep as string if conversion fails
                                pass
                            
                            params[key] = value
                        
                if params:
                    self.best_params = params
                else:
                    print(f"No valid parameters found in {filename}")
                    return False
                
            print(f"Parameters loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False

    def predict(self, X_test):
        """
        Generate predictions for test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted values
        """
        if X_test.shape[0] == 0:
            raise ValueError("Test data is empty")
        
        # Forward pass to get predictions
        self.forward_propagation(X_test)
        
        return self.output

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
        
        # Check which loss metric was used
        if 'loss_type_used' in other_params:
            loss_type = other_params['loss_type_used']
            print(f"\nLoss function used: {loss_type}")
            
        if 'mean_rmse' in other_params:
            print(f"\nPerformance: RMSE = {other_params['mean_rmse']:.6f}")
        elif 'mean_mae' in other_params:
            print(f"\nPerformance: MAE = {other_params['mean_mae']:.6f}")
        elif 'mean_huber' in other_params:
            print(f"\nPerformance: Huber Loss = {other_params['mean_huber']:.6f}")
        elif 'mean_weighted_rmse' in other_params:
            print(f"\nPerformance: Weighted RMSE = {other_params['mean_weighted_rmse']:.6f}")
        elif 'mean_log_cosh' in other_params:
            print(f"\nPerformance: Log-Cosh Loss = {other_params['mean_log_cosh']:.6f}")
        
        if 'timestamp' in other_params:
            print(f"\nSaved on: {other_params['timestamp']}")
        
        print("\nModel Architecture:")
        print(f"  Input size: {self.input_size}")
        print(f"  Hidden layers: {hyperparams.get('layer', 'Unknown')}")
        print(f"  Neurons per layer: {hyperparams.get('npl', 'Unknown')}")
        print(f"  Output size: {self.output_size}")
        
        print("="*50)