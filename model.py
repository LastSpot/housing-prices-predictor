import numpy as np
from sklearn.metrics import mean_absolute_error

class MLPNeuralNetwork:
    def __init__(self, input_size: int, output_size: int):
        self.model = 'MLP - Multi-Layer Perceptron'
        self.type = 'Regression'
        self.purpose = 'Housing Prices Prediction'
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.best_params = {}
        self.hidden_layers = []  # Ensure hidden layers are initialized
        
    def get_info(self):
        print(f"Model: {self.model}")
        print(f"Type: {self.type}")
        print(f"Purpose: {self.purpose}")
        print()
        
    def generate_weights_values(self, x, y):
        return np.random.randn(x, y) * np.sqrt(1.0 / x) 
        
    def generate_weights_matrix(self, neurons_per_layer: int, hidden_sizes: int):
        weight_matrix = [self.generate_weights_values(self.input_size + 1, neurons_per_layer)]
        for _ in range(1, hidden_sizes):
            weight_matrix.append(self.generate_weights_values(neurons_per_layer + 1, neurons_per_layer))
        weight_matrix.append(self.generate_weights_values(neurons_per_layer + 1, self.output_size))
        return weight_matrix
    
    def generate_delta(self):
        return [np.zeros_like(w) for w in self.weights]
    
    def generate_gradient(self):
        return [np.zeros_like(w) for w in self.weights]
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return 1.0 / (1.0 + np.exp(-z))
    
    def forward_propagation(self, data):
        a = np.insert(data, 0, 1, axis=1)  # Add bias term
        
        for i in range(len(self.hidden_layers)):
            z = a.dot(self.weights[i])
            
            if np.isnan(z).any():
                raise ValueError(f"NaN detected in z at layer {i}")
            
            self.hidden_layers[i] = self.sigmoid(z)
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)  # Add bias

        z = a.dot(self.weights[-1])
        
        if np.isnan(z).any():
            raise ValueError("NaN detected in final output layer before activation")
        
        self.output = z

        if np.isnan(self.output).any():
            raise ValueError("Error: self.output contains NaNs after forward propagation!")

    
    def backward_propagation(self, data, actual):
        delta = self.output - actual.reshape(-1, 1)
        batch_size = data.shape[0]

        self.delta[-1] = delta
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            sigmoid_derivative = self.hidden_layers[i] * (1 - self.hidden_layers[i])
            delta = delta.dot(self.weights[i + 1].T)[:, 1:] * sigmoid_derivative
            self.delta[i] = delta
        
        a = np.insert(data, 0, 1, axis=1)
        self.gradient[0] = a.T.dot(self.delta[0]) / batch_size
        for i in range(len(self.weights) - 1):
            a = np.insert(self.hidden_layers[i], 0, 1, axis=1)
            self.gradient[i + 1] = a.T.dot(self.delta[i + 1]) / batch_size
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, alpha: float, epsilon: float, lamb: float, max_epochs: int, batch_size: int):
        previous_cost = float('inf')
        num_samples = X_train.shape[0]

        for epoch in range(max_epochs):
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch)

                for j in range(len(self.weights)):
                    self.weights[j] -= alpha * (self.gradient[j] + (lamb / num_samples) * self.weights[j])  # Corrected weight update

            self.forward_propagation(X_train)
            cost = mean_absolute_error(self.output, y_train) + (lamb / (2 * num_samples)) * sum(np.sum(w ** 2) for w in self.weights)

            if abs(cost - previous_cost) <= epsilon:
                print(f"Converged at epoch {epoch}")
                return cost
            
            previous_cost = cost

        print(f"Training stopped after reaching max_epochs ({max_epochs}).")
    
    def test(self, X_test: np.ndarray, y_test: np.ndarray):
        self.forward_propagation(X_test)
        if np.isnan(self.output).any():
            raise ValueError("Error: self.output contains NaN values!")
        return mean_absolute_error(self.output, y_test)
    
    def grid_search(self, features: np.ndarray, labels: np.ndarray, params: dict, k_fold_index: object):
        alphas, lambdas, epsilons = params['alphas'], params['lambdas'], params['epsilons']
        hidden_sizes, neurons_per_layer = params['hidden_sizes'], params['neurons_per_layer']
        
        best_mae, best_params = float('inf'), {}

        for alpha in alphas:
            for lamb in lambdas:
                for epsilon in epsilons:
                    for layer in hidden_sizes:
                        for npl in neurons_per_layer:
                            self.hidden_layers = [np.zeros((features.shape[0], npl)) for _ in range(layer)]
                            self.weights = self.generate_weights_matrix(npl, layer)
                            self.delta = self.generate_delta()
                            self.gradient = self.generate_gradient()
                            mean_abs_errors = []
                            
                            for i, w in enumerate(self.weights):
                                if np.isnan(w).any() or np.isinf(w).any():
                                    raise ValueError(f"Error: Weights in layer {i} contain NaN or Inf values!")

                            for train_index, test_index in k_fold_index:
                                X_train, X_test = features[train_index], features[test_index]
                                y_train, y_test = labels[train_index], labels[test_index]

                                self.train(X_train, y_train, alpha, epsilon, lamb, max_epochs=5000, batch_size=64)
                                mae = self.test(X_test, y_test)
                                
                                if np.isnan(mae):
                                    raise ValueError("Error: MAE computation returned NaN!")
                                
                                mean_abs_errors.append(mae)

                            if len(mean_abs_errors) == 0:
                                raise ValueError("Error: mean_abs_errors is empty! Something went wrong in training/testing.")

                            mean_mae = np.mean(mean_abs_errors)
                            if mean_mae < best_mae:
                                best_mae, best_params = mean_mae, {'alpha': alpha, 'lambda': lamb, 'epsilon': epsilon, 'layer': layer, 'npl': npl}

        self.best_params = best_params
        print(self.best_params)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        if not self.best_params:
            raise ValueError("Model must be trained with grid_search first.")

        self.hidden_layers = [np.zeros((features.shape[0], self.best_params['npl'])) for _ in range(self.best_params['layer'])]
        self.weights = self.generate_weights_matrix(self.best_params['npl'], self.best_params['layer'])
        self.delta = self.generate_delta()
        self.gradient = self.generate_gradient()

        self.train(features, labels, self.best_params['alpha'], self.best_params['epsilon'], self.best_params['lambda'], 1000, 32)
        return self.test(features, labels)

    def predict(self, test_id: np.ndarray, X_test: np.ndarray):
        self.forward_propagation(X_test)
        return test_id, self.output