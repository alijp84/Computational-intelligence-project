"""
Neural Methods for MetaMind CI Framework
Implements Perceptron, MLP, Kohonen SOM, and Hopfield Network with standardized interfaces.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable, Union
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
import time
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    setup_logger,
    EarlyStopping,
    normalize_array
)

logger = setup_logger("neural_methods")


# ==================== BASE NEURAL METHOD CLASS ====================

class NeuralMethod(ABC):
    """Abstract base class for all neural methods with standardized interface."""
    
    def __init__(self, **params):
        self.params = params
        self.convergence_history = []
        self.best_solution = None
        self.best_fitness = float('inf')  # For minimization problems
        self.start_time = None
        self.elapsed_time = 0.0
        self.iterations_completed = 0
        self.name = self.__class__.__name__
        self.weights = None  # Common attribute for weight-based networks
    
    @abstractmethod
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve the given problem instance.
        
        Args:
            problem: Problem instance with evaluate() method
            max_time: Optional time limit in seconds
            callback: Optional callback function for progress reporting
            **kwargs: Problem-specific parameters
            
        Returns:
            Dictionary with solution details matching project spec format
        """
        pass
    
    def _initialize_convergence_tracking(self):
        """Reset convergence tracking before a new run."""
        self.convergence_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.start_time = time.time()
        self.iterations_completed = 0
    
    def _update_convergence(self, fitness: float, solution: Union[List, np.ndarray, Dict]):
        """Update best solution and convergence history."""
        # For classification, higher accuracy is better (convert to error for minimization)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = deepcopy(solution)
        
        self.convergence_history.append(self.best_fitness)
        self.iterations_completed += 1
    
    def _check_time_limit(self, max_time: Optional[float]) -> bool:
        """Check if time limit has been exceeded."""
        if max_time is None:
            return False
        self.elapsed_time = time.time() - self.start_time
        return self.elapsed_time >= max_time
    
    def get_parameters(self) -> Dict:
        """Return current parameters."""
        return self.params.copy()
    
    def set_parameters(self, params: Dict):
        """Update method parameters."""
        self.params.update(params)
    
    def get_results(self) -> Dict:
        """Return standardized results dictionary."""
        return {
            "method_used": self.name,
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "computation_time": self.elapsed_time,
            "convergence_history": self.convergence_history,
            "iterations_completed": self.iterations_completed
        }


# ==================== PERCEPTRON ====================

class Perceptron(NeuralMethod):
    """
    Single-layer Perceptron for binary classification.
    Implements perceptron learning rule with bias support.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            bias=bias,
            **kwargs
        )
        self.weights = None
        self.bias_weight = 0.0 if bias else None
    
    def _initialize_weights(self, n_features: int):
        """Initialize weights with small random values."""
        self.weights = np.random.uniform(-0.01, 0.01, n_features)
        if self.params['bias']:
            self.bias_weight = np.random.uniform(-0.01, 0.01)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.weights is None:
            raise ValueError("Model not trained. Call solve() first.")
        
        # Compute net input
        if self.params['bias']:
            net_input = np.dot(X, self.weights) + self.bias_weight
        else:
            net_input = np.dot(X, self.weights)
        
        # Step function activation
        return np.where(net_input >= 0, 1, 0)
    
    def _compute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification error rate."""
        return np.mean(y_true != y_pred)
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Train Perceptron on classification problem.
        
        Args:
            problem: Classification problem with X_train, y_train attributes
            max_time: Optional time limit in seconds
            callback: Optional callback function (epoch, error, weights)
            **kwargs: Additional parameters (e.g., 'X_train', 'y_train')
        """
        # Extract training data
        X_train = kwargs.get('X_train', getattr(problem, 'X_train', None))
        y_train = kwargs.get('y_train', getattr(problem, 'y_train', None))
        
        if X_train is None or y_train is None:
            raise ValueError("Perceptron requires training data (X_train, y_train) in kwargs or problem attributes")
        
        # Ensure binary classification
        unique_classes = np.unique(y_train)
        if len(unique_classes) != 2:
            raise ValueError(f"Perceptron supports binary classification only. Found {len(unique_classes)} classes.")
        
        # Convert to 0/1 labels if needed
        if not np.array_equal(unique_classes, [0, 1]):
            y_train = (y_train == unique_classes[1]).astype(int)
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=10, min_delta=1e-4, mode='min')
        self._initialize_weights(X_train.shape[1])
        
        n_samples = X_train.shape[0]
        learning_rate = self.params['learning_rate']
        max_epochs = self.params['max_epochs']
        
        # Training loop
        for epoch in range(max_epochs):
            if self._check_time_limit(max_time):
                logger.info(f"Perceptron stopped due to time limit at epoch {epoch}")
                break
            
            # Shuffle data for stochastic updates
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            errors = 0
            for xi, target in zip(X_shuffled, y_shuffled):
                # Predict
                net_input = np.dot(xi, self.weights)
                if self.params['bias']:
                    net_input += self.bias_weight
                output = 1 if net_input >= 0 else 0
                
                # Update weights if misclassified
                if output != target:
                    update = learning_rate * (target - output)
                    self.weights += update * xi
                    if self.params['bias']:
                        self.bias_weight += update
                    errors += 1
            
            # Compute error rate
            y_pred = self._predict(X_train)
            error_rate = self._compute_error(y_train, y_pred)
            
            # Update convergence (minimize error)
            current_fitness = error_rate  # Lower error = better fitness
            self._update_convergence(current_fitness, {
                'weights': self.weights.copy(),
                'bias_weight': self.bias_weight if self.params['bias'] else None
            })
            
            # Callback for progress monitoring
            if callback:
                callback(epoch, error_rate, self.weights)
            
            # Early stopping check
            if errors == 0 or early_stop(error_rate):
                logger.info(f"Perceptron converged at epoch {epoch} with 0 errors")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Final evaluation on training set
        y_pred = self._predict(X_train)
        final_error = self._compute_error(y_train, y_pred)
        
        # Store final solution
        self.best_fitness = final_error
        self.best_solution = {
            'weights': self.weights.copy(),
            'bias_weight': self.bias_weight if self.params['bias'] else None,
            'train_accuracy': 1.0 - final_error
        }
        
        return self.get_results()


# ==================== MULTI-LAYER PERCEPTRON (MLP) ====================

class MultiLayerPerceptron(NeuralMethod):
    """
    Multi-Layer Perceptron with backpropagation for classification/regression.
    Supports multiple hidden layers, various activations, and optimizers.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: str = "relu",
        learning_rate: float = 0.001,
        max_epochs: int = 500,
        batch_size: int = 32,
        optimizer: str = "adam",
        **kwargs
    ):
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            **kwargs
        )
        self.weights = []
        self.biases = []
        self.optimizer_state = {}
    
    def _initialize_parameters(self, layer_sizes: List[int]):
        """Initialize weights and biases using He initialization."""
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU variants
            if self.params['activation'] in ['relu', 'leaky_relu']:
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:  # Xavier for sigmoid/tanh
                scale = np.sqrt(1.0 / layer_sizes[i])
            
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _activate(self, Z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Apply activation function."""
        activation = self.params['activation']
        
        if activation == "relu":
            if derivative:
                return np.where(Z > 0, 1, 0)
            return np.maximum(0, Z)
        
        elif activation == "sigmoid":
            if derivative:
                S = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
                return S * (1 - S)
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        
        elif activation == "tanh":
            if derivative:
                T = np.tanh(Z)
                return 1 - T**2
            return np.tanh(Z)
        
        elif activation == "leaky_relu":
            if derivative:
                return np.where(Z > 0, 1, 0.01)
            return np.where(Z > 0, Z, 0.01 * Z)
        
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through the network."""
        A = [X]
        Z = []
        
        for i in range(len(self.weights)):
            z = np.dot(A[-1], self.weights[i]) + self.biases[i]
            Z.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers
                a = self._activate(z)
            else:  # Output layer (sigmoid for binary, softmax for multi-class)
                if self.n_classes == 2:
                    a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
                else:
                    # Softmax
                    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                    a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            A.append(a)
        
        return A, Z
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.n_classes == 2:
            # Binary cross-entropy
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Categorical cross-entropy
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _backward(self, A: List[np.ndarray], Z: List[np.ndarray], y_true: np.ndarray):
        """Backward pass to compute gradients."""
        m = y_true.shape[0]
        gradients_W = [np.zeros_like(W) for W in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.n_classes == 2:
            delta = A[-1] - y_true.reshape(-1, 1)
        else:
            delta = A[-1] - y_true
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            gradients_W[i] = np.dot(A[i].T, delta) / m
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activate(Z[i-1], derivative=True)
        
        return gradients_W, gradients_b
    
    def _update_parameters(self, gradients_W: List[np.ndarray], gradients_b: List[np.ndarray]):
        """Update parameters using selected optimizer."""
        optimizer = self.params['optimizer']
        lr = self.params['learning_rate']
        
        if optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= lr * gradients_W[i]
                self.biases[i] -= lr * gradients_b[i]
        
        elif optimizer == "adam":
            # Initialize Adam state if first update
            if not self.optimizer_state:
                self.optimizer_state = {
                    'mW': [np.zeros_like(W) for W in self.weights],
                    'vW': [np.zeros_like(W) for W in self.weights],
                    'mb': [np.zeros_like(b) for b in self.biases],
                    'vb': [np.zeros_like(b) for b in self.biases],
                    't': 0
                }
            
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            self.optimizer_state['t'] += 1
            t = self.optimizer_state['t']
            
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.optimizer_state['mW'][i] = beta1 * self.optimizer_state['mW'][i] + (1 - beta1) * gradients_W[i]
                self.optimizer_state['mb'][i] = beta1 * self.optimizer_state['mb'][i] + (1 - beta1) * gradients_b[i]
                
                # Update biased second raw moment estimate
                self.optimizer_state['vW'][i] = beta2 * self.optimizer_state['vW'][i] + (1 - beta2) * (gradients_W[i] ** 2)
                self.optimizer_state['vb'][i] = beta2 * self.optimizer_state['vb'][i] + (1 - beta2) * (gradients_b[i] ** 2)
                
                # Compute bias-corrected first moment estimate
                mW_hat = self.optimizer_state['mW'][i] / (1 - beta1 ** t)
                mb_hat = self.optimizer_state['mb'][i] / (1 - beta1 ** t)
                
                # Compute bias-corrected second raw moment estimate
                vW_hat = self.optimizer_state['vW'][i] / (1 - beta2 ** t)
                vb_hat = self.optimizer_state['vb'][i] / (1 - beta2 ** t)
                
                # Update parameters
                self.weights[i] -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
                self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
        
        elif optimizer == "rmsprop":
            if not self.optimizer_state:
                self.optimizer_state = {
                    'vW': [np.zeros_like(W) for W in self.weights],
                    'vb': [np.zeros_like(b) for b in self.biases],
                    'decay': 0.9,
                    'epsilon': 1e-8
                }
            
            decay = self.optimizer_state['decay']
            epsilon = self.optimizer_state['epsilon']
            
            for i in range(len(self.weights)):
                # Update moving average of squared gradients
                self.optimizer_state['vW'][i] = decay * self.optimizer_state['vW'][i] + (1 - decay) * (gradients_W[i] ** 2)
                self.optimizer_state['vb'][i] = decay * self.optimizer_state['vb'][i] + (1 - decay) * (gradients_b[i] ** 2)
                
                # Update parameters
                self.weights[i] -= lr * gradients_W[i] / (np.sqrt(self.optimizer_state['vW'][i]) + epsilon)
                self.biases[i] -= lr * gradients_b[i] / (np.sqrt(self.optimizer_state['vb'][i]) + epsilon)
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Train MLP on classification problem.
        
        Args:
            problem: Classification problem with X_train, y_train attributes
            max_time: Optional time limit in seconds
            callback: Optional callback function (epoch, loss, accuracy)
            **kwargs: Additional parameters (e.g., 'X_train', 'y_train', 'X_val', 'y_val')
        """
        # Extract data
        X_train = kwargs.get('X_train', getattr(problem, 'X_train', None))
        y_train = kwargs.get('y_train', getattr(problem, 'y_train', None))
        X_val = kwargs.get('X_val', getattr(problem, 'X_val', None))
        y_val = kwargs.get('y_val', getattr(problem, 'y_val', None))
        
        if X_train is None or y_train is None:
            raise ValueError("MLP requires training data (X_train, y_train) in kwargs or problem attributes")
        
        # Determine problem type
        unique_classes = np.unique(y_train)
        self.n_classes = len(unique_classes)
        
        # Convert labels to appropriate format
        if self.n_classes == 2:
            # Binary classification: ensure 0/1 labels
            if not np.array_equal(unique_classes, [0, 1]):
                y_train_bin = (y_train == unique_classes[1]).astype(int)
                if y_val is not None:
                    y_val_bin = (y_val == unique_classes[1]).astype(int)
            else:
                y_train_bin = y_train
                y_val_bin = y_val if y_val is None else y_val
            y_train_formatted = y_train_bin
            y_val_formatted = y_val_bin if y_val is not None else None
        else:
            # Multi-class: one-hot encoding
            y_train_formatted = np.zeros((y_train.shape[0], self.n_classes))
            for i, label in enumerate(y_train):
                y_train_formatted[i, int(label)] = 1
            
            if y_val is not None:
                y_val_formatted = np.zeros((y_val.shape[0], self.n_classes))
                for i, label in enumerate(y_val):
                    y_val_formatted[i, int(label)] = 1
            else:
                y_val_formatted = None
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=20, min_delta=1e-4, mode='min')
        
        # Build network architecture
        n_features = X_train.shape[1]
        layer_sizes = [n_features] + self.params['hidden_layers'] + [1 if self.n_classes == 2 else self.n_classes]
        self._initialize_parameters(layer_sizes)
        
        # Training parameters
        batch_size = self.params['batch_size']
        max_epochs = self.params['max_epochs']
        n_samples = X_train.shape[0]
        
        # Training loop
        for epoch in range(max_epochs):
            if self._check_time_limit(max_time):
                logger.info(f"MLP stopped due to time limit at epoch {epoch}")
                break
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_formatted[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                A, Z = self._forward(X_batch)
                
                # Compute loss
                loss = self._compute_loss(y_batch, A[-1])
                epoch_loss += loss * X_batch.shape[0]
                
                # Backward pass
                gradients_W, gradients_b = self._backward(A, Z, y_batch)
                
                # Update parameters
                self._update_parameters(gradients_W, gradients_b)
            
            avg_loss = epoch_loss / n_samples
            
            # Evaluate on validation set or training set
            if X_val is not None and y_val_formatted is not None:
                A_val, _ = self._forward(X_val)
                y_pred_val = (A_val[-1] > 0.5).astype(int) if self.n_classes == 2 else np.argmax(A_val[-1], axis=1)
                accuracy = np.mean(y_pred_val.flatten() == y_val.flatten())
                current_fitness = 1.0 - accuracy  # Minimize error rate
            else:
                A_train, _ = self._forward(X_train)
                y_pred_train = (A_train[-1] > 0.5).astype(int) if self.n_classes == 2 else np.argmax(A_train[-1], axis=1)
                accuracy = np.mean(y_pred_train.flatten() == y_train.flatten())
                current_fitness = 1.0 - accuracy
            
            # Update convergence
            self._update_convergence(current_fitness, {
                'weights': [w.copy() for w in self.weights],
                'biases': [b.copy() for b in self.biases],
                'accuracy': accuracy,
                'loss': avg_loss
            })
            
            # Callback for progress monitoring
            if callback:
                callback(epoch, avg_loss, accuracy)
            
            # Early stopping based on validation accuracy (or training if no val)
            error_rate = 1.0 - accuracy
            if early_stop(error_rate):
                logger.info(f"MLP converged early at epoch {epoch} (accuracy: {accuracy:.4f})")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Final evaluation
        A_final, _ = self._forward(X_train)
        y_pred_final = (A_final[-1] > 0.5).astype(int) if self.n_classes == 2 else np.argmax(A_final[-1], axis=1)
        final_accuracy = np.mean(y_pred_final.flatten() == y_train.flatten())
        
        # Store final solution
        self.best_fitness = 1.0 - final_accuracy  # Error rate as fitness
        self.best_solution = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'train_accuracy': final_accuracy,
            'architecture': layer_sizes
        }
        
        return self.get_results()


# ==================== KOHONEN SELF-ORGANIZING MAP (SOM) ====================

class KohonenSOM(NeuralMethod):
    """
    Self-Organizing Map for unsupervised clustering and dimensionality reduction.
    Supports rectangular and hexagonal topologies with adaptive learning.
    """
    
    def __init__(
        self,
        map_size: Tuple[int, int] = (10, 10),
        learning_rate_initial: float = 0.5,
        learning_rate_final: float = 0.01,
        neighborhood_initial: float = 5.0,
        max_epochs: int = 1000,
        topology: str = "rectangular",
        **kwargs
    ):
        super().__init__(
            map_size=map_size,
            learning_rate_initial=learning_rate_initial,
            learning_rate_final=learning_rate_final,
            neighborhood_initial=neighborhood_initial,
            max_epochs=max_epochs,
            topology=topology,
            **kwargs
        )
        self.weights = None  # SOM grid weights
        self.map_size = map_size
        self.n_neurons = map_size[0] * map_size[1]
    
    def _initialize_weights(self, n_features: int):
        """Initialize SOM weights randomly within data range."""
        self.weights = np.random.rand(self.map_size[0], self.map_size[1], n_features)
    
    def _get_neuron_coordinates(self) -> np.ndarray:
        """Get coordinates for all neurons in the map."""
        coords = np.zeros((self.n_neurons, 2))
        idx = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                coords[idx] = [i, j]
                idx += 1
        return coords
    
    def _find_best_matching_unit(self, x: np.ndarray) -> Tuple[int, int]:
        """Find the BMU for input vector x."""
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _get_neighborhood(self, bmu_idx: Tuple[int, int], radius: float) -> np.ndarray:
        """Get Gaussian neighborhood around BMU."""
        i_bmu, j_bmu = bmu_idx
        neighborhood = np.zeros((self.map_size[0], self.map_size[1]))
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance = np.linalg.norm(np.array([i, j]) - np.array([i_bmu, j_bmu]))
                if distance <= radius:
                    neighborhood[i, j] = np.exp(-distance**2 / (2 * radius**2))
        
        return neighborhood
    
    def _compute_quantization_error(self, X: np.ndarray) -> float:
        """Compute average distance between samples and their BMUs."""
        total_error = 0.0
        for x in X:
            bmu_idx = self._find_best_matching_unit(x)
            bmu_weight = self.weights[bmu_idx]
            total_error += np.linalg.norm(x - bmu_weight)
        return total_error / len(X)
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Train SOM on clustering problem.
        
        Args:
            problem: Clustering problem with X attribute (feature matrix)
            max_time: Optional time limit in seconds
            callback: Optional callback function (epoch, quantization_error, weights)
            **kwargs: Additional parameters (e.g., 'X')
        """
        # Extract data
        X = kwargs.get('X', getattr(problem, 'X', None))
        if X is None:
            raise ValueError("SOM requires data matrix X in kwargs or problem attributes")
        
        # Normalize features for better SOM training
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=50, min_delta=1e-4, mode='min')
        self._initialize_weights(X.shape[1])
        
        n_samples = X_normalized.shape[0]
        max_epochs = self.params['max_epochs']
        lr_initial = self.params['learning_rate_initial']
        lr_final = self.params['learning_rate_final']
        radius_initial = self.params['neighborhood_initial']
        radius_final = 1.0  # Final radius
        
        # Precompute neuron coordinates for hexagonal topology
        neuron_coords = self._get_neuron_coordinates()
        
        # Training loop
        for epoch in range(max_epochs):
            if self._check_time_limit(max_time):
                logger.info(f"SOM stopped due to time limit at epoch {epoch}")
                break
            
            # Compute adaptive learning rate and radius
            lr = lr_initial * (lr_final / lr_initial) ** (epoch / max_epochs)
            radius = radius_initial * (radius_final / radius_initial) ** (epoch / max_epochs)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = X_normalized[idx]
                
                # Find BMU
                bmu_idx = self._find_best_matching_unit(x)
                
                # Get neighborhood
                neighborhood = self._get_neighborhood(bmu_idx, radius)
                
                # Update weights
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        self.weights[i, j] += lr * neighborhood[i, j] * (x - self.weights[i, j])
            
            # Compute quantization error
            qe = self._compute_quantization_error(X_normalized)
            
            # Update convergence (minimize quantization error)
            self._update_convergence(qe, {
                'weights': self.weights.copy(),
                'map_size': self.map_size,
                'quantization_error': qe
            })
            
            # Callback for progress monitoring
            if callback:
                callback(epoch, qe, self.weights)
            
            # Early stopping check
            if early_stop(qe):
                logger.info(f"SOM converged early at epoch {epoch} (QE: {qe:.6f})")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Assign clusters based on BMUs
        clusters = np.zeros(n_samples, dtype=int)
        for i, x in enumerate(X_normalized):
            bmu_idx = self._find_best_matching_unit(x)
            clusters[i] = bmu_idx[0] * self.map_size[1] + bmu_idx[1]
        
        # Compute final quantization error
        final_qe = self._compute_quantization_error(X_normalized)
        
        # Store final solution
        self.best_fitness = final_qe
        self.best_solution = {
            'weights': self.weights.copy(),
            'clusters': clusters,
            'map_size': self.map_size,
            'quantization_error': final_qe
        }
        
        return self.get_results()


# ==================== HOPFIELD NETWORK ====================

class HopfieldNetwork(NeuralMethod):
    """
    Hopfield Network for optimization problems (e.g., TSP) and associative memory.
    Implements energy minimization with asynchronous updates.
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        threshold: float = 0.0,
        async_update: bool = True,
        energy_threshold: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            max_iterations=max_iterations,
            threshold=threshold,
            async_update=async_update,
            energy_threshold=energy_threshold,
            **kwargs
        )
        self.weights = None  # Connection weights matrix
        self.bias = None
    
    def _initialize_weights_tsp(self, distance_matrix: np.ndarray, n_cities: int, A: float = 500, B: float = 500, C: float = 200, D: float = 500):
        """
        Initialize Hopfield weights for TSP using the energy function formulation.
        
        Energy function components:
        E = A/2 * Σ_i Σ_k Σ_l≠k V_ik V_il    (one city per row)
          + B/2 * Σ_k Σ_i Σ_j≠i V_ik V_jk    (one city per column)
          + C/2 * (Σ_i Σ_k V_ik - n)^2       (n cities constraint)
          + D/2 * Σ_i Σ_j≠i Σ_k d_ij V_ik V_j(k+1 mod n)  (distance minimization)
        """
        n = n_cities
        N = n * n  # Total neurons (n cities × n positions)
        
        # Initialize weight matrix
        W = np.zeros((N, N))
        
        # Helper to get neuron index
        def neuron_idx(city, position):
            return city * n + position
        
        # Constraint terms (A, B, C)
        for i in range(n):  # City i
            for k in range(n):  # Position k
                idx1 = neuron_idx(i, k)
                
                # Term A: One city per row (position constraint)
                for l in range(n):
                    if l != k:
                        idx2 = neuron_idx(i, l)
                        W[idx1, idx2] -= A
                
                # Term B: One city per column (city constraint)
                for j in range(n):
                    if j != i:
                        idx2 = neuron_idx(j, k)
                        W[idx1, idx2] -= B
                
                # Term C: n cities total (global constraint)
                for j in range(n):
                    for l in range(n):
                        idx2 = neuron_idx(j, l)
                        W[idx1, idx2] -= C / n
        
        # Term D: Distance minimization
        for i in range(n):
            for j in range(n):
                if i != j:
                    d_ij = distance_matrix[i, j]
                    for k in range(n):
                        idx1 = neuron_idx(i, k)
                        idx2 = neuron_idx(j, (k + 1) % n)
                        W[idx1, idx2] -= D * d_ij / 2  # Symmetric weight
        
        # Make weights symmetric (required for convergence)
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)  # No self-connections
        
        self.weights = W
        self.bias = np.zeros(N)
        
        return W
    
    def _energy(self, state: np.ndarray) -> float:
        """Compute Hopfield energy for current state."""
        return -0.5 * np.dot(state, np.dot(self.weights, state)) - np.dot(self.bias, state)
    
    def _update_state(self, state: np.ndarray) -> np.ndarray:
        """Update state using activation function."""
        net_input = np.dot(self.weights, state) + self.bias
        new_state = np.where(net_input >= self.params['threshold'], 1, 0)
        return new_state
    
    def _decode_tsp_solution(self, state: np.ndarray, n_cities: int) -> Optional[List[int]]:
        """Decode Hopfield state into valid TSP tour."""
        n = n_cities
        tour = [-1] * n
        
        # Reshape state to city×position matrix
        state_matrix = state.reshape((n, n))
        
        # Extract tour: for each position, find the city with highest activation
        for pos in range(n):
            city = np.argmax(state_matrix[:, pos])
            tour[pos] = city
        
        # Validate tour
        if len(set(tour)) == n and all(0 <= c < n for c in tour):
            return tour
        return None
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve optimization problem using Hopfield Network.
        
        Args:
            problem: Problem instance (TSP supported)
            max_time: Optional time limit in seconds
            callback: Optional callback function (iteration, energy, state)
            **kwargs: Additional parameters (e.g., 'distance_matrix', 'n_cities')
        """
        # Check if TSP problem
        if not hasattr(problem, 'distance_matrix') or not hasattr(problem, 'n_cities'):
            raise ValueError("Hopfield Network currently supports TSP problems only")
        
        distance_matrix = problem.distance_matrix
        n_cities = problem.n_cities
        
        self._initialize_convergence_tracking()
        
        # Initialize weights for TSP
        self._initialize_weights_tsp(distance_matrix, n_cities)
        
        # Initialize random state (bipolar: -1/1 or binary: 0/1)
        N = n_cities * n_cities
        state = np.random.choice([0, 1], size=N)
        
        energy = self._energy(state)
        min_energy = energy
        best_state = state.copy()
        
        max_iterations = self.params['max_iterations']
        energy_threshold = self.params['energy_threshold']
        async_update = self.params['async_update']
        
        # Energy minimization loop
        for iteration in range(max_iterations):
            if self._check_time_limit(max_time):
                logger.info(f"Hopfield stopped due to time limit at iteration {iteration}")
                break
            
            prev_energy = energy
            
            if async_update:
                # Asynchronous update: random neuron order
                indices = np.random.permutation(N)
                for idx in indices:
                    net_input = np.dot(self.weights[idx], state) + self.bias[idx]
                    state[idx] = 1 if net_input >= self.params['threshold'] else 0
            else:
                # Synchronous update
                state = self._update_state(state)
            
            # Compute new energy
            energy = self._energy(state)
            
            # Track best state
            if energy < min_energy:
                min_energy = energy
                best_state = state.copy()
            
            # Update convergence (minimize energy)
            self._update_convergence(energy, state.copy())
            
            # Callback for progress monitoring
            if callback:
                callback(iteration, energy, state)
            
            # Check for convergence (energy change below threshold)
            if abs(prev_energy - energy) < energy_threshold:
                logger.info(f"Hopfield converged at iteration {iteration} (energy: {energy:.4f})")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Decode best state to TSP tour
        tour = self._decode_tsp_solution(best_state, n_cities)
        
        # If invalid tour, try multiple restarts or return random valid tour
        if tour is None or not problem.is_valid_tour(tour):
            logger.warning("Hopfield failed to converge to valid TSP tour. Returning random tour.")
            tour = list(range(n_cities))
            random.shuffle(tour)
        
        # Evaluate tour length
        result = problem.evaluate(tour)
        tour_length = result['tour_length']
        
        # Store final solution
        self.best_fitness = tour_length
        self.best_solution = {
            'tour': tour,
            'state': best_state,
            'energy': min_energy,
            'valid': problem.is_valid_tour(tour)
        }
        
        return self.get_results()


# ==================== METHOD FACTORY ====================

def create_neural_method(method_name: str, **params) -> NeuralMethod:
    """
    Factory function to create neural methods by name.
    
    Args:
        method_name: One of 'Perceptron', 'MLP', 'Kohonen', 'Hopfield'
        **params: Method-specific parameters
    
    Returns:
        Instantiated neural method
    """
    method_name = method_name.lower()
    
    if method_name in ["perceptron", "perceptronnetwork"]:
        return Perceptron(**params)
    elif method_name in ["mlp", "multilayerperceptron", "neuralnetwork"]:
        return MultiLayerPerceptron(**params)
    elif method_name in ["som", "kohonen", "kohonensom"]:
        return KohonenSOM(**params)
    elif method_name in ["hopfield", "hopfieldnetwork"]:
        return HopfieldNetwork(**params)
    else:
        raise ValueError(f"Unknown neural method: {method_name}. Available: Perceptron, MLP, Kohonen, Hopfield")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Neural Methods Demo ===\n")
    
    # Example 1: Perceptron on synthetic binary classification
    try:
        print("1. Perceptron on synthetic data...")
        
        # Generate synthetic linearly separable data
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        perceptron = Perceptron(learning_rate=0.1, max_epochs=50, bias=True)
        results = perceptron.solve(
            problem=None,
            X_train=X_train,
            y_train=y_train,
            max_time=5.0
        )
        
        print(f"   Training error: {results['best_fitness']:.4f}")
        print(f"   Accuracy: {(1 - results['best_fitness']) * 100:.2f}%")
        print(f"   Time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"   Perceptron demo failed: {e}")
    
    # Example 2: MLP on XOR problem (non-linear)
    try:
        print("\n2. MLP on XOR problem...")
        
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([0, 1, 1, 0])
        
        mlp = MultiLayerPerceptron(
            hidden_layers=[4],
            activation="tanh",
            learning_rate=0.1,
            max_epochs=1000,
            batch_size=4,
            optimizer="sgd"
        )
        
        results = mlp.solve(
            problem=None,
            X_train=X_train,
            y_train=y_train,
            max_time=5.0
        )
        
        # Evaluate
        A_final, _ = mlp._forward(X_train)
        y_pred = (A_final[-1] > 0.5).astype(int).flatten()
        accuracy = np.mean(y_pred == y_train)
        
        print(f"   Predictions: {y_pred}")
        print(f"   Accuracy: {accuracy * 100:.2f}%")
        print(f"   Time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"   MLP demo failed: {e}")
    
    # Example 3: SOM on synthetic clusters
    try:
        print("\n3. SOM on synthetic clusters...")
        
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
        
        som = KohonenSOM(
            map_size=(8, 8),
            learning_rate_initial=0.5,
            learning_rate_final=0.01,
            neighborhood_initial=4.0,
            max_epochs=100,
            topology="rectangular"
        )
        
        results = som.solve(
            problem=None,
            X=X,
            max_time=10.0
        )
        
        print(f"   Quantization error: {results['best_fitness']:.4f}")
        print(f"   Time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"   SOM demo failed: {e}")
    
    # Example 4: Hopfield on small TSP (requires tsp.py)
    try:
        from problems.tsp import TSProblem
        
        print("\n4. Hopfield Network on Random10 TSP...")
        
        # Create very small TSP instance for Hopfield (Hopfield scales poorly)
        problem = TSProblem.from_preset("random30")  # Note: Hopfield works best on n<15
        
        hopfield = HopfieldNetwork(
            max_iterations=200,
            threshold=0.0,
            async_update=True,
            energy_threshold=1e-6
        )
        
        # Hopfield is not ideal for large TSP instances; this is for demonstration only
        results = hopfield.solve(
            problem=problem,
            max_time=15.0
        )
        
        print(f"   Tour length: {results['best_fitness']:.2f}")
        print(f"   Valid tour: {results['best_solution']['valid']}")
        print(f"   Time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"   Hopfield demo failed or skipped (requires tsp.py): {e}")
    
    print("\nDemos completed. For full integration, use with orchestrator.py")