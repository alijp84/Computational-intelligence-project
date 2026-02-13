"""
Utility functions for MetaMind CI Framework
Provides helper functions for data loading, evaluation metrics, statistical analysis,
visualization, and LLM communication.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime

# ==================== LOGGING SETUP ====================

def setup_logger(name: str = "metamind", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==================== DATA LOADING & PREPROCESSING ====================

def load_tsp_distance_matrix(filepath: str) -> np.ndarray:
    """Load TSP distance matrix from TSPLIB format or CSV."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, header=None).values
    else:
        # Basic TSPLIB parser (supports EDGE_WEIGHT_TYPE: EUC_2D, FULL_MATRIX)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find start of matrix data
        start_idx = None
        dimension = None
        edge_weight_type = None
        
        for i, line in enumerate(lines):
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("EDGE_WEIGHT_SECTION") or line.startswith("1 "):
                start_idx = i if line.startswith("EDGE_WEIGHT_SECTION") else i - 1
                break
        
        if dimension is None:
            raise ValueError("DIMENSION not found in TSP file")
        
        # Read matrix data
        matrix_lines = [line.strip() for line in lines[start_idx+1:] if line.strip() and not line.startswith("EOF")]
        flat_data = []
        for line in matrix_lines:
            flat_data.extend([float(x) for x in line.split()])
        
        if edge_weight_type == "FULL_MATRIX":
            return np.array(flat_data).reshape(dimension, dimension)
        else:
            raise NotImplementedError(f"Edge weight type {edge_weight_type} not supported")


def preprocess_titanic_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess Titanic dataset for classification."""
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Drop high-missing columns
    df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # Separate features and target
    if 'Survived' in df.columns:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
    else:
        X = df
        y = None
    
    # Normalize numerical features
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    for col in numerical_cols:
        if col in X.columns:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
    
    return X, y


def generate_synthetic_clusters(
    n_samples: int = 500,
    n_features: int = 2,
    n_clusters: int = 5,
    cluster_std: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustering data using sklearn's make_blobs."""
    from sklearn.datasets import make_blobs
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )


# ==================== EVALUATION METRICS ====================

# Optimization/TSP Metrics
def tsp_tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total length of TSP tour."""
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total


def gap_to_optimal(found: float, optimal: float) -> float:
    """Calculate percentage gap to known optimum."""
    return abs(found - optimal) / optimal * 100.0 if optimal != 0 else float('inf')


def two_opt_improvement(tour: List[int], distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """Apply 2-opt local search to improve TSP tour."""
    n = len(tour)
    best_tour = tour[:]
    best_distance = tsp_tour_length(best_tour, distance_matrix)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                new_distance = tsp_tour_length(new_tour, distance_matrix)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return best_tour, best_distance


# Classification Metrics
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])


# Clustering Metrics
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Simplified silhouette score calculation."""
    from sklearn.metrics import silhouette_score as sk_silhouette
    return sk_silhouette(X, labels) if len(np.unique(labels)) > 1 else -1.0


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Davies-Bouldin index (lower is better)."""
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float('inf')


def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Calinski-Harabasz index (higher is better)."""
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0.0


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Calculate Adjusted Rand Index when true labels available."""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_true, labels_pred)


# ==================== STATISTICAL ANALYSIS ====================

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, and confidence interval."""
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    
    return {
        'mean': mean,
        'std': std,
        'min': np.min(arr),
        'max': np.max(arr),
        'ci_95': ci,
        'n': n
    }


def wilcoxon_test(sample1: List[float], sample2: List[float]) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test."""
    stat, p_value = stats.wilcoxon(sample1, sample2, zero_method='wilcox', correction=False)
    return {'statistic': stat, 'p_value': p_value}


def success_rate(values: List[float], threshold: float) -> float:
    """Calculate percentage of runs meeting threshold criterion."""
    return np.mean(np.array(values) <= threshold) * 100.0


# ==================== VISUALIZATION ====================

def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Convergence Curves",
    xlabel: str = "Iteration",
    ylabel: str = "Fitness",
    save_path: Optional[str] = None
):
    """Plot convergence curves for multiple methods."""
    plt.figure(figsize=(10, 6))
    
    for method, history in histories.items():
        if isinstance(history[0], list):  # Multiple runs
            history = np.array(history)
            mean_hist = np.mean(history, axis=0)
            std_hist = np.std(history, axis=0)
            iterations = np.arange(len(mean_hist))
            plt.plot(iterations, mean_hist, label=method, linewidth=2)
            plt.fill_between(iterations, mean_hist - std_hist, mean_hist + std_hist, alpha=0.3)
        else:  # Single run
            plt.plot(history, label=method, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Method Comparison"
):
    """Create heatmap-style comparison table."""
    methods = list(results.keys())
    data = [[results[method].get(metric, np.nan) for metric in metrics] for method in methods]
    
    fig, ax = plt.subplots(figsize=(12, len(methods) * 0.5))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    
    # Add values to cells
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = data[i][j]
            if not np.isnan(val):
                text = f"{val:.2f}" if isinstance(val, float) else str(val)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# ==================== LLM COMMUNICATION ====================

def format_problem_input(
    problem_type: str,
    problem_data: Dict,
    constraints: Optional[Dict] = None,
    preferences: Optional[Dict] = None
) -> str:
    """Format problem description for LLM input."""
    constraints = constraints or {}
    preferences = preferences or {}
    
    prompt = f"""Problem Type: {problem_type}
    
Problem Data:
"""
    for key, value in problem_data.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 10:
            prompt += f"  {key}: Array of shape {np.array(value).shape} (first 5: {value[:5]})\n"
        else:
            prompt += f"  {key}: {value}\n"
    
    if constraints:
        prompt += "\nConstraints:\n"
        for key, value in constraints.items():
            prompt += f"  {key}: {value}\n"
    
    if preferences:
        prompt += "\nPreferences:\n"
        for key, value in preferences.items():
            prompt += f"  {key}: {value}\n"
    
    prompt += "\nPlease analyze this problem and recommend the most suitable CI method with parameters."
    return prompt


def parse_llm_method_selection(llm_response: Union[str, Dict]) -> Dict:
    """Parse LLM's method selection response into structured format."""
    if isinstance(llm_response, str):
        try:
            # Try to extract JSON from text response
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            if start != -1 and end != -1:
                llm_response = json.loads(llm_response[start:end])
            else:
                raise ValueError("No JSON found in LLM response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    # Validate required fields
    required_fields = ['selected_method', 'parameters']
    for field in required_fields:
        if field not in llm_response:
            raise ValueError(f"LLM response missing required field: {field}")
    
    return {
        'method': llm_response['selected_method'],
        'parameters': llm_response['parameters'],
        'reasoning': llm_response.get('reasoning', ''),
        'backup_method': llm_response.get('backup_method'),
        'confidence': llm_response.get('confidence', 0.5)
    }


def format_execution_results_for_llm(
    method_used: str,
    best_solution: Union[List, np.ndarray],
    best_fitness: float,
    execution_time: float,
    convergence_history: List[float],
    known_optimal: Optional[float] = None,
    iterations_completed: Optional[int] = None
) -> Dict:
    """Format execution results for LLM feedback analysis."""
    result = {
        "method_used": method_used,
        "best_solution": best_solution[:10] if len(best_solution) > 10 else best_solution,  # Truncate for LLM
        "best_fitness": float(best_fitness),
        "computation_time": float(execution_time),
        "convergence_history": convergence_history[-10:],  # Last 10 iterations
        "iterations_completed": iterations_completed or len(convergence_history)
    }
    
    if known_optimal is not None:
        result["known_optimal"] = float(known_optimal)
        result["gap_percentage"] = float(gap_to_optimal(best_fitness, known_optimal))
    
    return result


# ==================== GENERAL UTILITIES ====================

def validate_parameters(params: Dict, param_ranges: Dict) -> Tuple[bool, List[str]]:
    """Validate method parameters against allowed ranges/types."""
    errors = []
    
    for param_name, expected in param_ranges.items():
        if param_name not in params:
            errors.append(f"Missing parameter: {param_name}")
            continue
        
        value = params[param_name]
        
        # Type checking
        if 'type' in expected:
            expected_type = expected['type']
            if not isinstance(value, expected_type):
                errors.append(f"Parameter {param_name} should be {expected_type.__name__}, got {type(value).__name__}")
                continue
        
        # Range checking for numeric types
        if isinstance(value, (int, float)):
            if 'min' in expected and value < expected['min']:
                errors.append(f"Parameter {param_name}={value} below minimum {expected['min']}")
            if 'max' in expected and value > expected['max']:
                errors.append(f"Parameter {param_name}={value} above maximum {expected['max']}")
        
        # Options checking
        if 'options' in expected and value not in expected['options']:
            errors.append(f"Parameter {param_name}={value} not in allowed options {expected['options']}")
    
    return len(errors) == 0, errors


def save_results(results: Dict, filepath: str):
    """Save experiment results to JSON file with timestamp."""
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val < 1e-8:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


# ==================== EARLY STOPPING ====================

class EarlyStopping:
    """Monitor convergence and trigger early stopping."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improvement = self.best_value - value > self.min_delta
        else:  # mode == 'max'
            improvement = value - self.best_value > self.min_delta
        
        if improvement:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.best_value = None
        self.early_stop = False