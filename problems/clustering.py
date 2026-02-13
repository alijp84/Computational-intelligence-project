"""
Clustering Problem Implementation for MetaMind CI Framework
Provides standardized interface for unsupervised clustering with internal/external validation.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from copy import deepcopy
import time
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    setup_logger,
    compute_statistics,
    success_rate,
    silhouette_score,
    davies_bouldin_index,
    calinski_harabasz_index,
    adjusted_rand_index
)

logger = setup_logger("clustering_problem")


# ==================== DATASET LOADING UTILITIES ====================

def load_iris_dataset() -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Load and preprocess Iris dataset for clustering validation.
    
    Returns:
        Dictionary with features, true labels, and metadata
    """
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Normalize features (important for distance-based clustering)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    logger.info(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} clusters")
    
    return {
        'X': X_normalized,
        'X_raw': X,
        'y_true': y,
        'feature_names': feature_names,
        'target_names': target_names,
        'n_clusters': len(np.unique(y)),
        'dataset_name': 'Iris',
        'has_true_labels': True,
        'scaler': scaler
    }


def load_mall_customers_dataset(filepath: Optional[str] = None) -> Dict[str, Union[np.ndarray, List[str]]]:
    """
    Load and preprocess Mall Customer Segmentation dataset.
    
    Args:
        filepath: Path to Mall_Customers.csv (if None, tries default locations)
    
    Returns:
        Dictionary with features and metadata
    """
    # Locate dataset file
    if filepath is None:
        possible_paths = [
            "data/mall/Mall_Customers.csv",
            "data/Mall_Customers.csv",
            "mall/Mall_Customers.csv",
            "Mall_Customers.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        else:
            raise FileNotFoundError(
                "Mall Customers dataset not found. Please download from Kaggle and place in data/mall/Mall_Customers.csv"
            )
    
    # Load data
    df = pd.read_csv(filepath)
    logger.info(f"Loaded Mall Customers dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Select relevant features for clustering (Age, Annual Income, Spending Score)
    # Note: CustomerID and Gender are excluded from clustering features
    # Gender could be one-hot encoded if needed, but typically not used for customer segmentation clustering
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        # Try alternative column names
        alt_names = {
            'Annual Income (k$)': ['Annual Income', 'Income'],
            'Spending Score (1-100)': ['Spending Score', 'Score']
        }
        for i, f in enumerate(features):
            if f in missing_features:
                for alt in alt_names.get(f, []):
                    if alt in df.columns:
                        features[i] = alt
                        missing_features.remove(f)
                        break
    
    if missing_features:
        raise ValueError(f"Required features missing: {missing_features}. Available columns: {df.columns.tolist()}")
    
    X = df[features].values.astype(np.float32)
    
    # Normalize features (critical for clustering with different scales)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Store metadata
    feature_names = features
    customer_ids = df['CustomerID'].values if 'CustomerID' in df.columns else np.arange(len(df))
    
    logger.info(f"Preprocessed Mall Customers: {X.shape[0]} samples, {X.shape[1]} features ({', '.join(feature_names)})")
    
    return {
        'X': X_normalized,
        'X_raw': X,
        'customer_ids': customer_ids,
        'feature_names': feature_names,
        'n_samples': X.shape[0],
        'dataset_name': 'Mall Customers',
        'has_true_labels': False,  # No ground truth clusters
        'scaler': scaler,
        'raw_df': df
    }


def generate_synthetic_clusters(
    n_samples: int = 500,
    n_features: int = 2,
    n_clusters: int = 5,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Generate synthetic clustering data using sklearn's make_blobs.
    
    Args:
        n_samples: Total number of points
        n_features: Number of features/dimensions
        n_clusters: Number of clusters to generate
        cluster_std: Standard deviation of clusters (lower = tighter clusters)
        center_box: Range for cluster center locations
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle samples after generation
    
    Returns:
        Dictionary with features, true labels, and metadata
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state,
        shuffle=shuffle
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.astype(np.float32))
    
    logger.info(
        f"Generated synthetic clusters: {n_samples} samples, {n_features}D, "
        f"{n_clusters} clusters, std={cluster_std}"
    )
    
    return {
        'X': X_normalized,
        'X_raw': X.astype(np.float32),
        'y_true': y.astype(np.int32),
        'n_clusters': n_clusters,
        'n_features': n_features,
        'cluster_std': cluster_std,
        'dataset_name': f'Synthetic_{n_samples}x{n_features}x{n_clusters}',
        'has_true_labels': True,
        'scaler': scaler,
        'random_state': random_state
    }


# ==================== BASE CLUSTERING PROBLEM CLASS ====================

class ClusteringProblem:
    """
    Standardized clustering problem interface for MetaMind orchestrator.
    Supports internal validation (Silhouette, DB, CH) and external validation (ARI, NMI) when labels available.
    """
    
    def __init__(
        self,
        dataset_name: str,
        n_clusters: Optional[int] = None,
        data_path: Optional[str] = None,
        random_state: int = 42,
        normalize: bool = True
    ):
        self.dataset_name = dataset_name.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.normalize = normalize
        
        # Load dataset
        if "iris" in self.dataset_name:
            self.data = load_iris_dataset()
            self.n_clusters = self.n_clusters or self.data['n_clusters']
        
        elif "mall" in self.dataset_name or "customer" in self.dataset_name:
            self.data = load_mall_customers_dataset(filepath=data_path)
            # Default to 5 clusters for Mall Customers (typical customer segments)
            self.n_clusters = self.n_clusters or 5
        
        elif "synthetic" in self.dataset_name:
            # Parse parameters from dataset_name if provided (e.g., "synthetic_500_5_1.0")
            params = self.dataset_name.replace("synthetic", "").strip("_").split("_")
            gen_n_samples = int(params[0]) if len(params) > 0 and params[0].isdigit() else 500
            gen_n_clusters = int(params[1]) if len(params) > 1 and params[1].isdigit() else 5
            gen_cluster_std = float(params[2]) if len(params) > 2 and params[2].replace('.', '', 1).isdigit() else 1.0
            
            self.data = generate_synthetic_clusters(
                n_samples=gen_n_samples,
                n_features=2,  # Default 2D for visualization
                n_clusters=gen_n_clusters,
                cluster_std=gen_cluster_std,
                random_state=random_state
            )
            self.n_clusters = self.n_clusters or gen_n_clusters
        
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Supported: 'iris', 'mall_customers', 'synthetic'"
            )
        
        # Extract data
        self.X = self.data['X']
        self.X_raw = self.data['X_raw']
        self.y_true = self.data.get('y_true')
        self.has_true_labels = self.data['has_true_labels']
        self.feature_names = self.data.get('feature_names', [f"feature_{i}" for i in range(self.X.shape[1])])
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.scaler = self.data['scaler']
        
        # Evaluation tracking
        self.cluster_assignments = None
        self.evaluation_history = []
        self.start_time = None
        
        logger.info(f"Initialized {self.dataset_name} clustering problem")
        logger.info(f"  Samples: {self.n_samples}, Features: {self.n_features}, Target clusters: {self.n_clusters}")
        if self.has_true_labels:
            logger.info(f"  True clusters available: {len(np.unique(self.y_true))}")
    
    def evaluate(
        self,
        labels: Optional[np.ndarray] = None,
        model: Optional[object] = None,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Evaluate clustering quality using internal and external validation metrics.
        
        Args:
            labels: Cluster assignments array (shape: [n_samples])
            model: Trained clustering model with predict() or transform() method
            n_clusters: Number of clusters (if not provided, inferred from labels)
            **kwargs: Additional parameters (e.g., 'X' for custom data)
        
        Returns:
            Dictionary with comprehensive clustering metrics:
            {
                'silhouette': float,          # [-1, 1], higher is better
                'davies_bouldin': float,      # Lower is better
                'calinski_harabasz': float,   # Higher is better
                'adjusted_rand': float,       # [0, 1], higher is better (if labels available)
                'normalized_mutual_info': float, # [0, 1], higher is better (if labels available)
                'inertia': float,             # Within-cluster sum of squares (for k-means like)
                'n_clusters': int,
                'cluster_sizes': List[int],
                'labels': np.ndarray
            }
        """
        # Get cluster assignments
        if labels is None:
            if model is None:
                raise ValueError("Either labels or model must be provided")
            
            # Handle different model types
            if hasattr(model, 'predict'):
                labels = model.predict(self.X)
            elif hasattr(model, 'labels_'):  # Scikit-learn fitted model
                labels = model.labels_
            elif hasattr(model, 'best_solution') and 'clusters' in model.best_solution:
                # Kohonen SOM solution
                labels = model.best_solution['clusters']
            elif isinstance(model, dict) and 'clusters' in model:
                labels = model['clusters']
            else:
                raise ValueError(f"Unsupported model type for clustering: {type(model)}")
        
        # Ensure proper shape and type
        labels = np.array(labels).flatten().astype(np.int32)
        
        if len(labels) != self.n_samples:
            raise ValueError(f"Labels length ({len(labels)}) != samples ({self.n_samples})")
        
        # Determine number of clusters
        unique_labels = np.unique(labels)
        n_clusters_actual = len(unique_labels)
        
        # Handle noise points (label = -1 in DBSCAN-like algorithms)
        if -1 in unique_labels:
            n_noise = np.sum(labels == -1)
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]
            valid_X = self.X[valid_mask]
            logger.debug(f"Detected {n_noise} noise points (label=-1). Evaluating on {len(valid_X)} core points.")
        else:
            valid_labels = labels
            valid_X = self.X
            n_noise = 0
        
        # Compute internal validation metrics (require at least 2 clusters)
        metrics = {
            'n_clusters': n_clusters_actual,
            'n_noise': int(n_noise),
            'cluster_sizes': [int(np.sum(labels == i)) for i in unique_labels],
            'labels': labels
        }
        
        # Silhouette score (requires at least 2 clusters and n_samples > n_clusters)
        try:
            if n_clusters_actual >= 2 and self.n_samples > n_clusters_actual:
                metrics['silhouette'] = float(sklearn_silhouette(valid_X, valid_labels))
            else:
                metrics['silhouette'] = -1.0  # Invalid case
        except Exception as e:
            logger.warning(f"Silhouette computation failed: {e}")
            metrics['silhouette'] = -1.0
        
        # Davies-Bouldin index (lower is better)
        try:
            if n_clusters_actual >= 2:
                metrics['davies_bouldin'] = float(davies_bouldin_score(valid_X, valid_labels))
            else:
                metrics['davies_bouldin'] = float('inf')
        except Exception as e:
            logger.warning(f"Davies-Bouldin computation failed: {e}")
            metrics['davies_bouldin'] = float('inf')
        
        # Calinski-Harabasz index (higher is better)
        try:
            if n_clusters_actual >= 2:
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(valid_X, valid_labels))
            else:
                metrics['calinski_harabasz'] = 0.0
        except Exception as e:
            logger.warning(f"Calinski-Harabasz computation failed: {e}")
            metrics['calinski_harabasz'] = 0.0
        
        # Inertia (within-cluster sum of squares) - approximate if not available
        try:
            if hasattr(model, 'inertia_'):
                metrics['inertia'] = float(model.inertia_)
            elif hasattr(model, 'best_solution') and 'quantization_error' in model.best_solution:
                # SOM quantization error as proxy
                metrics['inertia'] = float(model.best_solution['quantization_error'])
            else:
                # Compute manually
                inertia = 0.0
                for i in unique_labels:
                    if i == -1:
                        continue
                    cluster_points = valid_X[valid_labels == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        inertia += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
                metrics['inertia'] = float(inertia)
        except Exception as e:
            logger.warning(f"Inertia computation failed: {e}")
            metrics['inertia'] = float('inf')
        
        # External validation metrics (if true labels available)
        if self.has_true_labels and self.y_true is not None:
            # Align label numbering for fair comparison (using Hungarian algorithm approximation)
            try:
                from scipy.optimize import linear_sum_assignment
                
                # Build contingency matrix
                n_true = len(np.unique(self.y_true))
                n_pred = n_clusters_actual
                contingency = np.zeros((n_true, n_pred))
                
                for i in range(self.n_samples):
                    if labels[i] != -1:  # Skip noise points
                        contingency[self.y_true[i], labels[i]] += 1
                
                # Optimal label mapping
                row_ind, col_ind = linear_sum_assignment(-contingency)
                label_map = {col: row for row, col in zip(row_ind, col_ind)}
                
                # Map predicted labels to true labels
                aligned_labels = np.array([label_map.get(l, -1) for l in labels])
                
                # Compute metrics on aligned labels (excluding noise)
                valid_mask = aligned_labels != -1
                metrics['adjusted_rand'] = float(adjusted_rand_score(
                    self.y_true[valid_mask], 
                    aligned_labels[valid_mask]
                ))
                metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(
                    self.y_true[valid_mask], 
                    aligned_labels[valid_mask]
                ))
            except Exception as e:
                logger.warning(f"External validation failed (using direct comparison): {e}")
                # Fallback to direct comparison (less accurate but robust)
                metrics['adjusted_rand'] = float(adjusted_rand_score(self.y_true, labels))
                metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(self.y_true, labels))
        else:
            metrics['adjusted_rand'] = None
            metrics['normalized_mutual_info'] = None
        
        # Store evaluation
        evaluation = {
            'timestamp': time.time(),
            'metrics': metrics,
            'n_clusters_requested': n_clusters or self.n_clusters,
            'n_clusters_actual': n_clusters_actual
        }
        
        self.evaluation_history.append(evaluation)
        self.cluster_assignments = labels
        
        return metrics
    
    def reset(self):
        """Reset evaluation history for new experiment."""
        self.evaluation_history = []
        self.cluster_assignments = None
        self.start_time = time.time()
    
    def get_problem_description(self) -> Dict:
        """Return structured problem description for LLM input."""
        return {
            "problem_type": "unsupervised_clustering",
            "domain": "customer_segmentation_or_pattern_discovery",
            "name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "target_clusters": self.n_clusters,
            "has_true_labels": self.has_true_labels,
            "characteristics": self._get_characteristics(),
            "evaluation_approach": (
                "Internal validation (Silhouette, DB, CH) + External validation (ARI, NMI) when labels available"
            )
        }
    
    def _get_characteristics(self) -> str:
        """Return dataset characteristics for LLM context."""
        if "iris" in self.dataset_name:
            return "Small benchmark dataset (150 samples) with 4 features and 3 known species; well-separated clusters"
        elif "mall" in self.dataset_name:
            return "Customer segmentation data (200 samples) with Age, Income, Spending Score; natural customer groups expected"
        elif "synthetic" in self.dataset_name:
            return f"Controlled synthetic data with {self.n_clusters} clusters; adjustable separation via cluster_std parameter"
        else:
            return "Clustering problem with mixed characteristics"
    
    def get_llm_problem_prompt(self, preferences: Optional[Dict] = None) -> str:
        """
        Generate formatted problem description for LLM input (Step 1 in project doc).
        
        Args:
            preferences: Optional dict with keys like 'time_limit', 'priority'
        
        Returns:
            Formatted string prompt for LLM
        """
        desc = self.get_problem_description()
        preferences = preferences or {}
        
        prompt = f"""Problem: Unsupervised Clustering
Dataset: {desc['name'].replace('_', ' ').title()}
Samples: {desc['n_samples']}
Features: {desc['n_features']} ({', '.join(desc['feature_names'][:3]) + ('...' if len(desc['feature_names']) > 3 else '')})
Target Clusters: {desc['target_clusters']}

Problem Characteristics:
- {desc['characteristics']}
- True labels: {'Available for validation' if desc['has_true_labels'] else 'Not available (purely unsupervised)'}
- Primary challenge: Discover natural groupings without supervision

Evaluation Metrics:
- Primary (always): Silhouette Score (higher = better separated clusters)
- Secondary (always): Davies-Bouldin Index (lower = better), Calinski-Harabasz Index (higher = better)
- External validation (if labels available): Adjusted Rand Index, Normalized Mutual Information
- Supplementary: Inertia (within-cluster sum of squares)

Available CI Methods:
- Kohonen SOM: Self-Organizing Map - excels at topology preservation and visual cluster discovery
- K-Means (via GA optimization): Genetic Algorithm can optimize cluster centroids
- Fuzzy C-Means (conceptual): Not directly implemented but fuzzy controller could adapt
- Hierarchical approaches: Not directly available in current method set

Task:
Analyze this clustering problem and recommend the most appropriate CI method with justified parameter configuration.
Consider dataset size, feature dimensionality, expected cluster shapes, and availability of validation labels.
For SOM: recommend map_size based on n_samples and n_clusters (rule of thumb: sqrt(5*sqrt(n_samples))).
"""
        
        if preferences:
            prompt += "\nUser Preferences:\n"
            for key, value in preferences.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt += """
Note: Clustering is unsupervised - no "optimal" solution exists. Quality measured by internal cohesion/separation metrics.
Higher Silhouette Score indicates better clustering (target > 0.5 for meaningful clusters).
"""
        
        return prompt
    
    def format_for_llm_feedback(self, execution_results: Dict) -> Dict:
        """
        Format execution results for LLM interpretation (Step 5 in project doc).
        
        Args:
            execution_results: Dictionary from method execution containing:
                - method_used
                - best_solution (cluster assignments or SOM weights)
                - best_fitness (quantization error or 1 - silhouette)
                - computation_time
                - convergence_history
                - iterations_completed (optional)
        
        Returns:
            Formatted dictionary matching project specification
        """
        # Extract metrics from latest evaluation
        if self.evaluation_history:
            metrics = self.evaluation_history[-1]['metrics']
            silhouette = metrics.get('silhouette', -1.0)
            db_index = metrics.get('davies_bouldin', float('inf'))
            ch_index = metrics.get('calinski_harabasz', 0.0)
            ari = metrics.get('adjusted_rand', None)
            nmi = metrics.get('normalized_mutual_info', None)
            n_clusters = metrics['n_clusters']
        else:
            # Fallback estimates based on fitness
            fitness = execution_results["best_fitness"]
            silhouette = max(0.0, 1.0 - fitness) if fitness < 2.0 else -0.5  # Heuristic
            db_index = fitness if fitness > 0 else 1.0
            ch_index = 1000.0 / (fitness + 1e-6) if fitness > 0 else 0.0
            ari = nmi = None
            n_clusters = self.n_clusters
        
        formatted = {
            "method_used": execution_results["method_used"],
            "silhouette_score": float(silhouette),
            "davies_bouldin_index": float(db_index),
            "calinski_harabasz_index": float(ch_index),
            "n_clusters_found": int(n_clusters),
            "n_clusters_target": self.n_clusters,
            "computation_time": float(execution_results["computation_time"]),
            "convergence_history": execution_results.get("convergence_history", [])[-10:],
            "iterations_completed": execution_results.get("iterations_completed", 0)
        }
        
        # Add external validation if available
        if ari is not None:
            formatted["adjusted_rand_index"] = float(ari)
        if nmi is not None:
            formatted["normalized_mutual_info"] = float(nmi)
        
        # Performance assessment based on Silhouette
        if silhouette >= 0.7:
            performance = "GOOD"
            assessment = "Strong clustering structure detected"
        elif silhouette >= 0.5:
            performance = "ACCEPTABLE"
            assessment = "Reasonable clustering with moderate separation"
        elif silhouette >= 0.25:
            performance = "POOR"
            assessment = "Weak clustering structure; clusters overlap significantly"
        else:
            performance = "VERY POOR"
            assessment = "No meaningful clustering detected; consider different method/parameters"
        
        formatted["performance_rating"] = performance
        formatted["assessment"] = assessment
        
        return formatted
    
    def get_evaluation_metrics(
        self,
        method_name: str,
        labels_list: List[np.ndarray],
        computation_times: List[float]
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics across multiple runs.
        
        Args:
            method_name: Name of CI method
            labels_list: List of cluster assignment arrays from multiple runs
            computation_times: List of computation times per run
        
        Returns:
            Dictionary with aggregated metrics matching Section 5.4
        """
        # Compute metrics for each run
        silhouette_scores = []
        db_indices = []
        ch_indices = []
        ari_scores = []
        nmi_scores = []
        n_clusters_list = []
        
        for labels in labels_list:
            metrics = self.evaluate(labels=labels)
            
            silhouette_scores.append(metrics['silhouette'])
            db_indices.append(metrics['davies_bouldin'])
            ch_indices.append(metrics['calinski_harabasz'])
            n_clusters_list.append(metrics['n_clusters'])
            
            if metrics['adjusted_rand'] is not None:
                ari_scores.append(metrics['adjusted_rand'])
            if metrics['normalized_mutual_info'] is not None:
                nmi_scores.append(metrics['normalized_mutual_info'])
        
        # Compute statistics
        sil_stats = compute_statistics(silhouette_scores)
        db_stats = compute_statistics(db_indices)
        ch_stats = compute_statistics(ch_indices)
        n_clusters_stats = compute_statistics(n_clusters_list)
        time_stats = compute_statistics(computation_times)
        
        # External validation stats if available
        ari_stats = compute_statistics(ari_scores) if ari_scores else None
        nmi_stats = compute_statistics(nmi_scores) if nmi_scores else None
        
        # Success rate: % of runs with silhouette > 0.5 (meaningful clusters)
        success_rate_sil = success_rate(silhouette_scores, 0.5)
        
        return {
            "method": method_name,
            "silhouette": {
                "mean": sil_stats['mean'],
                "std": sil_stats['std'],
                "best": sil_stats['max'],
                "worst": sil_stats['min']
            },
            "davies_bouldin": {
                "mean": db_stats['mean'],
                "std": db_stats['std'],
                "best": db_stats['min']  # Lower is better
            },
            "calinski_harabasz": {
                "mean": ch_stats['mean'],
                "std": ch_stats['std'],
                "best": ch_stats['max']  # Higher is better
            },
            "n_clusters": {
                "mean": n_clusters_stats['mean'],
                "std": n_clusters_stats['std'],
                "requested": self.n_clusters
            },
            "success_rate_silhouette_050": success_rate_sil,  # % runs with Silhouette > 0.5
            "computation_time": {
                "mean": time_stats['mean'],
                "std": time_stats['std'],
                "total": sum(computation_times)
            },
            "n_runs": len(labels_list)
        }
    
    def visualize_clusters(
        self,
        labels: Optional[np.ndarray] = None,
        title: str = "Cluster Visualization",
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters in 2D (using first two features or PCA for higher dimensions).
        
        Args:
            labels: Cluster assignments (if None, uses last evaluation)
            title: Plot title
            save_path: Path to save figure (if None, displays interactively)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get labels
            if labels is None:
                if self.cluster_assignments is None:
                    raise ValueError("No cluster assignments available. Run evaluate() first.")
                labels = self.cluster_assignments
            
            # Prepare data for visualization
            if self.n_features >= 2:
                X_vis = self.X[:, :2]  # First two features
                xlabel, ylabel = self.feature_names[0], self.feature_names[1]
            else:
                X_vis = self.X
                xlabel, ylabel = "Feature 1", "Feature 2"
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=X_vis[:, 0],
                y=X_vis[:, 1],
                hue=labels,
                palette="tab10",
                s=60,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.5
            )
            
            plt.title(f"{title}\nDataset: {self.dataset_name.title()} | Clusters: {len(np.unique(labels))}", fontsize=14, fontweight='bold')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Cluster visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib/seaborn not available - skipping visualization")
        except Exception as e:
            logger.error(f"Error visualizing clusters: {e}")


# ==================== FACTORY FUNCTIONS ====================

def create_clustering_problem(
    dataset_name: str,
    n_clusters: Optional[int] = None,
    data_path: Optional[str] = None,
    **kwargs
) -> ClusteringProblem:
    """
    Factory function to create clustering problem instances.
    
    Args:
        dataset_name: Dataset identifier ("iris", "mall_customers", "synthetic")
        n_clusters: Target number of clusters (optional, dataset-dependent defaults used if None)
        data_path: Path to dataset file (for Mall Customers)
        **kwargs: Additional parameters for ClusteringProblem
    
    Returns:
        Configured ClusteringProblem instance
    """
    return ClusteringProblem(
        dataset_name=dataset_name,
        n_clusters=n_clusters,
        data_path=data_path,
        **kwargs
    )


# ==================== PRESET INSTANCE REGISTRY ====================

CLUSTERING_PRESETS = {
    "validation": ["iris"],  # Known labels for validation
    "real_world": ["mall_customers"],  # Real customer data
    "synthetic": ["synthetic_500_5_1.0", "synthetic_500_5_0.5", "synthetic_500_10_1.0"],  # Controlled experiments
    "all": ["iris", "mall_customers", "synthetic_500_5_1.0"]
}


def get_clustering_preset(preset_name: str = "all") -> List[ClusteringProblem]:
    """
    Get list of ClusteringProblem instances for benchmarking.
    
    Args:
        preset_name: 'validation', 'real_world', 'synthetic', 'all'
    
    Returns:
        List of ClusteringProblem instances
    """
    preset_name = preset_name.lower()
    dataset_names = CLUSTERING_PRESETS.get(preset_name, CLUSTERING_PRESETS['all'])
    
    problems = []
    for name in dataset_names:
        try:
            # Extract parameters from synthetic dataset names
            if "synthetic" in name:
                problem = create_clustering_problem(name)
            else:
                problem = create_clustering_problem(name, n_clusters=None)
            
            problems.append(problem)
            logger.info(f"Loaded clustering preset: {name} ({problem.n_samples} samples, {problem.n_features}D)")
        except Exception as e:
            logger.warning(f"Failed to load clustering preset {name}: {e}")
    
    return problems


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Clustering Problem Module Demo ===\n")
    
    # Example 1: Iris dataset with true labels
    try:
        print("1. Loading Iris dataset (with true labels for validation)...")
        
        problem = create_clustering_problem("iris", n_clusters=3)
        
        print(f"\nDataset Summary:")
        print(f"  Samples: {problem.n_samples}")
        print(f"  Features: {problem.n_features} ({', '.join(problem.feature_names)})")
        print(f"  True clusters: {len(np.unique(problem.y_true))}")
        print(f"  Target clusters: {problem.n_clusters}")
        
        # Generate "perfect" labels (using true labels for demonstration)
        perfect_labels = problem.y_true
        
        # Evaluate clustering quality
        metrics = problem.evaluate(labels=perfect_labels)
        
        print(f"\nClustering Metrics (with true labels):")
        print(f"  Silhouette Score:       {metrics['silhouette']:.4f} (higher > 0.5 is good)")
        print(f"  Davies-Bouldin Index:   {metrics['davies_bouldin']:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index:{metrics['calinski_harabasz']:.2f} (higher is better)")
        print(f"  Adjusted Rand Index:    {metrics['adjusted_rand']:.4f} (higher is better)")
        print(f"  NMI:                    {metrics['normalized_mutual_info']:.4f} (higher is better)")
        
    except Exception as e:
        print(f"Error with Iris dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Mall Customers (no true labels)
    try:
        print("\n2. Loading Mall Customers dataset (unsupervised segmentation)...")
        
        # Note: Requires Mall_Customers.csv in data/mall/ directory
        try:
            problem_mall = create_clustering_problem("mall_customers", n_clusters=5)
            
            print(f"\nDataset Summary:")
            print(f"  Samples: {problem_mall.n_samples}")
            print(f"  Features: {problem_mall.n_features} ({', '.join(problem_mall.feature_names)})")
            print(f"  Target clusters: {problem_mall.n_clusters}")
            
            # Generate heuristic labels (k-means like assignment for demonstration)
            np.random.seed(42)
            heuristic_labels = np.random.randint(0, problem_mall.n_clusters, problem_mall.n_samples)
            
            # Evaluate
            metrics_mall = problem_mall.evaluate(labels=heuristic_labels)
            
            print(f"\nClustering Metrics (unsupervised):")
            print(f"  Silhouette Score:       {metrics_mall['silhouette']:.4f}")
            print(f"  Davies-Bouldin Index:   {metrics_mall['davies_bouldin']:.4f}")
            print(f"  Calinski-Harabasz Index:{metrics_mall['calinski_harabasz']:.2f}")
            print(f"  Note: No external validation (true labels unavailable)")
            
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            print("  Download Mall_Customers.csv from Kaggle and place in data/mall/")
    
    except Exception as e:
        print(f"Error with Mall Customers: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Synthetic data with controlled properties
    try:
        print("\n3. Generating synthetic clusters (controlled experiment)...")
        
        problem_synth = create_clustering_problem("synthetic_500_5_0.5", n_clusters=5)
        
        print(f"\nDataset Summary:")
        print(f"  Samples: {problem_synth.n_samples}")
        print(f"  Features: {problem_synth.n_features}D")
        print(f"  True clusters: {problem_synth.data['n_clusters']}")
        print(f"  Cluster tightness: std={problem_synth.data['cluster_std']}")
        
        # Perfect clustering (using true labels)
        perfect_synth = problem_synth.evaluate(labels=problem_synth.y_true)
        
        print(f"\nClustering Metrics (perfect assignment):")
        print(f"  Silhouette Score: {perfect_synth['silhouette']:.4f}")
        print(f"  ARI: {perfect_synth['adjusted_rand']:.4f}")
        
    except Exception as e:
        print(f"Error with synthetic data: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 4: LLM integration
    try:
        print("\n4. Generating LLM problem prompt for Iris clustering...")
        
        prompt = problem.get_llm_problem_prompt(
            preferences={"time_limit": "30 seconds", "priority": "cluster separation quality"}
        )
        
        print("\n" + "="*70)
        print("LLM PROMPT (first 600 characters):")
        print("="*70)
        print(prompt[:600] + "...")
        print("="*70)
        
    except Exception as e:
        print(f"Error generating LLM prompt: {e}")
    
    print("\nDemos completed. For full integration, use with orchestrator.py")
    print("\nNote: Mall Customers dataset requires manual download from Kaggle")