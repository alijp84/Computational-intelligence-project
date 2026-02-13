"""
Classification Problem Implementation for MetaMind CI Framework
Provides standardized interface for Titanic survival prediction with comprehensive metrics.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from copy import deepcopy
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix
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
    accuracy as util_accuracy,
    precision as util_precision,
    recall as util_recall,
    f1_score as util_f1_score,
    confusion_matrix as util_confusion_matrix
)

logger = setup_logger("classification_problem")


# ==================== TITANIC DATA PREPROCESSING ====================

def load_and_preprocess_titanic(
    filepath: Optional[str] = None,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    normalize: bool = True
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Load and preprocess Titanic dataset with standardized pipeline.
    
    Args:
        filepath: Path to train.csv (if None, tries default locations)
        test_size: Fraction of data for final testing
        validation_size: Fraction of training data for validation
        random_state: Random seed for reproducibility
        normalize: Whether to normalize numerical features
    
    Returns:
        Dictionary with preprocessed datasets and metadata
    """
    # Locate dataset file
    if filepath is None:
        possible_paths = [
            "data/titanic/train.csv",
            "data/titanic.csv",
            "titanic/train.csv",
            "train.csv"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        else:
            raise FileNotFoundError(
                "Titanic dataset not found. Please download from Kaggle and place in data/titanic/train.csv"
            )
    
    # Load data
    df = pd.read_csv(filepath)
    original_df = df.copy()
    logger.info(f"Loaded Titanic dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Store passenger IDs for reference
    passenger_ids = df['PassengerId'].values if 'PassengerId' in df.columns else np.arange(len(df))
    
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']  # Cabin has ~77% missing values
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Handle missing values
    # Age: Fill with median (most common approach)
    if 'Age' in df.columns:
        age_median = df['Age'].median()
        df['Age'].fillna(age_median, inplace=True)
        logger.debug(f"Filled {df['Age'].isna().sum()} missing Age values with median: {age_median:.2f}")
    
    # Embarked: Fill with mode (most frequent port)
    if 'Embarked' in df.columns:
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'].fillna(embarked_mode, inplace=True)
        logger.debug(f"Filled {df['Embarked'].isna().sum()} missing Embarked values with mode: {embarked_mode}")
    
    # Fare: Fill with median (for any remaining missing values)
    if 'Fare' in df.columns:
        fare_median = df['Fare'].median()
        df['Fare'].fillna(fare_median, inplace=True)
        logger.debug(f"Filled {df['Fare'].isna().sum()} missing Fare values with median: {fare_median:.2f}")
    
    # Encode categorical variables
    encoders = {}
    
    # Sex: Binary encoding (0=female, 1=male)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        logger.debug("Encoded Sex: female=0, male=1")
    
    # Embarked: One-hot encoding (creates 2 columns for 3 categories)
    if 'Embarked' in df.columns:
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
        df = pd.concat([df.drop('Embarked', axis=1), embarked_dummies], axis=1)
        logger.debug(f"One-hot encoded Embarked into {embarked_dummies.shape[1]} columns")
    
    # Store feature names for interpretability
    feature_names = df.drop('Survived', axis=1).columns.tolist() if 'Survived' in df.columns else df.columns.tolist()
    
    # Separate features and target
    if 'Survived' in df.columns:
        X = df.drop('Survived', axis=1).values.astype(np.float32)
        y = df['Survived'].values.astype(np.int32)
        has_target = True
    else:
        X = df.values.astype(np.float32)
        y = None
        has_target = False
    
    # Normalize numerical features
    scaler = None
    if normalize and X.size > 0:
        # Identify numerical columns (all except one-hot encoded Embarked)
        numerical_cols = [i for i, name in enumerate(feature_names) 
                         if not name.startswith('Embarked_')]
        
        scaler = StandardScaler()
        if numerical_cols:
            X[:, numerical_cols] = scaler.fit_transform(X[:, numerical_cols])
            logger.debug(f"Normalized {len(numerical_cols)} numerical features")
        else:
            X = scaler.fit_transform(X)
            logger.debug("Normalized all features")
    
    # Create splits if target available
    splits = {}
    if has_target:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test, pid_temp, pid_test = train_test_split(
            X, y, passenger_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: separate validation set from remaining
        X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
            X_temp, y_temp, pid_temp,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
            stratify=y_temp
        )
        
        splits = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_ids': pid_train,
            'val_ids': pid_val,
            'test_ids': pid_test
        }
        
        logger.info(f"Data splits created: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return {
        'X': X,
        'y': y,
        'X_train': splits.get('X_train'),
        'y_train': splits.get('y_train'),
        'X_val': splits.get('X_val'),
        'y_val': splits.get('y_val'),
        'X_test': splits.get('X_test'),
        'y_test': splits.get('y_test'),
        'feature_names': feature_names,
        'scaler': scaler,
        'original_df': original_df,
        'preprocessing_info': {
            'missing_age_filled': age_median if 'Age' in locals() else None,
            'missing_embarked_filled': embarked_mode if 'embarked_mode' in locals() else None,
            'normalization': normalize,
            'test_size': test_size,
            'validation_size': validation_size,
            'random_state': random_state
        }
    }


# ==================== BASE CLASSIFICATION PROBLEM CLASS ====================

class ClassificationProblem:
    """
    Standardized classification problem interface for MetaMind orchestrator.
    Implements Titanic survival prediction with comprehensive evaluation metrics.
    """
    
    def __init__(
        self,
        name: str = "Titanic",
        data_path: Optional[str] = None,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42,
        normalize: bool = True
    ):
        self.name = name
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.normalize = normalize
        
        # Load and preprocess data
        self.data = load_and_preprocess_titanic(
            filepath=data_path,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            normalize=normalize
        )
        
        # Extract splits
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train']
        self.X_val = self.data['X_val']
        self.y_val = self.data['y_val']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']
        self.feature_names = self.data['feature_names']
        
        # Problem metadata
        self.n_samples = len(self.X_train) + len(self.X_val) + len(self.X_test)
        self.n_features = self.X_train.shape[1] if self.X_train is not None else 0
        self.n_classes = 2  # Binary classification
        self.class_balance = {
            'train': np.bincount(self.y_train) / len(self.y_train) if self.y_train is not None else None,
            'val': np.bincount(self.y_val) / len(self.y_val) if self.y_val is not None else None,
            'test': np.bincount(self.y_test) / len(self.y_test) if self.y_test is not None else None
        }
        
        # Evaluation tracking
        self.predictions = {}
        self.evaluation_history = []
        self.start_time = None
        
        logger.info(f"Initialized {name} classification problem")
        logger.info(f"  Samples: {self.n_samples} (train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)})")
        logger.info(f"  Features: {self.n_features} ({', '.join(self.feature_names[:5]) + ('...' if len(self.feature_names) > 5 else '')})")
        logger.info(f"  Classes: {self.n_classes} (binary survival prediction)")
        logger.info(f"  Class balance (train): {self.class_balance['train'][1]:.1%} survived")
    
    def evaluate(
        self,
        model: Optional[object] = None,
        predictions: Optional[np.ndarray] = None,
        dataset: str = "validation",
        **kwargs
    ) -> Dict:
        """
        Evaluate classification performance on specified dataset.
        
        Args:
            model: Trained model with predict() method (used if predictions not provided)
            predictions: Pre-computed predictions array (shape: [n_samples])
            dataset: Which dataset to evaluate on ("train", "validation", "test", "all")
            **kwargs: Additional parameters (e.g., 'X', 'y' for custom evaluation)
        
        Returns:
            Dictionary with comprehensive classification metrics:
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'auc_roc': float,
                'confusion_matrix': [[TN, FP], [FN, TP]],
                'predictions': np.ndarray,
                'true_labels': np.ndarray
            }
        """
        # Determine evaluation dataset
        if dataset == "train":
            X_eval = self.X_train
            y_true = self.y_train
        elif dataset == "validation" or dataset == "val":
            X_eval = self.X_val
            y_true = self.y_val
        elif dataset == "test":
            X_eval = self.X_test
            y_true = self.y_test
        elif dataset == "all":
            X_eval = np.vstack([self.X_train, self.X_val, self.X_test])
            y_true = np.concatenate([self.y_train, self.y_val, self.y_test])
        else:
            # Custom dataset from kwargs
            X_eval = kwargs.get('X', self.X_val)
            y_true = kwargs.get('y', self.y_val)
        
        if y_true is None:
            raise ValueError(f"True labels not available for dataset '{dataset}'")
        
        # Generate predictions if not provided
        if predictions is None:
            if model is None:
                raise ValueError("Either model or predictions must be provided")
            
            # Handle different model types
            if hasattr(model, 'predict'):
                predictions = model.predict(X_eval)
            elif hasattr(model, '_inference'):  # Fuzzy controller
                predictions = np.array([1 if model._inference(x) >= 0.5 else 0 for x in X_eval])
            elif isinstance(model, dict) and 'weights' in model:  # Perceptron/MLP solution dict
                # Extract weights and make predictions
                if 'architecture' in model:  # MLP
                    # Simplified MLP inference for evaluation
                    weights = model['weights']
                    biases = model['biases']
                    
                    # Forward pass
                    A = X_eval
                    for i in range(len(weights)):
                        Z = np.dot(A, weights[i]) + biases[i]
                        if i < len(weights) - 1:  # Hidden layers
                            A = np.maximum(0, Z)  # ReLU
                        else:  # Output layer
                            A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Sigmoid
                    
                    predictions = (A > 0.5).astype(int).flatten()
                else:  # Perceptron
                    weights = model['weights']
                    bias = model.get('bias_weight', 0.0)
                    net_input = np.dot(X_eval, weights) + bias
                    predictions = (net_input >= 0).astype(int)
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
        
        # Ensure binary predictions
        predictions = np.round(predictions).astype(int).flatten()
        predictions = np.clip(predictions, 0, 1)  # Ensure 0/1 labels
        
        # Compute metrics
        acc = accuracy_score(y_true, predictions)
        prec = precision_score(y_true, predictions, zero_division=0)
        rec = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        # AUC-ROC requires predicted probabilities; approximate if only hard predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_eval)[:, 1]
            elif hasattr(model, '_inference'):  # Fuzzy controller outputs probability
                y_score = np.array([model._inference(x) for x in X_eval])
            elif isinstance(model, dict) and 'weights' in model and 'architecture' in model:  # MLP
                # Re-run forward pass to get probabilities
                weights = model['weights']
                biases = model['biases']
                A = X_eval
                for i in range(len(weights)):
                    Z = np.dot(A, weights[i]) + biases[i]
                    if i < len(weights) - 1:
                        A = np.maximum(0, Z)
                    else:
                        A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
                y_score = A.flatten()
            else:
                # Approximate AUC using hard predictions (less accurate)
                y_score = predictions.astype(float)
            
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.5  # Default for binary classification with no probability estimates
        
        # Confusion matrix
        cm = sklearn_confusion_matrix(y_true, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:  # Handle edge cases (all one class)
            if np.all(predictions == 0):
                tn, fp, fn, tp = len(y_true) - np.sum(y_true), 0, np.sum(y_true), 0
            else:
                tn, fp, fn, tp = 0, np.sum(y_true == 0), 0, np.sum(y_true == 1)
        
        # Store evaluation
        evaluation = {
            'dataset': dataset,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'predictions': predictions,
            'true_labels': y_true,
            'n_samples': len(y_true),
            'timestamp': time.time()
        }
        
        self.evaluation_history.append(evaluation)
        self.predictions[dataset] = predictions
        
        return evaluation
    
    def reset(self):
        """Reset evaluation history for new experiment."""
        self.evaluation_history = []
        self.predictions = {}
        self.start_time = time.time()
    
    def get_problem_description(self) -> Dict:
        """Return structured problem description for LLM input."""
        return {
            "problem_type": "supervised_classification",
            "domain": "tabular_binary_classification",
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "n_classes": self.n_classes,
            "class_balance": {
                "survived": f"{self.class_balance['train'][1]:.1%}",
                "died": f"{self.class_balance['train'][0]:.1%}"
            },
            "features_description": (
                "Pclass (ticket class), Sex (binary), Age (years), SibSp (# siblings/spouses), "
                "Parch (# parents/children), Fare (ticket fare), Embarked (port of embarkation)"
            ),
            "preprocessing": "Missing values imputed (Age median, Embarked mode), categorical encoded, numerical normalized",
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        }
    
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
        
        prompt = f"""Problem: Binary Classification (Titanic Survival Prediction)
Dataset: Kaggle Titanic Dataset
Samples: {desc['n_samples']} total ({len(self.X_train)} train, {len(self.X_val)} validation, {len(self.X_test)} test)
Features: {desc['n_features']} preprocessed features
  - {', '.join(desc['feature_names'][:4])}
  - {', '.join(desc['feature_names'][4:]) if len(desc['feature_names']) > 4 else ''}
Classes: 2 (Survived: 0=No, 1=Yes)
Class Balance: {desc['class_balance']['survived']} survived, {desc['class_balance']['died']} died in training set

Problem Characteristics:
- Tabular data with mixed feature types (categorical + numerical)
- Moderate class imbalance (~38% survival rate)
- Preprocessing applied: missing value imputation, encoding, normalization
- Evaluation focus: F1-score (balances precision/recall for imbalanced data)

Evaluation Metrics:
- Primary: F1-Score (harmonic mean of precision and recall)
- Secondary: Accuracy, Precision, Recall, AUC-ROC
- Confusion matrix for detailed breakdown

Available CI Methods:
- MLP: Multi-Layer Perceptron - strong baseline for tabular classification
- Perceptron: Simple linear classifier (may struggle with non-linear patterns)
- Fuzzy Controller: Interpretable rule-based system using Wang-Mendel rule generation
- SVM/GA hybrids: Not directly available but GA could optimize MLP hyperparameters

Task:
Analyze this classification problem and recommend the most appropriate CI method with justified parameter configuration.
Consider class imbalance, feature interactions, and interpretability requirements when making your recommendation.
"""
        
        if preferences:
            prompt += "\nUser Preferences:\n"
            for key, value in preferences.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt += """
Note: For classification problems, the objective is to MAXIMIZE F1-score (not minimize).
Higher F1-score indicates better performance. Convert to error metric (1 - F1) for minimization if needed.
"""
        
        return prompt
    
    def format_for_llm_feedback(self, execution_results: Dict) -> Dict:
        """
        Format execution results for LLM interpretation (Step 5 in project doc).
        
        Args:
            execution_results: Dictionary from method execution containing:
                - method_used
                - best_solution (model weights/rules)
                - best_fitness (1 - F1 score for minimization)
                - computation_time
                - convergence_history
                - iterations_completed (optional)
        
        Returns:
            Formatted dictionary matching project specification
        """
        # Extract F1 score from fitness (assuming fitness = 1 - F1 for minimization)
        fitness = execution_results["best_fitness"]
        f1_score_val = 1.0 - fitness
        
        # Get latest evaluation if available
        if self.evaluation_history:
            latest_eval = self.evaluation_history[-1]
            accuracy = latest_eval['accuracy']
            precision = latest_eval['precision']
            recall = latest_eval['recall']
            auc = latest_eval['auc_roc']
            cm = latest_eval['confusion_matrix']
        else:
            # Fallback estimates
            accuracy = f1_score_val  # Rough approximation
            precision = recall = f1_score_val
            auc = 0.5 + f1_score_val / 2
            cm = [[50, 10], [15, 40]]  # Example confusion matrix
        
        formatted = {
            "method_used": execution_results["method_used"],
            "f1_score": float(f1_score_val),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc_roc": float(auc),
            "confusion_matrix": cm,
            "computation_time": float(execution_results["computation_time"]),
            "convergence_history": [
                1.0 - f for f in execution_results.get("convergence_history", [])[-10:]
            ],  # Convert back to F1 for LLM
            "iterations_completed": execution_results.get("iterations_completed", 0),
            "dataset_size": len(self.X_val) if self.X_val is not None else len(self.X_train)
        }
        
        # Performance assessment
        if f1_score_val >= 0.85:
            performance = "GOOD"
        elif f1_score_val >= 0.75:
            performance = "ACCEPTABLE"
        else:
            performance = "POOR"
        
        formatted["performance_rating"] = performance
        
        return formatted
    
    def get_evaluation_metrics(
        self,
        method_name: str,
        predictions_list: List[np.ndarray],
        computation_times: List[float],
        dataset: str = "validation"
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics across multiple runs.
        
        Args:
            method_name: Name of CI method
            predictions_list: List of prediction arrays from multiple runs
            computation_times: List of computation times per run
            dataset: Dataset used for evaluation ("train", "validation", "test")
        
        Returns:
            Dictionary with aggregated metrics matching Section 5.3
        """
        if dataset == "train":
            y_true = self.y_train
        elif dataset == "validation" or dataset == "val":
            y_true = self.y_val
        else:  # test
            y_true = self.y_test
        
        # Compute metrics for each run
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []
        
        for preds in predictions_list:
            # Ensure proper shape
            preds = np.round(preds).astype(int).flatten()[:len(y_true)]
            
            # Handle edge cases (all same prediction)
            if len(np.unique(preds)) < 2:
                acc = accuracy_score(y_true, preds)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                f1 = f1_score(y_true, preds, zero_division=0)
                auc = 0.5
            else:
                acc = accuracy_score(y_true, preds)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                f1 = f1_score(y_true, preds, zero_division=0)
                try:
                    auc = roc_auc_score(y_true, preds)
                except:
                    auc = 0.5
            
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            auc_scores.append(auc)
        
        # Compute statistics
        acc_stats = compute_statistics(accuracies)
        prec_stats = compute_statistics(precisions)
        rec_stats = compute_statistics(recalls)
        f1_stats = compute_statistics(f1_scores)
        auc_stats = compute_statistics(auc_scores)
        time_stats = compute_statistics(computation_times)
        
        # Confusion matrix (average across runs)
        avg_cm = np.zeros((2, 2))
        for preds in predictions_list:
            preds = np.round(preds).astype(int).flatten()[:len(y_true)]
            cm = sklearn_confusion_matrix(y_true, preds)
            if cm.shape == (2, 2):
                avg_cm += cm
            elif np.all(preds == 0):
                avg_cm[0, 0] += np.sum(y_true == 0)
                avg_cm[1, 0] += np.sum(y_true == 1)
            else:
                avg_cm[0, 1] += np.sum(y_true == 0)
                avg_cm[1, 1] += np.sum(y_true == 1)
        avg_cm = (avg_cm / len(predictions_list)).astype(int).tolist()
        
        return {
            "method": method_name,
            "dataset": dataset,
            "accuracy": {
                "mean": acc_stats['mean'],
                "std": acc_stats['std'],
                "best": acc_stats['max']
            },
            "precision": {
                "mean": prec_stats['mean'],
                "std": prec_stats['std'],
                "best": prec_stats['max']
            },
            "recall": {
                "mean": rec_stats['mean'],
                "std": rec_stats['std'],
                "best": rec_stats['max']
            },
            "f1_score": {
                "mean": f1_stats['mean'],
                "std": f1_stats['std'],
                "best": f1_stats['max'],
                "median": np.median(f1_scores)
            },
            "auc_roc": {
                "mean": auc_stats['mean'],
                "std": auc_stats['std']
            },
            "confusion_matrix": avg_cm,
            "computation_time": {
                "mean": time_stats['mean'],
                "std": time_stats['std'],
                "total": sum(computation_times)
            },
            "n_runs": len(predictions_list),
            "success_rate_f1_080": success_rate(f1_scores, 0.80)  # % runs with F1 >= 0.80
        }
    
    def cross_validation(
        self,
        model_factory: Callable,
        n_folds: int = 5,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Perform k-fold cross-validation for model evaluation.
        
        Args:
            model_factory: Callable that returns a fresh model instance
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Combine train + validation for CV
        X_cv = np.vstack([self.X_train, self.X_val])
        y_cv = np.concatenate([self.y_train, self.y_val])
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state or self.random_state)
        scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
            X_fold_train, X_fold_val = X_cv[train_idx], X_cv[val_idx]
            y_fold_train, y_fold_val = y_cv[train_idx], y_cv[val_idx]
            
            # Train model
            model = model_factory()
            start_time = time.time()
            
            # Handle different model training interfaces
            if hasattr(model, 'solve'):
                # MetaMind CI method
                results = model.solve(
                    problem=self,
                    X_train=X_fold_train,
                    y_train=y_fold_train,
                    max_time=30.0  # Time limit per fold
                )
                # Extract predictions
                if 'best_solution' in results and 'weights' in results['best_solution']:
                    # Neural network - use weights to predict
                    predictions = self.evaluate(model=results['best_solution'], X=X_fold_val, y=y_fold_val)['predictions']
                else:
                    # Fallback: use stored predictions if available
                    predictions = self.predictions.get('validation', np.zeros(len(y_fold_val)))
            elif hasattr(model, 'fit'):
                # Scikit-learn compatible
                model.fit(X_fold_train, y_fold_train)
                predictions = model.predict(X_fold_val)
            else:
                raise ValueError(f"Model must have 'solve' or 'fit' method")
            
            # Evaluate
            eval_result = self.evaluate(predictions=predictions, y_true=y_fold_val)
            fold_time = time.time() - start_time
            
            scores.append(eval_result['f1_score'])
            fold_details.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'f1_score': eval_result['f1_score'],
                'accuracy': eval_result['accuracy'],
                'time': fold_time
            })
            
            logger.debug(f"Fold {fold+1}/{n_folds}: F1={eval_result['f1_score']:.4f}, Time={fold_time:.2f}s")
        
        # Aggregate results
        cv_stats = compute_statistics(scores)
        
        return {
            "n_folds": n_folds,
            "f1_scores": scores,
            "mean_f1": cv_stats['mean'],
            "std_f1": cv_stats['std'],
            "min_f1": cv_stats['min'],
            "max_f1": cv_stats['max'],
            "fold_details": fold_details,
            "mean_time_per_fold": np.mean([f['time'] for f in fold_details])
        }


# ==================== FACTORY FUNCTIONS ====================

def create_classification_problem(
    problem_name: str = "titanic",
    data_path: Optional[str] = None,
    **kwargs
) -> ClassificationProblem:
    """
    Factory function to create classification problem instances.
    
    Args:
        problem_name: Problem identifier (currently only "titanic" supported)
        data_path: Optional path to dataset file
        **kwargs: Additional parameters for ClassificationProblem
    
    Returns:
        Configured ClassificationProblem instance
    """
    problem_name = problem_name.lower().strip()
    
    if problem_name in ["titanic", "titanic_survival", "kaggle_titanic"]:
        return ClassificationProblem(
            name="Titanic",
            data_path=data_path,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown classification problem: {problem_name}. "
            f"Available: 'titanic'"
        )


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Classification Problem Module Demo ===\n")
    
    # Example 1: Load and inspect Titanic dataset
    try:
        print("1. Loading and preprocessing Titanic dataset...")
        
        problem = create_classification_problem(
            problem_name="titanic",
            test_size=0.2,
            validation_size=0.2,
            random_state=42
        )
        
        print(f"\nDataset Summary:")
        print(f"  Total samples: {problem.n_samples}")
        print(f"  Features: {problem.n_features}")
        print(f"    - {', '.join(problem.feature_names)}")
        print(f"  Class distribution (train): {problem.class_balance['train'][1]:.1%} survived")
        print(f"  Splits: train={len(problem.X_train)}, val={len(problem.X_val)}, test={len(problem.X_test)}")
        
    except Exception as e:
        print(f"Error loading Titanic dataset: {e}")
        print("Note: Download train.csv from https://www.kaggle.com/c/titanic/data and place in data/titanic/")
        import traceback
        traceback.print_exc()
    
    # Example 2: Dummy model evaluation
    try:
        print("\n2. Evaluating dummy prediction model...")
        
        # Create dummy predictions (predict survival based on sex: females survive)
        dummy_preds = np.where(problem.X_val[:, problem.feature_names.index('Sex')] == 0, 1, 0)
        
        eval_result = problem.evaluate(predictions=dummy_preds, dataset="validation")
        
        print(f"\nDummy Model Performance (Sex-based heuristic):")
        print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall:    {eval_result['recall']:.4f}")
        print(f"  F1-Score:  {eval_result['f1_score']:.4f}")
        print(f"  AUC-ROC:   {eval_result['auc_roc']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"     TN={eval_result['confusion_matrix'][0][0]}  FP={eval_result['confusion_matrix'][0][1]}")
        print(f"     FN={eval_result['confusion_matrix'][1][0]}  TP={eval_result['confusion_matrix'][1][1]}")
        
    except Exception as e:
        print(f"Error in dummy evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: LLM integration
    try:
        print("\n3. Generating LLM problem prompt...")
        
        prompt = problem.get_llm_problem_prompt(
            preferences={"time_limit": "60 seconds", "priority": "F1-score optimization"}
        )
        
        print("\n" + "="*70)
        print("LLM PROMPT (first 600 characters):")
        print("="*70)
        print(prompt[:600] + "...")
        print("="*70)
        
    except Exception as e:
        print(f"Error generating LLM prompt: {e}")
    
    # Example 4: Cross-validation with dummy model
    try:
        print("\n4. Performing 3-fold cross-validation with dummy model...")
        
        def dummy_model_factory():
            """Factory for dummy sex-based classifier."""
            class DummyModel:
                def __init__(self):
                    self.sex_idx = problem.feature_names.index('Sex') if 'Sex' in problem.feature_names else 0
                
                def predict(self, X):
                    return np.where(X[:, self.sex_idx] == 0, 1, 0)
            
            return DummyModel()
        
        cv_results = problem.cross_validation(
            model_factory=dummy_model_factory,
            n_folds=3,
            random_state=42
        )
        
        print(f"\nCross-Validation Results (3 folds):")
        print(f"  Mean F1-Score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"  Range: [{cv_results['min_f1']:.4f}, {cv_results['max_f1']:.4f}]")
        print(f"  Mean time per fold: {cv_results['mean_time_per_fold']:.2f}s")
        
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDemos completed. For full integration, use with orchestrator.py")
    print("\nNote: For real CI method evaluation, pass trained models to problem.evaluate()")