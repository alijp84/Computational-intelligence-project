"""
Fuzzy Controller for MetaMind CI Framework
Implements fuzzy inference system with configurable membership functions,
rule generation strategies, and defuzzification methods.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable, Union
import logging
from abc import ABC, abstractmethod
import time
from copy import deepcopy

from utils import (
    setup_logger,
    EarlyStopping
)

logger = setup_logger("fuzzy_methods")


# ==================== MEMBERSHIP FUNCTION CLASSES ====================

class MembershipFunction(ABC):
    """Abstract base class for membership functions."""
    
    def __init__(self, params: Dict[str, float]):
        self.params = params
    
    @abstractmethod
    def membership(self, x: float) -> float:
        """Compute membership degree for input x."""
        pass
    
    @abstractmethod
    def support(self) -> Tuple[float, float]:
        """Return the support interval (min, max) where membership > 0."""
        pass


class TriangularMF(MembershipFunction):
    """Triangular membership function."""
    
    def __init__(self, a: float, b: float, c: float):
        """
        Args:
            a: Left vertex (membership = 0)
            b: Peak vertex (membership = 1)
            c: Right vertex (membership = 0)
        """
        super().__init__({'a': a, 'b': b, 'c': c})
        self.a = a
        self.b = b
        self.c = c
    
    def membership(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:  # x <= self.c
            return (self.c - x) / (self.c - self.b)
    
    def support(self) -> Tuple[float, float]:
        return (self.a, self.c)


class GaussianMF(MembershipFunction):
    """Gaussian membership function."""
    
    def __init__(self, mean: float, sigma: float):
        super().__init__({'mean': mean, 'sigma': sigma})
        self.mean = mean
        self.sigma = sigma
    
    def membership(self, x: float) -> float:
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)
    
    def support(self) -> Tuple[float, float]:
        # Approximate support (99.7% interval)
        return (self.mean - 3 * self.sigma, self.mean + 3 * self.sigma)


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function."""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Args:
            a: Left bottom vertex
            b: Left top vertex
            c: Right top vertex
            d: Right bottom vertex
        """
        super().__init__({'a': a, 'b': b, 'c': c, 'd': d})
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def membership(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        elif x <= self.c:
            return 1.0
        else:  # x < self.d
            return (self.d - x) / (self.d - self.c)
    
    def support(self) -> Tuple[float, float]:
        return (self.a, self.d)


# ==================== FUZZY VARIABLE ====================

class FuzzyVariable:
    """Represents a fuzzy variable with multiple membership functions."""
    
    def __init__(
        self,
        name: str,
        universe: Tuple[float, float],
        n_mfs: int = 3,
        mf_type: str = "triangular"
    ):
        self.name = name
        self.universe = universe  # (min, max) range
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.membership_functions: List[MembershipFunction] = []
        self._initialize_mfs()
    
    def _initialize_mfs(self):
        """Initialize evenly spaced membership functions over the universe."""
        min_val, max_val = self.universe
        range_val = max_val - min_val
        
        if self.mf_type == "triangular":
            # Create overlapping triangular MFs
            step = range_val / (self.n_mfs - 1) if self.n_mfs > 1 else range_val
            for i in range(self.n_mfs):
                a = min_val + (i - 1) * step
                b = min_val + i * step
                c = min_val + (i + 1) * step
                
                # Clamp to universe boundaries
                a = max(a, min_val - range_val)
                c = min(c, max_val + range_val)
                
                self.membership_functions.append(TriangularMF(a, b, c))
        
        elif self.mf_type == "gaussian":
            # Create Gaussian MFs with 50% overlap
            step = range_val / (self.n_mfs - 1) if self.n_mfs > 1 else range_val
            sigma = step / 2.0  # 50% overlap at 0.5 membership
            
            for i in range(self.n_mfs):
                mean = min_val + i * step
                self.membership_functions.append(GaussianMF(mean, sigma))
        
        elif self.mf_type == "trapezoidal":
            # Create trapezoidal MFs with flat tops
            if self.n_mfs == 1:
                self.membership_functions.append(TrapezoidalMF(min_val, min_val, max_val, max_val))
            else:
                step = range_val / (self.n_mfs - 1)
                width = step * 0.6  # Width of flat top
                
                for i in range(self.n_mfs):
                    center = min_val + i * step
                    a = center - step
                    b = center - width / 2
                    c = center + width / 2
                    d = center + step
                    
                    # Clamp boundaries
                    a = max(a, min_val - range_val)
                    d = min(d, max_val + range_val)
                    
                    self.membership_functions.append(TrapezoidalMF(a, b, c, d))
        
        else:
            raise ValueError(f"Unsupported membership function type: {self.mf_type}")
    
    def fuzzify(self, x: float) -> List[float]:
        """Compute membership degrees for all MFs given input x."""
        # Clamp input to universe boundaries
        x = max(min(x, self.universe[1]), self.universe[0])
        return [mf.membership(x) for mf in self.membership_functions]
    
    def get_mf_labels(self) -> List[str]:
        """Get linguistic labels for membership functions."""
        labels = ["Low", "Medium", "High"]
        if self.n_mfs == 5:
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        elif self.n_mfs == 7:
            labels = ["Extremely Low", "Very Low", "Low", "Medium", "High", "Very High", "Extremely High"]
        return labels[:self.n_mfs]


# ==================== FUZZY RULE ====================

class FuzzyRule:
    """Represents a fuzzy IF-THEN rule."""
    
    def __init__(
        self,
        antecedent_indices: List[int],
        consequent: float,
        weight: float = 1.0
    ):
        """
        Args:
            antecedent_indices: List of MF indices for each input variable
            consequent: Output value (for singleton consequents) or MF index
            weight: Rule weight (0.0 to 1.0)
        """
        self.antecedent_indices = antecedent_indices
        self.consequent = consequent
        self.weight = weight
    
    def firing_strength(self, input_memberships: List[List[float]]) -> float:
        """Compute rule firing strength using min operator."""
        strengths = []
        for var_idx, mf_idx in enumerate(self.antecedent_indices):
            if mf_idx < len(input_memberships[var_idx]):
                strengths.append(input_memberships[var_idx][mf_idx])
            else:
                strengths.append(0.0)
        
        # Min operator for AND conjunction
        return min(strengths) * self.weight if strengths else 0.0


# ==================== BASE FUZZY METHOD CLASS ====================

class FuzzyMethod(ABC):
    """Abstract base class for fuzzy methods with standardized interface."""
    
    def __init__(self, **params):
        self.params = params
        self.convergence_history = []
        self.best_solution = None
        self.best_fitness = float('inf')  # For minimization problems
        self.start_time = None
        self.elapsed_time = 0.0
        self.iterations_completed = 0
        self.name = self.__class__.__name__
        self.rules: List[FuzzyRule] = []
        self.input_vars: List[FuzzyVariable] = []
        self.output_var: Optional[FuzzyVariable] = None
    
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
            "iterations_completed": self.iterations_completed,
            "n_rules": len(self.rules)
        }


# ==================== FUZZY CONTROLLER ====================

class FuzzyController(FuzzyMethod):
    """
    Fuzzy Controller for classification and regression problems.
    Supports Wang-Mendel rule generation and multiple defuzzification methods.
    """
    
    def __init__(
        self,
        n_membership_functions: int = 3,
        membership_type: str = "triangular",
        defuzzification: str = "centroid",
        rule_generation: str = "wang_mendel",
        **kwargs
    ):
        super().__init__(
            n_membership_functions=n_membership_functions,
            membership_type=membership_type,
            defuzzification=defuzzification,
            rule_generation=rule_generation,
            **kwargs
        )
        self.n_mfs = n_membership_functions
        self.mf_type = membership_type
        self.defuzz_method = defuzzification
        self.rule_gen_method = rule_generation
    
    def _create_fuzzy_variables(
        self,
        n_features: int,
        feature_ranges: List[Tuple[float, float]],
        output_range: Tuple[float, float]
    ):
        """Create input and output fuzzy variables."""
        # Create input variables
        self.input_vars = []
        for i in range(n_features):
            var = FuzzyVariable(
                name=f"feature_{i}",
                universe=feature_ranges[i],
                n_mfs=self.n_mfs,
                mf_type=self.mf_type
            )
            self.input_vars.append(var)
        
        # Create output variable (for regression/classification probability)
        self.output_var = FuzzyVariable(
            name="output",
            universe=output_range,
            n_mfs=self.n_mfs,
            mf_type=self.mf_type
        )
    
    def _wang_mendel_generate_rules(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[FuzzyRule]:
        """
        Generate fuzzy rules using Wang-Mendel method.
        
        Steps:
        1. For each training sample, find the most activated MF for each input variable
        2. Form a rule antecedent from these MF indices
        3. Assign the consequent as the output value (for regression) or class probability
        4. For conflicting rules (same antecedent, different consequents), use weighted average
        """
        n_samples = X.shape[0]
        n_vars = len(self.input_vars)
        rule_consequents: Dict[Tuple[int, ...], List[float]] = {}
        rule_weights: Dict[Tuple[int, ...], List[float]] = {}
        
        # Process each sample
        for i in range(n_samples):
            x = X[i]
            output_val = y[i]
            
            # Find most activated MF for each input variable
            antecedent_indices = []
            firing_strength = 1.0
            
            for var_idx, var in enumerate(self.input_vars):
                memberships = var.fuzzify(x[var_idx])
                if not memberships:
                    continue
                
                # Find MF with maximum membership
                mf_idx = int(np.argmax(memberships))
                antecedent_indices.append(mf_idx)
                firing_strength *= memberships[mf_idx]  # Product for overall strength
            
            # Use tuple as dictionary key
            antecedent_key = tuple(antecedent_indices)
            
            # Store consequent value weighted by firing strength
            if antecedent_key not in rule_consequents:
                rule_consequents[antecedent_key] = []
                rule_weights[antecedent_key] = []
            
            rule_consequents[antecedent_key].append(output_val)
            rule_weights[antecedent_key].append(firing_strength)
        
        # Resolve conflicting rules by weighted averaging
        rules = []
        for antecedent_key, consequents in rule_consequents.items():
            weights = rule_weights[antecedent_key]
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Weighted average of consequents
                weighted_consequent = sum(c * w for c, w in zip(consequents, weights)) / total_weight
            else:
                weighted_consequent = np.mean(consequents)
            
            rules.append(FuzzyRule(
                antecedent_indices=list(antecedent_key),
                consequent=weighted_consequent,
                weight=1.0
            ))
        
        logger.info(f"Wang-Mendel generated {len(rules)} rules from {n_samples} samples")
        return rules
    
    def _manual_generate_rules(self, n_features: int) -> List[FuzzyRule]:
        """Generate simple manual rules for demonstration."""
        rules = []
        
        # For binary classification (Titanic), create simple rules
        # Rule 1: IF all features are Low THEN output = 0.2 (low survival prob)
        rules.append(FuzzyRule(
            antecedent_indices=[0] * n_features,  # All "Low" MFs
            consequent=0.2,
            weight=1.0
        ))
        
        # Rule 2: IF all features are High THEN output = 0.8 (high survival prob)
        rules.append(FuzzyRule(
            antecedent_indices=[self.n_mfs - 1] * n_features,  # All "High" MFs
            consequent=0.8,
            weight=1.0
        ))
        
        # Rule 3: IF mixed features THEN output = 0.5 (medium survival prob)
        mid_idx = self.n_mfs // 2
        rules.append(FuzzyRule(
            antecedent_indices=[mid_idx] * n_features,
            consequent=0.5,
            weight=0.8
        ))
        
        logger.info(f"Manual rule generation created {len(rules)} rules")
        return rules
    
    def _inference(self, x: np.ndarray) -> float:
        """
        Perform fuzzy inference for a single input sample.
        
        Returns:
            Crisp output value after defuzzification
        """
        if not self.rules:
            raise ValueError("No rules defined. Call solve() first.")
        
        # Fuzzify inputs
        input_memberships = [var.fuzzify(x[i]) for i, var in enumerate(self.input_vars)]
        
        # Compute firing strengths and aggregate output fuzzy set
        rule_outputs = []
        firing_strengths = []
        
        for rule in self.rules:
            fs = rule.firing_strength(input_memberships)
            if fs > 0:
                rule_outputs.append(rule.consequent)
                firing_strengths.append(fs)
        
        if not firing_strengths:
            # No rule fired - return middle of output universe
            return (self.output_var.universe[0] + self.output_var.universe[1]) / 2.0
        
        # Defuzzification
        if self.defuzz_method == "centroid":
            # Center of gravity (for singleton consequents, this is weighted average)
            numerator = sum(fs * out for fs, out in zip(firing_strengths, rule_outputs))
            denominator = sum(firing_strengths)
            return numerator / denominator if denominator > 0 else 0.5
        
        elif self.defuzz_method == "bisector":
            # Find value that divides area into two equal parts
            # For singleton consequents, approximate with weighted median
            pairs = sorted(zip(rule_outputs, firing_strengths), key=lambda x: x[0])
            total_weight = sum(firing_strengths)
            cumulative = 0.0
            for out, fs in pairs:
                cumulative += fs
                if cumulative >= total_weight / 2.0:
                    return out
            return pairs[-1][0]
        
        elif self.defuzz_method == "mom":
            # Mean of maximum
            max_fs = max(firing_strengths)
            max_outputs = [out for out, fs in zip(rule_outputs, firing_strengths) if fs >= max_fs * 0.99]
            return np.mean(max_outputs) if max_outputs else 0.5
        
        elif self.defuzz_method == "som":
            # Smallest of maximum
            max_fs = max(firing_strengths)
            for out, fs in sorted(zip(rule_outputs, firing_strengths), key=lambda x: x[0]):
                if fs >= max_fs * 0.99:
                    return out
            return 0.5
        
        else:
            raise ValueError(f"Unsupported defuzzification method: {self.defuzz_method}")
    
    def _compute_classification_error(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Compute classification error rate."""
        y_pred = np.array([1 if self._inference(x) >= threshold else 0 for x in X])
        return np.mean(y_true != y_pred)
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Train and evaluate fuzzy controller on classification problem.
        
        Args:
            problem: Classification problem with X_train, y_train attributes
            max_time: Optional time limit in seconds
            callback: Optional callback function (iteration, error, rules)
            **kwargs: Additional parameters (e.g., 'X_train', 'y_train', 'X_val', 'y_val')
        """
        # Extract data
        X_train = kwargs.get('X_train', getattr(problem, 'X_train', None))
        y_train = kwargs.get('y_train', getattr(problem, 'y_train', None))
        X_val = kwargs.get('X_val', getattr(problem, 'X_val', None))
        y_val = kwargs.get('y_val', getattr(problem, 'y_val', None))
        
        if X_train is None or y_train is None:
            raise ValueError("FuzzyController requires training data (X_train, y_train) in kwargs or problem attributes")
        
        # For classification, ensure binary labels (0/1)
        unique_classes = np.unique(y_train)
        if len(unique_classes) != 2:
            raise ValueError(f"FuzzyController currently supports binary classification only. Found {len(unique_classes)} classes.")
        
        # Convert to 0/1 if needed
        if not np.array_equal(unique_classes, [0, 1]):
            y_train = (y_train == unique_classes[1]).astype(int)
            if y_val is not None:
                y_val = (y_val == unique_classes[1]).astype(int)
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=15, min_delta=1e-4, mode='min')
        
        # Determine feature ranges from training data
        feature_ranges = [(X_train[:, i].min(), X_train[:, i].max()) for i in range(X_train.shape[1])]
        output_range = (0.0, 1.0)  # For binary classification probability
        
        # Create fuzzy variables
        self._create_fuzzy_variables(
            n_features=X_train.shape[1],
            feature_ranges=feature_ranges,
            output_range=output_range
        )
        
        # Generate rules
        if self.rule_gen_method == "wang_mendel":
            self.rules = self._wang_mendel_generate_rules(X_train, y_train)
        elif self.rule_gen_method == "manual":
            self.rules = self._manual_generate_rules(X_train.shape[1])
        else:
            raise ValueError(f"Unsupported rule generation method: {self.rule_gen_method}")
        
        # Evaluate on validation set or training set
        if X_val is not None and y_val is not None:
            error = self._compute_classification_error(X_val, y_val)
            evaluation_set = "validation"
        else:
            error = self._compute_classification_error(X_train, y_train)
            evaluation_set = "training"
        
        # Update convergence (minimize error rate)
        self._update_convergence(error, {
            'rules': self.rules,
            'input_vars': self.input_vars,
            'output_var': self.output_var,
            'error_rate': error,
            'evaluation_set': evaluation_set
        })
        
        self.elapsed_time = time.time() - self.start_time
        
        # Final evaluation metrics
        if X_val is not None and y_val is not None:
            y_pred_val = np.array([1 if self._inference(x) >= 0.5 else 0 for x in X_val])
            accuracy = np.mean(y_pred_val == y_val)
            error_rate = 1.0 - accuracy
        else:
            y_pred_train = np.array([1 if self._inference(x) >= 0.5 else 0 for x in X_train])
            accuracy = np.mean(y_pred_train == y_train)
            error_rate = 1.0 - accuracy
        
        # Store final solution
        self.best_fitness = error_rate
        self.best_solution = {
            'rules': self.rules,
            'n_rules': len(self.rules),
            'accuracy': accuracy,
            'error_rate': error_rate,
            'defuzzification_method': self.defuzz_method,
            'membership_type': self.mf_type,
            'n_membership_functions': self.n_mfs
        }
        
        # Callback for final reporting
        if callback:
            callback(0, error_rate, self.rules)
        
        return self.get_results()


# ==================== METHOD FACTORY ====================

def create_fuzzy_method(**params) -> FuzzyController:
    """
    Factory function to create fuzzy controller with specified parameters.
    
    Args:
        **params: FuzzyController parameters per project spec
        
    Returns:
        Instantiated FuzzyController
    """
    return FuzzyController(**params)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Fuzzy Controller Demo ===\n")
    
    # Example 1: Triangular MFs with Wang-Mendel on synthetic data
    try:
        print("1. Fuzzy Controller on synthetic binary classification...")
        
        # Generate synthetic data (simple decision boundary)
        np.random.seed(42)
        n_samples = 200
        X = np.random.uniform(0, 1, (n_samples, 2))
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # Simple diagonal boundary
        
        # Split train/validation
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        fuzzy = FuzzyController(
            n_membership_functions=3,
            membership_type="triangular",
            defuzzification="centroid",
            rule_generation="wang_mendel"
        )
        
        results = fuzzy.solve(
            problem=None,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_time=10.0
        )
        
        print(f"   Rules generated: {results['n_rules']}")
        print(f"   Validation error: {results['best_fitness']:.4f}")
        print(f"   Accuracy: {(1 - results['best_fitness']) * 100:.2f}%")
        print(f"   Time: {results['computation_time']:.3f}s")
        
    except Exception as e:
        print(f"   Fuzzy demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Gaussian MFs with different defuzzification
    try:
        print("\n2. Fuzzy Controller with Gaussian MFs and SOM defuzzification...")
        
        fuzzy2 = FuzzyController(
            n_membership_functions=5,
            membership_type="gaussian",
            defuzzification="som",
            rule_generation="wang_mendel"
        )
        
        results2 = fuzzy2.solve(
            problem=None,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_time=10.0
        )
        
        print(f"   Rules generated: {results2['n_rules']}")
        print(f"   Validation error: {results2['best_fitness']:.4f}")
        print(f"   Defuzzification: {fuzzy2.defuzz_method}")
        
    except Exception as e:
        print(f"   Gaussian MF demo failed: {e}")
    
    # Example 3: Manual rule generation
    try:
        print("\n3. Fuzzy Controller with manual rule generation...")
        
        fuzzy3 = FuzzyController(
            n_membership_functions=3,
            membership_type="triangular",
            defuzzification="centroid",
            rule_generation="manual"
        )
        
        results3 = fuzzy3.solve(
            problem=None,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_time=5.0
        )
        
        print(f"   Rules generated: {results3['n_rules']}")
        print(f"   Validation error: {results3['best_fitness']:.4f}")
        print(f"   Rule generation: {fuzzy3.rule_gen_method}")
        
    except Exception as e:
        print(f"   Manual rule demo failed: {e}")
    
    print("\nDemos completed. For full integration, use with orchestrator.py")