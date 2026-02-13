"""
Optimization Problem Implementation for MetaMind CI Framework
Provides standardized interface for multimodal function optimization benchmarks.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from copy import deepcopy
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    setup_logger,
    compute_statistics,
    success_rate
)

logger = setup_logger("optimization_problem")


# ==================== BASE OPTIMIZATION PROBLEM CLASS ====================

class OptimizationProblem:
    """
    Base class for continuous optimization problems.
    Provides standardized interface for function minimization.
    """
    
    def __init__(
        self,
        name: str,
        dimension: int,
        function: Callable,
        bounds: List[Tuple[float, float]],
        global_optimum: float = 0.0,
        optimum_location: Optional[np.ndarray] = None,
        description: str = ""
    ):
        self.name = name
        self.dimension = dimension
        self.function = function
        self.bounds = bounds
        self.global_optimum = global_optimum
        self.optimum_location = optimum_location if optimum_location is not None else np.zeros(dimension)
        self.description = description
        self.evaluation_count = 0
        self.history = []  # Store (iteration, solution, fitness) tuples
        self.start_time = None
        
        # Validate bounds
        if len(bounds) != dimension:
            raise ValueError(f"Bounds length ({len(bounds)}) must match dimension ({dimension})")
        
        logger.info(f"Initialized {name} optimization problem (D={dimension})")
    
    def evaluate(self, x: Union[List[float], np.ndarray], **kwargs) -> Dict:
        """
        Evaluate objective function at point x.
        
        Args:
            x: Solution vector of length self.dimension
            **kwargs: Additional parameters (e.g., 'iteration' for history tracking)
        
        Returns:
            Dictionary with evaluation results:
            {
                'fitness': float,          # Objective function value (to minimize)
                'x': np.ndarray,           # Solution vector
                'error': float,            # |f(x) - f_opt|
                'is_optimal': bool,        # Within tolerance of optimum
                'function_evaluations': int
            }
        """
        # Convert to numpy array and validate
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"Expected list or np.ndarray, got {type(x)}")
        
        if x.shape[0] != self.dimension:
            raise ValueError(f"Solution dimension {x.shape[0]} != problem dimension {self.dimension}")
        
        # Clamp to bounds (for methods that may violate constraints)
        for i, (low, high) in enumerate(self.bounds):
            x[i] = np.clip(x[i], low, high)
        
        # Evaluate function
        start_eval = time.time()
        fitness = self.function(x)
        eval_time = time.time() - start_eval
        
        # Track evaluations
        self.evaluation_count += 1
        
        # Compute error metrics
        error = abs(fitness - self.global_optimum)
        is_optimal = error < 1e-4  # Success criterion per Section 5.2
        
        # Store in history if iteration provided
        iteration = kwargs.get('iteration', len(self.history))
        self.history.append({
            'iteration': iteration,
            'x': x.copy(),
            'fitness': fitness,
            'error': error,
            'time': time.time() - (self.start_time if self.start_time else time.time())
        })
        
        return {
            'fitness': float(fitness),
            'x': x,
            'error': float(error),
            'is_optimal': is_optimal,
            'function_evaluations': self.evaluation_count,
            'evaluation_time': eval_time
        }
    
    def reset(self):
        """Reset evaluation counter and history for new run."""
        self.evaluation_count = 0
        self.history = []
        self.start_time = time.time()
    
    def get_convergence_history(self) -> List[float]:
        """Return best fitness history for convergence plotting."""
        if not self.history:
            return []
        
        # Compute running minimum (for minimization problems)
        fitnesses = [h['fitness'] for h in self.history]
        running_min = []
        current_best = float('inf')
        for f in fitnesses:
            current_best = min(current_best, f)
            running_min.append(current_best)
        return running_min
    
    def get_problem_description(self) -> Dict:
        """Return structured problem description for LLM input."""
        # Compute search space volume for LLM context
        volume = 1.0
        for low, high in self.bounds:
            volume *= (high - low)
        
        return {
            "problem_type": "continuous_optimization",
            "domain": "multimodal_function_minimization",
            "name": self.name,
            "dimension": self.dimension,
            "bounds": self.bounds,
            "global_optimum": self.global_optimum,
            "optimum_known": True,
            "characteristics": self._get_characteristics(),
            "search_space_volume": f"{volume:.2e}",
            "success_criterion": "error < 1e-4"
        }
    
    def _get_characteristics(self) -> str:
        """Return problem characteristics string for LLM."""
        if "rastrigin" in self.name.lower():
            return "Highly multimodal with regular grid of local minima; large search space"
        elif "ackley" in self.name.lower():
            return "Large nearly flat outer region surrounding deep global optimum; deceptive"
        elif "rosenbrock" in self.name.lower():
            return "Narrow, parabolic valley leading to global optimum; difficult convergence"
        elif "sphere" in self.name.lower():
            return "Unimodal, convex, smooth; baseline for comparison"
        else:
            return "Continuous optimization problem"
    
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
        
        prompt = f"""Problem: Function Optimization
Function: {desc['name'].replace('_', ' ').title()}
Dimension: {desc['dimension']}D
Domain: Continuous space with bounds {self.bounds[0]} per dimension
Objective: Minimize function value (global optimum = {self.global_optimum})

Problem Characteristics:
- Type: {desc['characteristics']}
- Search space volume: {desc['search_space_volume']}
- Global optimum known: Yes ({self.global_optimum})
- Success criterion: Solution error < 1e-4
"""
        
        if preferences:
            prompt += "\nUser Preferences:\n"
            for key, value in preferences.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt += """
Available CI Methods:
- PSO: Particle Swarm Optimization - excels on multimodal functions with moderate dimensions (<30D)
- GA: Genetic Algorithm - robust for high-dimensional spaces and irregular landscapes
- ACO: Ant Colony Optimization - can be adapted via continuous variants (ACO_R)
- Hopfield Network: Less suitable for continuous optimization (better for combinatorial)

Task:
Analyze this optimization problem and recommend the most appropriate CI method with justified parameter configuration.
Consider dimensionality, multimodality, and user preferences when making your recommendation.
"""
        
        return prompt
    
    def format_for_llm_feedback(self, execution_results: Dict) -> Dict:
        """
        Format execution results for LLM interpretation (Step 5 in project doc).
        
        Args:
            execution_results: Dictionary from method execution containing:
                - method_used
                - best_solution (x vector)
                - best_fitness
                - computation_time
                - convergence_history
                - iterations_completed (optional)
                - function_evaluations (optional)
        
        Returns:
            Formatted dictionary matching project specification
        """
        formatted = {
            "method_used": execution_results["method_used"],
            "best_solution": execution_results["best_solution"][:5].tolist() + ["..."] 
                if len(execution_results["best_solution"]) > 10 
                else execution_results["best_solution"].tolist(),
            "best_fitness": float(execution_results["best_fitness"]),
            "computation_time": float(execution_results["computation_time"]),
            "convergence_history": execution_results.get("convergence_history", [])[-10:],  # Last 10 iterations
            "iterations_completed": execution_results.get("iterations_completed", 0),
            "function_evaluations": execution_results.get("function_evaluations", self.evaluation_count)
        }
        
        # Add optimality metrics
        error = abs(execution_results["best_fitness"] - self.global_optimum)
        formatted.update({
            "global_optimum": float(self.global_optimum),
            "error": float(error),
            "success": error < 1e-4
        })
        
        return formatted
    
    def get_evaluation_metrics(
        self,
        fitness_values: List[float],
        computation_times: List[float],
        function_evaluations: List[int]
    ) -> Dict:
        """
        Compute all required evaluation metrics for multiple runs.
        
        Returns:
            Dictionary with metrics matching Section 5.2 of project doc
        """
        stats = compute_statistics(fitness_values)
        error_values = [abs(f - self.global_optimum) for f in fitness_values]
        error_stats = compute_statistics(error_values)
        
        metrics = {
            "best_fitness": stats['min'],
            "mean_fitness": stats['mean'],
            "std_fitness": stats['std'],
            "median_fitness": np.median(fitness_values),
            "error_best": error_stats['min'],
            "error_mean": error_stats['mean'],
            "error_std": error_stats['std'],
            "success_rate": success_rate(error_values, 1e-4),  # % within 1e-4 of optimum
            "mean_time": np.mean(computation_times),
            "std_time": np.std(computation_times, ddof=1) if len(computation_times) > 1 else 0.0,
            "mean_evaluations": np.mean(function_evaluations),
            "std_evaluations": np.std(function_evaluations, ddof=1) if len(function_evaluations) > 1 else 0.0,
            "n_runs": len(fitness_values)
        }
        
        # Add convergence metrics if history available
        if self.history:
            convergence_history = self.get_convergence_history()
            metrics.update({
                "convergence_history": convergence_history,
                "final_improvement": convergence_history[-1] if convergence_history else None
            })
        
        return metrics


# ==================== BENCHMARK FUNCTIONS ====================

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function - highly multimodal with regular structure.
    
    f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function - large flat region with deep global optimum.
    
    f(x) = -20exp(-0.2√(1/n Σx_i²)) - exp(1/n Σcos(2πx_i)) + 20 + e
    Domain: x_i ∈ [-5, 5]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function - narrow curved valley, difficult convergence.
    
    f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
    Domain: x_i ∈ [-5, 10]
    Global minimum: f(1, 1, ..., 1) = 0
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def sphere(x: np.ndarray) -> float:
    """
    Sphere function - unimodal baseline for comparison.
    
    f(x) = Σx_i²
    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    return np.sum(x**2)


# ==================== PROBLEM FACTORY ====================

def create_optimization_problem(
    problem_name: str,
    dimension: Optional[int] = None
) -> OptimizationProblem:
    """
    Factory function to create optimization problem instances.
    
    Args:
        problem_name: Problem identifier (e.g., "rastrigin_10d", "ackley_20d", "rosenbrock", "sphere_30d")
        dimension: Optional override for dimension (if not in name)
    
    Returns:
        Configured OptimizationProblem instance
    """
    problem_name = problem_name.lower().strip()
    
    # Extract dimension from name if present
    if dimension is None:
        for dim_str in ['_30d', '_20d', '_10d', '30d', '20d', '10d']:
            if dim_str in problem_name:
                dimension = int(dim_str.replace('d', '').replace('_', ''))
                problem_name = problem_name.replace(dim_str, '').rstrip('_')
                break
        else:
            dimension = 10  # Default dimension
    
    # Create problem based on name
    if "rastrigin" in problem_name:
        bounds = [(-5.12, 5.12)] * dimension
        return OptimizationProblem(
            name=f"Rastrigin_{dimension}D",
            dimension=dimension,
            function=rastrigin,
            bounds=bounds,
            global_optimum=0.0,
            optimum_location=np.zeros(dimension),
            description="Highly multimodal function with regular grid of local minima"
        )
    
    elif "ackley" in problem_name:
        bounds = [(-5.0, 5.0)] * dimension
        return OptimizationProblem(
            name=f"Ackley_{dimension}D",
            dimension=dimension,
            function=ackley,
            bounds=bounds,
            global_optimum=0.0,
            optimum_location=np.zeros(dimension),
            description="Function with large flat region surrounding deep global optimum"
        )
    
    elif "rosenbrock" in problem_name or "banana" in problem_name:
        bounds = [(-5.0, 10.0)] * dimension
        optimum = np.ones(dimension)
        return OptimizationProblem(
            name=f"Rosenbrock_{dimension}D",
            dimension=dimension,
            function=rosenbrock,
            bounds=bounds,
            global_optimum=0.0,
            optimum_location=optimum,
            description="Function with narrow, parabolic valley leading to global optimum"
        )
    
    elif "sphere" in problem_name:
        bounds = [(-5.12, 5.12)] * dimension
        return OptimizationProblem(
            name=f"Sphere_{dimension}D",
            dimension=dimension,
            function=sphere,
            bounds=bounds,
            global_optimum=0.0,
            optimum_location=np.zeros(dimension),
            description="Unimodal, convex, smooth baseline function"
        )
    
    else:
        raise ValueError(
            f"Unknown optimization problem: {problem_name}. "
            f"Available: rastrigin, ackley, rosenbrock, sphere (with optional _10d/_20d/_30d suffix)"
        )


# ==================== PRESET INSTANCE REGISTRY ====================

OPTIMIZATION_PRESETS = {
    "small": ["rastrigin_10d", "ackley_10d", "rosenbrock_10d", "sphere_10d"],
    "medium": ["rastrigin_20d", "ackley_20d", "rosenbrock_20d"],
    "large": ["rastrigin_30d", "ackley_30d", "rosenbrock_30d"],
    "all": [
        "rastrigin_10d", "rastrigin_20d", "rastrigin_30d",
        "ackley_10d", "ackley_20d", "ackley_30d",
        "rosenbrock_10d", "rosenbrock_20d", "rosenbrock_30d",
        "sphere_10d", "sphere_20d", "sphere_30d"
    ],
    "multimodal": ["rastrigin_10d", "ackley_10d", "rosenbrock_10d"],
    "baseline": ["sphere_10d"]
}


def get_optimization_preset(preset_name: str = "all") -> List[OptimizationProblem]:
    """
    Get list of OptimizationProblem instances for benchmarking.
    
    Args:
        preset_name: 'small', 'medium', 'large', 'all', 'multimodal', 'baseline'
    
    Returns:
        List of OptimizationProblem instances
    """
    preset_name = preset_name.lower()
    instance_names = OPTIMIZATION_PRESETS.get(preset_name, OPTIMIZATION_PRESETS['all'])
    
    problems = []
    for name in instance_names:
        try:
            problem = create_optimization_problem(name)
            problems.append(problem)
            logger.info(f"Loaded optimization preset: {name} ({problem.dimension}D)")
        except Exception as e:
            logger.warning(f"Failed to load optimization preset {name}: {e}")
    
    return problems


# ==================== STATISTICAL ANALYSIS UTILITIES ====================

def analyze_optimization_results(
    problem: OptimizationProblem,
    method_results: Dict[str, List[float]],
    computation_times: Dict[str, List[float]],
    function_evals: Dict[str, List[int]]
) -> Dict:
    """
    Comprehensive statistical analysis of optimization results.
    
    Args:
        problem: OptimizationProblem instance
        method_results: Dict mapping method names to lists of best fitness values
        computation_times: Dict mapping method names to lists of computation times
        function_evals: Dict mapping method names to lists of function evaluations
    
    Returns:
        Dictionary with statistical analysis results
    """
    analysis = {
        'problem': problem.name,
        'dimension': problem.dimension,
        'methods': {},
        'comparisons': {}
    }
    
    # Analyze each method
    for method_name, fitness_values in method_results.items():
        error_values = [abs(f - problem.global_optimum) for f in fitness_values]
        
        analysis['methods'][method_name] = {
            'fitness_stats': compute_statistics(fitness_values),
            'error_stats': compute_statistics(error_values),
            'success_rate': success_rate(error_values, 1e-4),
            'time_stats': compute_statistics(computation_times.get(method_name, [])),
            'eval_stats': compute_statistics(function_evals.get(method_name, [])),
            'n_runs': len(fitness_values)
        }
    
    # Pairwise Wilcoxon tests between methods
    method_names = list(method_results.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method_i = method_names[i]
            method_j = method_names[j]
            
            # Ensure equal length (use minimum runs)
            n = min(len(method_results[method_i]), len(method_results[method_j]))
            if n < 2:
                continue
            
            try:
                from utils import wilcoxon_test
                wilcox = wilcoxon_test(
                    method_results[method_i][:n],
                    method_results[method_j][:n]
                )
                
                analysis['comparisons'][f"{method_i}_vs_{method_j}"] = {
                    'method_i': method_i,
                    'method_j': method_j,
                    'p_value': wilcox['p_value'],
                    'significant': wilcox['p_value'] < 0.05,
                    'mean_i': np.mean(method_results[method_i][:n]),
                    'mean_j': np.mean(method_results[method_j][:n]),
                    'better_method': method_i if np.mean(method_results[method_i][:n]) < np.mean(method_results[method_j][:n]) else method_j
                }
            except Exception as e:
                logger.warning(f"Wilcoxon test failed for {method_i} vs {method_j}: {e}")
                continue
    
    return analysis


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Optimization Problem Module Demo ===\n")
    
    # Example 1: Create and evaluate Rastrigin 10D
    try:
        print("1. Rastrigin 10D function evaluation...")
        
        problem = create_optimization_problem("rastrigin_10d")
        print(f"Problem: {problem.name}")
        print(f"Dimension: {problem.dimension}")
        print(f"Bounds: {problem.bounds[0]}")
        print(f"Global optimum: {problem.global_optimum}\n")
        
        # Evaluate at origin (optimum)
        result_opt = problem.evaluate(np.zeros(10))
        print(f"At optimum (origin):")
        print(f"  Fitness: {result_opt['fitness']:.6f}")
        print(f"  Error: {result_opt['error']:.2e}")
        print(f"  Optimal: {result_opt['is_optimal']}\n")
        
        # Evaluate at random point
        np.random.seed(42)
        random_x = np.random.uniform(-5.12, 5.12, 10)
        result_rand = problem.evaluate(random_x)
        print(f"At random point:")
        print(f"  Fitness: {result_rand['fitness']:.4f}")
        print(f"  Error: {result_rand['error']:.4f}")
        print(f"  Function evaluations: {result_rand['function_evaluations']}\n")
        
    except Exception as e:
        print(f"Error in Rastrigin demo: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Rosenbrock 10D with known optimum
    try:
        print("2. Rosenbrock 10D at known optimum (all ones)...")
        
        problem = create_optimization_problem("rosenbrock_10d")
        optimum = np.ones(10)
        
        result = problem.evaluate(optimum)
        print(f"Fitness at optimum: {result['fitness']:.6f}")
        print(f"Error: {result['error']:.2e}")
        print(f"Success (error < 1e-4): {result['is_optimal']}\n")
        
    except Exception as e:
        print(f"Error in Rosenbrock demo: {e}")
    
    # Example 3: Get problem description for LLM
    try:
        print("3. LLM problem prompt for Ackley 20D...")
        
        problem = create_optimization_problem("ackley_20d")
        prompt = problem.get_llm_problem_prompt(preferences={"time_limit": "60 seconds", "priority": "solution quality"})
        
        print(prompt[:500] + "...\n")  # Print first 500 chars
        
    except Exception as e:
        print(f"Error in LLM prompt demo: {e}")
    
    # Example 4: Preset loading
    try:
        print("4. Loading multimodal preset (10D functions)...")
        
        problems = get_optimization_preset("multimodal")
        print(f"Loaded {len(problems)} problems:")
        for p in problems:
            print(f"  - {p.name}")
        
    except Exception as e:
        print(f"Error in preset demo: {e}")
    
    print("\nDemos completed. For full integration, use with orchestrator.py")