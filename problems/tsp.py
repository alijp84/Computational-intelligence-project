"""
TSP Problem Implementation for MetaMind CI Framework
Provides standardized interface for Traveling Salesman Problem instances
with evaluation metrics, instance management, and LLM integration support.
"""

import os
import json
import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_tsp_distance_matrix,
    tsp_tour_length,
    gap_to_optimal,
    two_opt_improvement,
    setup_logger,
    EarlyStopping
)

logger = setup_logger("tsp_problem")


class TSPInstance:
    """Represents a single TSP instance with distance matrix and metadata."""
    
    # Known optimal solutions for benchmark instances (from TSPLIB)
    KNOWN_OPTIMA = {
        'eil51': 426,
        'berlin52': 7542,
        'kroA100': 21282,
        'kroB100': 22141,
        'kroC100': 20749,
        'kroD100': 21294,
        'kroE100': 22068,
        'rat99': 1211,
        'rd100': 7910,
    }
    
    def __init__(
        self,
        name: str,
        distance_matrix: np.ndarray,
        optimal_tour: Optional[List[int]] = None,
        optimal_value: Optional[float] = None,
        description: str = ""
    ):
        self.name = name
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.optimal_tour = optimal_tour
        self.optimal_value = optimal_value or self.KNOWN_OPTIMA.get(name.lower())
        self.description = description
        
        # Validate distance matrix
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(f"Distance matrix must be square, got {distance_matrix.shape}")
        if not np.allclose(distance_matrix.diagonal(), 0):
            logger.warning("Distance matrix diagonal not zero - correcting")
            np.fill_diagonal(distance_matrix, 0)
        
        logger.info(f"Loaded TSP instance '{name}' with {self.n_cities} cities")
    
    @classmethod
    def from_file(cls, filepath: str, name: Optional[str] = None) -> 'TSPInstance':
        """Load TSP instance from TSPLIB or CSV file."""
        filepath = str(filepath)
        if name is None:
            name = Path(filepath).stem
        
        distance_matrix = load_tsp_distance_matrix(filepath)
        optimal_value = cls.KNOWN_OPTIMA.get(name.lower())
        
        return cls(
            name=name,
            distance_matrix=distance_matrix,
            optimal_value=optimal_value,
            description=f"TSP instance loaded from {filepath}"
        )
    
    @classmethod
    def random_euclidean(
        cls,
        n_cities: int,
        seed: Optional[int] = None,
        name: str = "Random"
    ) -> 'TSPInstance':
        """Generate random Euclidean TSP instance."""
        rng = np.random.RandomState(seed) if seed is not None else np.random
        coords = rng.uniform(0, 100, size=(n_cities, 2))
        
        # Compute Euclidean distance matrix
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return cls(
            name=f"{name}{n_cities}",
            distance_matrix=dist_matrix,
            description=f"Random Euclidean TSP with {n_cities} cities"
        )
    
    def get_optimal_value(self) -> Optional[float]:
        """Return known optimal tour length if available."""
        return self.optimal_value
    
    def is_valid_tour(self, tour: List[int]) -> bool:
        """Validate that tour visits each city exactly once and returns to start."""
        if len(tour) != self.n_cities:
            return False
        
        # Check all cities visited exactly once
        if sorted(tour) != list(range(self.n_cities)):
            return False
        
        return True
    
    def evaluate_tour(
        self,
        tour: List[int],
        apply_2opt: bool = False
    ) -> Dict[str, Union[float, List[int], bool]]:
        """
        Evaluate a TSP tour and return comprehensive metrics.
        
        Args:
            tour: List of city indices representing the tour
            apply_2opt: Whether to apply 2-opt local search improvement
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_valid_tour(tour):
            raise ValueError(f"Invalid tour: must contain each city 0-{self.n_cities-1} exactly once")
        
        # Calculate base tour length
        base_length = tsp_tour_length(tour, self.distance_matrix)
        improved_tour = tour
        improved_length = base_length
        
        # Apply 2-opt improvement if requested
        if apply_2opt:
            improved_tour, improved_length = two_opt_improvement(tour, self.distance_matrix)
        
        # Calculate gap to optimum if known
        gap = None
        if self.optimal_value is not None:
            gap = gap_to_optimal(improved_length, self.optimal_value)
        
        return {
            'tour': improved_tour,
            'tour_length': improved_length,
            'base_length': base_length,
            'improved': apply_2opt and (improved_length < base_length - 1e-6),
            'gap_to_optimal': gap,
            'optimal_known': self.optimal_value is not None,
            'optimal_value': self.optimal_value,
            'valid': True
        }
    
    def visualize_tour(
        self,
        tour: List[int],
        title: str = "TSP Tour",
        save_path: Optional[str] = None
    ):
        """Visualize TSP tour (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            # For Euclidean instances only (approximate coordinates)
            # This is a heuristic visualization - actual coordinates not stored
            rng = np.random.RandomState(42)
            coords = rng.uniform(0, 10, size=(self.n_cities, 2))
            
            plt.figure(figsize=(8, 8))
            # Plot cities
            plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, zorder=2)
            
            # Plot tour edges
            for i in range(self.n_cities):
                city1 = tour[i]
                city2 = tour[(i + 1) % self.n_cities]
                plt.plot(
                    [coords[city1, 0], coords[city2, 0]],
                    [coords[city1, 1], coords[city2, 1]],
                    'r-', linewidth=1, alpha=0.6, zorder=1
                )
            
            # Highlight start/end city
            start_city = tour[0]
            plt.scatter(
                coords[start_city, 0], coords[start_city, 1],
                c='red', s=100, marker='*', zorder=3, label='Start/End'
            )
            
            plt.title(f"{title}\nTour Length: {tsp_tour_length(tour, self.distance_matrix):.2f}")
            plt.legend()
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available - skipping visualization")
        except Exception as e:
            logger.error(f"Error visualizing tour: {e}")


class TSProblem:
    """
    Standardized TSP problem interface for MetaMind orchestrator.
    Implements the problem API expected by the LLM orchestrator.
    """
    
    def __init__(self, instance: TSPInstance):
        self.instance = instance
        self.name = instance.name
        self.n_cities = instance.n_cities
        self.distance_matrix = instance.distance_matrix
        self.optimal_value = instance.get_optimal_value()
        self.history = []  # Store evaluation history for convergence tracking
    
    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        data_dir: str = "data/tsp"
    ) -> 'TSProblem':
        """
        Load a preset TSP instance by name.
        
        Supported presets:
          - 'eil51', 'berlin52', 'kroA100' (TSPLIB instances)
          - 'random30', 'random50' (random Euclidean instances)
        """
        preset_name_lower = preset_name.lower()
        
        # Handle random instances
        if preset_name_lower.startswith('random'):
            n_cities = int(''.join(filter(str.isdigit, preset_name_lower)))
            instance = TSPInstance.random_euclidean(
                n_cities=n_cities,
                seed=42,  # Deterministic random instances for reproducibility
                name=f"Random{n_cities}"
            )
            return cls(instance)
        
        # Handle TSPLIB instances
        filepath = os.path.join(data_dir, f"{preset_name_lower}.tsp")
        if not os.path.exists(filepath):
            # Try .txt extension as fallback
            filepath = os.path.join(data_dir, f"{preset_name_lower}.txt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"TSP instance file not found: {filepath}. "
                    f"Please download TSPLIB instances to {data_dir}/"
                )
        
        instance = TSPInstance.from_file(filepath, name=preset_name)
        return cls(instance)
    
    def get_problem_description(self) -> Dict:
        """Return structured problem description for LLM input."""
        return {
            "problem_type": "combinatorial_optimization",
            "domain": "routing",
            "name": self.name,
            "n_cities": self.n_cities,
            "symmetric": np.allclose(self.distance_matrix, self.distance_matrix.T),
            "optimal_known": self.optimal_value is not None,
            "optimal_value": self.optimal_value,
            "constraints": {
                "visit_each_city_once": True,
                "return_to_start": True
            },
            "objective": "minimize_total_distance",
            "search_space_size": f"{self.n_cities}! ≈ {math.factorial(self.n_cities):.2e}"
        }
    
    def evaluate(
        self,
        solution: Union[List[int], np.ndarray],
        **kwargs
    ) -> Dict:
        """
        Evaluate a candidate solution (tour).
        
        Args:
            solution: Tour as list/array of city indices
            **kwargs: Additional parameters (e.g., 'apply_2opt')
            
        Returns:
            Dictionary with evaluation results matching project spec format
        """
        # Convert to list if numpy array
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        
        # Validate and evaluate tour
        result = self.instance.evaluate_tour(
            tour=solution,
            apply_2opt=kwargs.get('apply_2opt', False)
        )
        
        # Record in history for convergence tracking
        self.history.append({
            'iteration': len(self.history),
            'tour_length': result['tour_length'],
            'tour': result['tour'][:5] + ['...'] if len(result['tour']) > 10 else result['tour']  # Truncate for storage
        })
        
        return result
    
    def is_optimal(self, tour_length: float, tolerance: float = 1e-6) -> bool:
        """Check if solution is within tolerance of known optimum."""
        if self.optimal_value is None:
            return False
        return abs(tour_length - self.optimal_value) <= tolerance
    
    def get_convergence_history(self) -> List[float]:
        """Return history of best tour lengths over evaluations."""
        if not self.history:
            return []
        # Extract tour lengths and compute running minimum
        lengths = [h['tour_length'] for h in self.history]
        running_min = []
        current_min = float('inf')
        for length in lengths:
            current_min = min(current_min, length)
            running_min.append(current_min)
        return running_min
    
    def reset_history(self):
        """Clear evaluation history (for new runs)."""
        self.history = []
    
    def get_evaluation_metrics(
        self,
        tour: List[int],
        computation_time: float,
        iterations: int = 1
    ) -> Dict:
        """
        Compute all required evaluation metrics for a solution.
        
        Returns:
            Dictionary with metrics matching Section 5.1 of project doc
        """
        eval_result = self.evaluate(tour)
        
        metrics = {
            "tour_length": eval_result['tour_length'],
            "computation_time": computation_time,
            "iterations": iterations,
            "valid": eval_result['valid']
        }
        
        # Add optimality metrics if known
        if self.optimal_value is not None:
            metrics.update({
                "known_optimal": self.optimal_value,
                "gap_percentage": eval_result['gap_to_optimal'],
                "is_optimal": self.is_optimal(eval_result['tour_length'])
            })
        
        # Add convergence metrics
        history = self.get_convergence_history()
        if history:
            metrics.update({
                "convergence_history": history,
                "final_improvement": history[-1] if history else None,
                "iterations_to_90pct": self._iterations_to_percentile(history, 0.90)
            })
        
        return metrics
    
    def _iterations_to_percentile(self, history: List[float], percentile: float) -> Optional[int]:
        """Calculate iterations needed to reach X% of final solution quality."""
        if not history or len(history) < 2:
            return None
        
        final_value = history[-1]
        initial_value = history[0]
        target_value = initial_value - (initial_value - final_value) * percentile
        
        for i, value in enumerate(history):
            if value <= target_value:
                return i + 1  # 1-indexed iterations
        
        return len(history)
    
    def format_for_llm_feedback(self, execution_results: Dict) -> Dict:
        """
        Format execution results for LLM interpretation (Step 5 in project doc).
        
        Args:
            execution_results: Dictionary from method execution containing:
                - method_used
                - best_solution
                - best_fitness
                - computation_time
                - convergence_history
                - iterations_completed (optional)
        
        Returns:
            Formatted dictionary matching project specification
        """
        formatted = {
            "method_used": execution_results["method_used"],
            "best_solution": execution_results["best_solution"][:10] + ["..."] 
                if len(execution_results["best_solution"]) > 15 
                else execution_results["best_solution"],
            "best_fitness": float(execution_results["best_fitness"]),
            "computation_time": float(execution_results["computation_time"]),
            "convergence_history": execution_results.get("convergence_history", [])[-10:],  # Last 10 iterations
            "iterations_completed": execution_results.get("iterations_completed", 
                                                          len(execution_results.get("convergence_history", [])))
        }
        
        # Add optimality metrics if available
        if self.optimal_value is not None:
            formatted.update({
                "known_optimal": float(self.optimal_value),
                "gap_percentage": float(gap_to_optimal(
                    execution_results["best_fitness"], 
                    self.optimal_value
                ))
            })
        
        return formatted
    
    def get_llm_problem_prompt(self, preferences: Optional[Dict] = None) -> str:
        """
        Generate formatted problem description for LLM input (Step 1 in project doc).
        
        Args:
            preferences: Optional dict with keys like 'time_limit', 'priority' (speed vs quality)
        
        Returns:
            Formatted string prompt for LLM
        """
        desc = self.get_problem_description()
        preferences = preferences or {}
        
        prompt = f"""Problem: Traveling Salesman Problem (TSP)
Instance: {desc['name']}
Cities: {desc['n_cities']}
Type: {'Symmetric' if desc['symmetric'] else 'Asymmetric'} Euclidean TSP
Objective: Minimize total tour distance (visit each city exactly once and return to start)

Problem Characteristics:
- Search space size: {desc['search_space_size']}
- Optimal solution known: {'Yes' if desc['optimal_known'] else 'No'}
"""
        
        if desc['optimal_known']:
            prompt += f"- Known optimal tour length: {desc['optimal_value']}\n"
        
        if preferences:
            prompt += "\nUser Preferences:\n"
            for key, value in preferences.items():
                prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt += """
Available CI Methods:
- Ant Colony Optimization (ACO): Naturally suited for graph/routing problems, uses pheromone trails
- Genetic Algorithm (GA): Effective for combinatorial optimization, uses crossover/mutation
- Particle Swarm Optimization (PSO): Can be adapted for discrete spaces via mapping techniques
- Hopfield Network: Neural approach for optimization, may struggle with larger instances

Task:
Analyze this TSP instance and recommend the most appropriate CI method with justified parameter configuration.
Consider problem size, known characteristics, and user preferences when making your recommendation.
"""
        
        return prompt


# ==================== PRESET INSTANCE REGISTRY ====================

TSP_PRESETS = {
    "small": ["eil51", "berlin52", "random30"],
    "medium": ["kroA100", "kroB100", "random50"],
    "large": ["rat99", "rd100"],  # Extend as needed
    "all": ["eil51", "berlin52", "kroA100", "random30", "random50"]
}


def get_tsp_preset(preset_name: str = "all") -> List[TSProblem]:
    """
    Get list of TSProblem instances for benchmarking.
    
    Args:
        preset_name: 'small', 'medium', 'large', or 'all'
    
    Returns:
        List of TSProblem instances
    """
    preset_name = preset_name.lower()
    instance_names = TSP_PRESETS.get(preset_name, TSP_PRESETS['all'])
    
    problems = []
    for name in instance_names:
        try:
            problem = TSProblem.from_preset(name)
            problems.append(problem)
            logger.info(f"Loaded TSP preset: {name} ({problem.n_cities} cities)")
        except Exception as e:
            logger.warning(f"Failed to load TSP preset {name}: {e}")
    
    return problems


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Load and solve a small TSP instance
    print("=== TSP Problem Module Demo ===\n")
    
    # Load eil51 instance
    try:
        problem = TSProblem.from_preset("eil51")
        print(f"Loaded instance: {problem.name} ({problem.n_cities} cities)")
        print(f"Optimal tour length: {problem.optimal_value}\n")
        
        # Generate random valid tour
        random_tour = list(range(problem.n_cities))
        random.shuffle(random_tour)
        
        # Evaluate tour
        result = problem.evaluate(random_tour, apply_2opt=True)
        print(f"Random tour length: {result['base_length']:.2f}")
        print(f"After 2-opt improvement: {result['tour_length']:.2f}")
        print(f"Gap to optimal: {result['gap_to_optimal']:.2f}%\n")
        
        # Show LLM prompt
        print("Sample LLM Prompt:")
        print("=" * 60)
        print(problem.get_llm_problem_prompt(preferences={"time_limit": "60 seconds", "priority": "solution quality"}))
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in demo: {e}")
        print("\nNote: TSPLIB instances must be downloaded to data/tsp/ directory")
        print("Download from: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/")