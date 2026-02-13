"""
Evolutionary Methods for MetaMind CI Framework
Implements GA, PSO, ACO, and GP with standardized interfaces for optimization problems.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable, Union
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
import math
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    setup_logger,
    EarlyStopping,
    tsp_tour_length,
    two_opt_improvement,
    normalize_array
)

logger = setup_logger("evolutionary_methods")


# ==================== BASE EVOLUTIONARY METHOD CLASS ====================

class EvolutionaryMethod(ABC):
    """Abstract base class for all evolutionary methods with standardized interface."""
    
    def __init__(self, **params):
        self.params = params
        self.convergence_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.start_time = None
        self.elapsed_time = 0.0
        self.iterations_completed = 0
        self.name = self.__class__.__name__
        
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
    
    def _update_convergence(self, fitness: float, solution: Union[List, np.ndarray]):
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
            "iterations_completed": self.iterations_completed
        }


# ==================== GENETIC ALGORITHM (GA) ====================

class GeneticAlgorithm(EvolutionaryMethod):
    """
    Genetic Algorithm for combinatorial and continuous optimization.
    Supports permutation encoding (TSP) and real-valued encoding (function optimization).
    """
    
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 500,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 2,
        crossover_type: str = "pmx",  # 'pmx', 'ox', 'cx' for permutations; 'sbx' for real
        mutation_type: str = "swap",  # 'swap', 'inversion' for permutations; 'gaussian' for real
        **kwargs
    ):
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection=selection,
            tournament_size=tournament_size,
            elitism=elitism,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            **kwargs
        )
        self.population = []
        self.fitness_values = []
        self.bounds = None  # Will be set in solve()
    
    def _initialize_population(self, problem: object, problem_type: str) -> List:
        """Initialize population based on problem type."""
        n = self.params['population_size']
        
        if problem_type == "tsp":
            # Permutation encoding for TSP
            n_cities = problem.n_cities
            return [list(np.random.permutation(n_cities)) for _ in range(n)]
        
        elif problem_type == "function_optimization":
            # Real-valued encoding for continuous optimization
            # FIX: Use self.bounds set in solve() instead of undefined kwargs
            if self.bounds is None:
                # Default Rastrigin bounds for 10 dimensions
                self.bounds = [(-5.12, 5.12)] * 10
                logger.warning("No bounds provided for function optimization; using default Rastrigin bounds (10D)")
            
            dim = len(self.bounds)
            population = []
            for _ in range(n):
                individual = np.array([
                    random.uniform(low, high) for (low, high) in self.bounds
                ])
                population.append(individual)
            return population
        
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
    
    def _evaluate_population(self, problem: object, population: List) -> List[float]:
        """Evaluate fitness of entire population."""
        fitness_values = []
        for individual in population:
            result = problem.evaluate(individual)
            # For minimization problems (all our cases), lower fitness is better
            if 'tour_length' in result:
                fitness = result['tour_length']
            elif 'fitness' in result:
                fitness = result['fitness']
            elif 'value' in result:
                fitness = result['value']
            else:
                # Fallback: assume result is the fitness value itself
                fitness = result if isinstance(result, (int, float)) else float('inf')
            fitness_values.append(fitness)
        return fitness_values
    
    def _selection(self, population: List, fitness_values: List) -> List:
        """Select parents using specified selection method."""
        selection_method = self.params['selection']
        tournament_size = self.params['tournament_size']
        n_parents = len(population)
        parents = []
        
        if selection_method == "tournament":
            for _ in range(n_parents):
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_values[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]  # Minimization
                parents.append(deepcopy(population[winner_idx]))
        
        elif selection_method == "roulette":
            # Fitness proportionate selection (for minimization, use inverse fitness)
            max_fitness = max(fitness_values)
            inverted_fitness = [max_fitness - f + 1e-6 for f in fitness_values]  # Avoid zero
            total_fitness = sum(inverted_fitness)
            probabilities = [f / total_fitness for f in inverted_fitness]
            
            for _ in range(n_parents):
                r = random.random()
                cumulative = 0.0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        parents.append(deepcopy(population[i]))
                        break
        
        elif selection_method == "rank":
            # Rank-based selection
            sorted_indices = np.argsort(fitness_values)
            ranks = np.zeros(len(population))
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = len(population) - rank  # Higher rank for better fitness
            
            total_rank = sum(ranks)
            probabilities = [r / total_rank for r in ranks]
            
            for _ in range(n_parents):
                r = random.random()
                cumulative = 0.0
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        parents.append(deepcopy(population[i]))
                        break
        
        return parents
    
    def _crossover(self, parent1: Union[List, np.ndarray], parent2: Union[List, np.ndarray]) -> Tuple:
        """Perform crossover based on problem type."""
        crossover_type = self.params['crossover_type']
        crossover_rate = self.params['crossover_rate']
        
        if random.random() > crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        if isinstance(parent1, list):  # Permutation (TSP)
            n = len(parent1)
            if crossover_type == "pmx":
                # Partially Mapped Crossover
                point1, point2 = sorted(random.sample(range(n), 2))
                child1 = [-1] * n
                child2 = [-1] * n
                
                # Copy segment
                child1[point1:point2] = parent1[point1:point2]
                child2[point1:point2] = parent2[point1:point2]
                
                # Fill remaining positions with mapping
                def fill_child(child, parent, other_parent, p1, p2):
                    mapping = {}
                    for i in range(p1, p2):
                        mapping[other_parent[i]] = parent[i]
                    
                    for i in range(n):
                        if i < p1 or i >= p2:
                            value = parent[i]
                            while value in mapping:
                                value = mapping[value]
                            child[i] = value
                
                fill_child(child1, parent2, parent1, point1, point2)
                fill_child(child2, parent1, parent2, point1, point2)
                
                return child1, child2
            
            elif crossover_type == "ox":
                # Order Crossover
                point1, point2 = sorted(random.sample(range(n), 2))
                child1 = [-1] * n
                child2 = [-1] * n
                
                # Copy segment
                child1[point1:point2] = parent1[point1:point2]
                child2[point1:point2] = parent2[point1:point2]
                
                # Fill remaining in order of other parent
                def fill_remaining(child, parent, start_idx):
                    idx = start_idx
                    for gene in parent:
                        if gene not in child:
                            while child[idx] != -1:
                                idx = (idx + 1) % n
                            child[idx] = gene
                
                fill_remaining(child1, parent2, point2 % n)
                fill_remaining(child2, parent1, point2 % n)
                
                return child1, child2
            
            elif crossover_type == "cx":
                # Cycle Crossover
                child1 = [-1] * n
                child2 = [-1] * n
                
                # Find cycles
                index = 0
                cycle_indices = []
                while True:
                    cycle_indices.append(index)
                    index = parent1.index(parent2[index])
                    if index == cycle_indices[0]:
                        break
                
                # Copy cycle to child1, rest to child2
                for i in range(n):
                    if i in cycle_indices:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                    else:
                        child1[i] = parent2[i]
                        child2[i] = parent1[i]
                
                return child1, child2
        
        else:  # Real-valued encoding
            if crossover_type == "sbx":
                # Simulated Binary Crossover
                eta = 20  # Distribution index
                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent2)
                
                for i in range(len(parent1)):
                    if random.random() <= 0.5:
                        beta = (2 * random.random()) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                        child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                    else:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                
                # Clip to bounds
                if self.bounds is not None:
                    for i, (low, high) in enumerate(self.bounds):
                        child1[i] = np.clip(child1[i], low, high)
                        child2[i] = np.clip(child2[i], low, high)
                
                return child1, child2
            
            else:  # Default single-point crossover
                point = random.randint(1, len(parent1) - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
                # Clip to bounds
                if self.bounds is not None:
                    for i, (low, high) in enumerate(self.bounds):
                        if i < len(child1):
                            child1[i] = np.clip(child1[i], low, high)
                        if i < len(child2):
                            child2[i] = np.clip(child2[i], low, high)
                return child1, child2
    
    def _mutation(self, individual: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
        """Perform mutation based on problem type."""
        mutation_rate = self.params['mutation_rate']
        mutation_type = self.params['mutation_type']
        
        if random.random() > mutation_rate:
            return individual
        
        if isinstance(individual, list):  # Permutation (TSP)
            n = len(individual)
            mutated = individual.copy()
            
            if mutation_type == "swap":
                # Swap two random positions
                i, j = random.sample(range(n), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            
            elif mutation_type == "inversion":
                # Invert a random segment
                i, j = sorted(random.sample(range(n), 2))
                mutated[i:j] = reversed(mutated[i:j])
            
            return mutated
        
        else:  # Real-valued encoding
            mutated = individual.copy()
            dim = len(individual)
            
            if mutation_type == "gaussian":
                # Gaussian perturbation on random dimension
                idx = random.randint(0, dim - 1)
                if self.bounds is not None and idx < len(self.bounds):
                    low, high = self.bounds[idx]
                    sigma = 0.1 * (high - low + 1e-8)
                    mutated[idx] += random.gauss(0, sigma)
                    mutated[idx] = np.clip(mutated[idx], low, high)
                else:
                    mutated[idx] += random.gauss(0, 0.1)
            else:  # Uniform mutation
                idx = random.randint(0, dim - 1)
                if self.bounds is not None and idx < len(self.bounds):
                    low, high = self.bounds[idx]
                    mutated[idx] = random.uniform(low, high)
            
            return mutated
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve problem using Genetic Algorithm.
        
        Args:
            problem: Problem instance with evaluate() method
            max_time: Optional time limit in seconds
            callback: Optional callback function (iteration, best_fitness, population)
            **kwargs: Additional parameters (e.g., 'problem_type', 'bounds')
        """
        problem_type = kwargs.get('problem_type', 'tsp')
        self.bounds = kwargs.get('bounds', None)  # Store bounds for _initialize_population
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=50, min_delta=1e-6, mode='min')
        
        # Initialize population
        self.population = self._initialize_population(problem, problem_type)
        self.fitness_values = self._evaluate_population(problem, self.population)
        
        # Update convergence with initial best
        best_idx = np.argmin(self.fitness_values)
        self._update_convergence(self.fitness_values[best_idx], self.population[best_idx])
        
        # Main GA loop
        for generation in range(self.params['generations']):
            if self._check_time_limit(max_time):
                logger.info(f"GA stopped due to time limit at generation {generation}")
                break
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(self.fitness_values)[:self.params['elitism']]
            elites = [deepcopy(self.population[i]) for i in elite_indices]
            
            # Selection
            parents = self._selection(self.population, self.fitness_values)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                offspring.append(self._mutation(child1))
                offspring.append(self._mutation(child2))
            
            # Replace population with elites + offspring
            self.population = elites + offspring[:len(self.population) - len(elites)]
            self.fitness_values = self._evaluate_population(problem, self.population)
            
            # Update convergence
            best_idx = np.argmin(self.fitness_values)
            self._update_convergence(self.fitness_values[best_idx], self.population[best_idx])
            
            # Callback for progress monitoring
            if callback:
                callback(generation, self.best_fitness, self.population)
            
            # Early stopping check
            if early_stop(self.best_fitness):
                logger.info(f"GA converged early at generation {generation}")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Apply 2-opt local search for TSP if requested
        if problem_type == "tsp" and kwargs.get('apply_2opt', False):
            improved_tour, improved_length = two_opt_improvement(
                self.best_solution, 
                problem.distance_matrix
            )
            self.best_solution = improved_tour
            self.best_fitness = improved_length
            self.convergence_history.append(improved_length)
        
        return self.get_results()


# ==================== PARTICLE SWARM OPTIMIZATION (PSO) ====================

class ParticleSwarmOptimization(EvolutionaryMethod):
    """
    Particle Swarm Optimization for continuous optimization problems.
    """
    
    def __init__(
        self,
        n_particles: int = 50,
        max_iterations: int = 500,
        w: float = 0.7,           # inertia weight
        c1: float = 1.5,          # cognitive coefficient
        c2: float = 1.5,          # social coefficient
        w_decay: bool = True,     # linearly decrease inertia weight
        velocity_clamp: float = 0.5,  # fraction of search range
        **kwargs
    ):
        super().__init__(
            n_particles=n_particles,
            max_iterations=max_iterations,
            w=w,
            c1=c1,
            c2=c2,
            w_decay=w_decay,
            velocity_clamp=velocity_clamp,
            **kwargs
        )
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_fitness = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.bounds = None
    
    def _initialize_swarm(self, bounds: List[Tuple[float, float]]):
        """Initialize particles and velocities within bounds."""
        self.bounds = bounds
        dim = len(bounds)
        self.particles = []
        self.velocities = []
        
        for _ in range(self.params['n_particles']):
            # Initialize position uniformly within bounds
            position = np.array([random.uniform(low, high) for low, high in bounds])
            self.particles.append(position)
            
            # Initialize velocity within [-vmax, vmax] where vmax = velocity_clamp * range
            velocity = np.array([
                random.uniform(
                    -self.params['velocity_clamp'] * (high - low),
                    self.params['velocity_clamp'] * (high - low)
                )
                for low, high in bounds
            ])
            self.velocities.append(velocity)
        
        self.personal_best_positions = [p.copy() for p in self.particles]
        self.personal_best_fitness = [float('inf')] * self.params['n_particles']
        self.global_best_position = None
        self.global_best_fitness = float('inf')
    
    def _update_velocity(self, particle_idx: int, iteration: int):
        """Update particle velocity using standard PSO equation."""
        w = self.params['w']
        if self.params['w_decay']:
            w = self.params['w'] * (1 - iteration / self.params['max_iterations'])
        
        c1 = self.params['c1']
        c2 = self.params['c2']
        
        r1 = random.random()
        r2 = random.random()
        
        cognitive = c1 * r1 * (self.personal_best_positions[particle_idx] - self.particles[particle_idx])
        social = c2 * r2 * (self.global_best_position - self.particles[particle_idx])
        
        new_velocity = w * self.velocities[particle_idx] + cognitive + social
        
        # Velocity clamping
        for i, (low, high) in enumerate(self.bounds):
            vmax = self.params['velocity_clamp'] * (high - low)
            new_velocity[i] = np.clip(new_velocity[i], -vmax, vmax)
        
        self.velocities[particle_idx] = new_velocity
    
    def _update_position(self, particle_idx: int):
        """Update particle position and apply boundary handling."""
        new_position = self.particles[particle_idx] + self.velocities[particle_idx]
        
        # Boundary handling: reflect at boundaries with damping
        for i, (low, high) in enumerate(self.bounds):
            if new_position[i] < low:
                new_position[i] = low + (low - new_position[i]) * 0.5  # Reflect with damping
                self.velocities[particle_idx][i] *= -0.5
            elif new_position[i] > high:
                new_position[i] = high - (new_position[i] - high) * 0.5
                self.velocities[particle_idx][i] *= -0.5
        
        self.particles[particle_idx] = new_position
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve continuous optimization problem using PSO.
        
        Args:
            problem: Problem instance with evaluate() method expecting real-valued vectors
            max_time: Optional time limit in seconds
            callback: Optional callback function (iteration, best_fitness, particles)
            **kwargs: Must include 'bounds' - list of (min, max) tuples for each dimension
        """
        bounds = kwargs.get('bounds')
        if bounds is None:
            raise ValueError("PSO requires 'bounds' parameter specifying search space")
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=30, min_delta=1e-8, mode='min')
        self._initialize_swarm(bounds)
        
        # Evaluate initial swarm
        for i, particle in enumerate(self.particles):
            result = problem.evaluate(particle)
            fitness = result.get('fitness', result.get('value', float('inf')))
            self.personal_best_fitness[i] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.copy()
        
        self._update_convergence(self.global_best_fitness, self.global_best_position)
        
        # Main PSO loop
        for iteration in range(self.params['max_iterations']):
            if self._check_time_limit(max_time):
                logger.info(f"PSO stopped due to time limit at iteration {iteration}")
                break
            
            for i in range(self.params['n_particles']):
                # Update velocity and position
                self._update_velocity(i, iteration)
                self._update_position(i)
                
                # Evaluate new position
                result = problem.evaluate(self.particles[i])
                fitness = result.get('fitness', result.get('value', float('inf')))
                
                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                    
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.particles[i].copy()
            
            # Update convergence history
            self._update_convergence(self.global_best_fitness, self.global_best_position)
            
            # Callback for progress monitoring
            if callback:
                callback(iteration, self.best_fitness, self.particles)
            
            # Early stopping check
            if early_stop(self.global_best_fitness):
                logger.info(f"PSO converged early at iteration {iteration}")
                break
        
        self.elapsed_time = time.time() - self.start_time
        return self.get_results()


# ==================== ANT COLONY OPTIMIZATION (ACO) ====================

class AntColonyOptimization(EvolutionaryMethod):
    """
    Ant Colony Optimization specifically designed for TSP problems.
    """
    
    def __init__(
        self,
        n_ants: int = 50,
        max_iterations: int = 500,
        alpha: float = 1.0,           # pheromone importance
        beta: float = 2.0,            # heuristic importance
        evaporation_rate: float = 0.5,
        q: float = 1.0,               # pheromone deposit factor
        initial_pheromone: float = 0.1,
        local_search: bool = True,    # apply 2-opt improvement
        **kwargs
    ):
        super().__init__(
            n_ants=n_ants,
            max_iterations=max_iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            q=q,
            initial_pheromone=initial_pheromone,
            local_search=local_search,
            **kwargs
        )
        self.pheromone = None
        self.heuristic = None
        self.n_cities = None
        self.distance_matrix = None
    
    def _initialize_pheromone(self, n_cities: int):
        """Initialize pheromone matrix with uniform values."""
        self.pheromone = np.full((n_cities, n_cities), self.params['initial_pheromone'])
        np.fill_diagonal(self.pheromone, 0)  # No self-loops
    
    def _compute_heuristic(self, distance_matrix: np.ndarray):
        """Compute heuristic information (eta = 1/distance)."""
        # Avoid division by zero
        safe_distances = distance_matrix.copy()
        safe_distances[safe_distances == 0] = 1e-10
        self.heuristic = 1.0 / safe_distances
    
    def _construct_solution(self) -> List[int]:
        """Construct a tour using ant colony probabilistic rule."""
        n = self.n_cities
        tour = [random.randint(0, n - 1)]  # Start from random city
        visited = set(tour)
        
        for _ in range(n - 1):
            current = tour[-1]
            unvisited = [i for i in range(n) if i not in visited]
            
            # Calculate probabilities for next city
            probs = []
            for city in unvisited:
                tau = self.pheromone[current, city] ** self.params['alpha']
                eta = self.heuristic[current, city] ** self.params['beta']
                probs.append(tau * eta)
            
            # Normalize probabilities
            total = sum(probs)
            if total == 0:
                probs = [1.0 / len(unvisited)] * len(unvisited)
            else:
                probs = [p / total for p in probs]
            
            # Select next city using roulette wheel
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probs):
                cumulative += prob
                if r <= cumulative:
                    next_city = unvisited[i]
                    break
            else:
                next_city = unvisited[-1]
            
            tour.append(next_city)
            visited.add(next_city)
        
        return tour
    
    def _update_pheromone(self, ants: List[List[int]], fitness_values: List[float]):
        """Update pheromone matrix using elite ant strategy."""
        # Evaporation
        self.pheromone *= (1 - self.params['evaporation_rate'])
        
        # Deposit pheromone - use best ant only (elite strategy)
        best_ant_idx = np.argmin(fitness_values)
        best_tour = ants[best_ant_idx]
        best_fitness = fitness_values[best_ant_idx]
        
        # Deposit pheromone on edges of best tour
        for i in range(self.n_cities):
            j = (i + 1) % self.n_cities
            city_i = best_tour[i]
            city_j = best_tour[j]
            deposit = self.params['q'] / (best_fitness + 1e-10)  # Avoid division by zero
            self.pheromone[city_i, city_j] += deposit
            self.pheromone[city_j, city_i] += deposit  # Symmetric TSP
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve TSP using Ant Colony Optimization.
        
        Args:
            problem: TSP problem instance with distance_matrix attribute
            max_time: Optional time limit in seconds
            callback: Optional callback function (iteration, best_fitness, pheromone)
            **kwargs: Additional parameters
        """
        self.n_cities = problem.n_cities
        self.distance_matrix = problem.distance_matrix
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=40, min_delta=1e-4, mode='min')
        
        # Initialize pheromone and heuristic
        self._initialize_pheromone(self.n_cities)
        self._compute_heuristic(self.distance_matrix)
        
        # Main ACO loop
        for iteration in range(self.params['max_iterations']):
            if self._check_time_limit(max_time):
                logger.info(f"ACO stopped due to time limit at iteration {iteration}")
                break
            
            # Construct solutions for all ants
            ants = [self._construct_solution() for _ in range(self.params['n_ants'])]
            
            # Evaluate solutions
            fitness_values = []
            for tour in ants:
                result = problem.evaluate(tour, apply_2opt=False)  # Apply 2-opt later globally
                fitness_values.append(result['tour_length'])
            
            # Update pheromone
            self._update_pheromone(ants, fitness_values)
            
            # Track best solution
            best_idx = np.argmin(fitness_values)
            best_tour = ants[best_idx]
            best_fitness = fitness_values[best_idx]
            
            # Apply 2-opt local search to best solution if enabled
            if self.params['local_search']:
                improved_tour, improved_fitness = two_opt_improvement(
                    best_tour, 
                    self.distance_matrix
                )
                if improved_fitness < best_fitness:
                    best_tour = improved_tour
                    best_fitness = improved_fitness
            
            # Update convergence
            self._update_convergence(best_fitness, best_tour)
            
            # Callback for progress monitoring
            if callback:
                callback(iteration, self.best_fitness, self.pheromone.copy())
            
            # Early stopping check
            if early_stop(self.best_fitness):
                logger.info(f"ACO converged early at iteration {iteration}")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Final 2-opt refinement on best solution
        if self.params['local_search']:
            final_tour, final_fitness = two_opt_improvement(
                self.best_solution, 
                self.distance_matrix
            )
            if final_fitness < self.best_fitness:
                self.best_solution = final_tour
                self.best_fitness = final_fitness
                self.convergence_history.append(final_fitness)
        
        return self.get_results()


# ==================== GENETIC PROGRAMMING (GP) ====================

class GeneticProgramming(EvolutionaryMethod):
    """
    Genetic Programming for symbolic regression and function finding.
    """
    
    def __init__(
        self,
        population_size: int = 200,
        generations: int = 50,
        max_depth: int = 6,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        function_set: List[str] = None,
        terminal_set: List[str] = None,
        parsimony_coefficient: float = 0.001,
        **kwargs
    ):
        super().__init__(
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            parsimony_coefficient=parsimony_coefficient,
            **kwargs
        )
        self.function_set = function_set or ["+", "-", "*", "/"]
        self.terminal_set = terminal_set or ["x", "const"]
        self.population = []
        self.fitness_values = []
    
    class Node:
        """Tree node for GP representation."""
        def __init__(self, value: str, children: List = None):
            self.value = value
            self.children = children if children else []
        
        def copy(self):
            """Deep copy of subtree."""
            return GeneticProgramming.Node(
                self.value,
                [child.copy() for child in self.children]
            )
        
        def size(self) -> int:
            """Count nodes in subtree."""
            return 1 + sum(child.size() for child in self.children)
        
        def depth(self) -> int:
            """Compute depth of subtree."""
            if not self.children:
                return 0
            return 1 + max(child.depth() for child in self.children)
        
        def __str__(self):
            if not self.children:
                return str(self.value)
            if len(self.children) == 1:
                return f"({self.value} {self.children[0]})"
            return f"({self.value} {' '.join(str(c) for c in self.children)})"
    
    def _ramp_half_and_half(self) -> Node:
        """Generate random tree using ramped half-and-half method."""
        method = random.choice(['full', 'grow'])
        max_depth = random.randint(2, self.params['max_depth'])
        return self._generate_tree(0, max_depth, method)
    
    def _generate_tree(self, depth: int, max_depth: int, method: str) -> Node:
        """Recursively generate random tree."""
        if depth == max_depth or (method == 'grow' and random.random() < 0.3):
            # Terminal node
            terminal = random.choice(self.terminal_set)
            if terminal == "const":
                return self.Node(str(round(random.uniform(-5, 5), 2)))
            return self.Node(terminal)
        else:
            # Function node
            func = random.choice(self.function_set)
            if func in ["+", "-", "*", "/"]:
                left = self._generate_tree(depth + 1, max_depth, method)
                right = self._generate_tree(depth + 1, max_depth, method)
                return self.Node(func, [left, right])
            elif func in ["sin", "cos", "exp", "log"]:
                child = self._generate_tree(depth + 1, max_depth, method)
                return self.Node(func, [child])
            else:
                # Unary functions or variables
                child = self._generate_tree(depth + 1, max_depth, method)
                return self.Node(func, [child])
    
    def _evaluate_tree(self, tree: Node, x: float) -> float:
        """Evaluate tree expression for given input x."""
        try:
            if not tree.children:  # Terminal
                if tree.value == "x":
                    return x
                else:  # Constant
                    return float(tree.value)
            
            # Evaluate children first
            child_values = [self._evaluate_tree(child, x) for child in tree.children]
            
            # Apply function
            func = tree.value
            if func == "+":
                return child_values[0] + child_values[1]
            elif func == "-":
                return child_values[0] - child_values[1]
            elif func == "*":
                return child_values[0] * child_values[1]
            elif func == "/":
                # Protected division
                return child_values[0] / child_values[1] if abs(child_values[1]) > 1e-6 else 1e-6
            elif func == "sin":
                return math.sin(child_values[0])
            elif func == "cos":
                return math.cos(child_values[0])
            elif func == "exp":
                # Avoid overflow
                val = min(child_values[0], 10)
                return math.exp(val)
            elif func == "log":
                return math.log(abs(child_values[0]) + 1e-6)
            else:
                return child_values[0]  # Default fallback
        except (OverflowError, ValueError, ZeroDivisionError, RecursionError):
            return 1e10  # Penalize invalid expressions
    
    def _fitness(self, tree: Node, X: np.ndarray, y_true: np.ndarray) -> float:
        """Compute fitness as RMSE with parsimony pressure."""
        try:
            y_pred = np.array([self._evaluate_tree(tree, x) for x in X])
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            # Parsimony pressure: penalize large trees
            complexity_penalty = self.params['parsimony_coefficient'] * tree.size()
            return rmse + complexity_penalty
        except:
            return 1e10
    
    def _subtree_crossover(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        """Perform subtree crossover between two parents."""
        if random.random() > self.params['crossover_rate']:
            return parent1.copy(), parent2.copy()
        
        # Find random subtrees to swap
        nodes1 = self._get_all_nodes(parent1)
        nodes2 = self._get_all_nodes(parent2)
        
        if not nodes1[1:] or not nodes2[1:]:  # Skip root
            return parent1.copy(), parent2.copy()
        
        subtree1 = random.choice(nodes1[1:])  # Exclude root
        subtree2 = random.choice(nodes2[1:])
        
        # Copy parents and swap subtrees
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Find and replace subtree1 in child1 with copy of subtree2
        self._replace_subtree(child1, subtree1, subtree2.copy())
        self._replace_subtree(child2, subtree2, subtree1.copy())
        
        return child1, child2
    
    def _subtree_mutation(self, parent: Node) -> Node:
        """Perform subtree mutation (replace random subtree with new random tree)."""
        if random.random() > self.params['mutation_rate']:
            return parent.copy()
        
        # Find random subtree to replace
        nodes = self._get_all_nodes(parent)
        if len(nodes) <= 1:
            return parent.copy()
        
        target = random.choice(nodes[1:])  # Exclude root
        
        # Generate replacement subtree with limited depth
        remaining_depth = self.params['max_depth'] - self._get_depth_to_node(parent, target)
        if remaining_depth < 1:
            remaining_depth = 1
        
        replacement = self._generate_tree(0, min(remaining_depth, 3), 'grow')
        
        # Create child and replace subtree
        child = parent.copy()
        self._replace_subtree(child, target, replacement)
        
        return child
    
    def _get_all_nodes(self, node: Node) -> List[Node]:
        """Collect all nodes in tree using DFS."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _replace_subtree(self, tree: Node, target: Node, replacement: Node):
        """Replace target subtree with replacement in tree."""
        if tree is target:
            tree.value = replacement.value
            tree.children = replacement.children
            return True
        
        for i, child in enumerate(tree.children):
            if self._replace_subtree(child, target, replacement):
                return True
        return False
    
    def _get_depth_to_node(self, tree: Node, target: Node, current_depth: int = 0) -> int:
        """Find depth of target node within tree."""
        if tree is target:
            return current_depth
        for child in tree.children:
            depth = self._get_depth_to_node(child, target, current_depth + 1)
            if depth >= 0:
                return depth
        return -1
    
    def solve(
        self,
        problem: object,
        max_time: float = None,
        callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Solve symbolic regression problem using Genetic Programming.
        
        Args:
            problem: Problem instance with training data (X_train, y_train)
            max_time: Optional time limit in seconds
            callback: Optional callback function (generation, best_fitness, population)
            **kwargs: Must include 'X_train' and 'y_train' arrays
        """
        X_train = kwargs.get('X_train')
        y_train = kwargs.get('y_train')
        
        if X_train is None or y_train is None:
            raise ValueError("GP requires 'X_train' and 'y_train' in kwargs")
        
        self._initialize_convergence_tracking()
        early_stop = EarlyStopping(patience=15, min_delta=1e-4, mode='min')
        
        # Initialize population
        self.population = [self._ramp_half_and_half() for _ in range(self.params['population_size'])]
        self.fitness_values = [
            self._fitness(ind, X_train, y_train) for ind in self.population
        ]
        
        # Update convergence with initial best
        best_idx = np.argmin(self.fitness_values)
        self._update_convergence(self.fitness_values[best_idx], self.population[best_idx])
        
        # Main GP loop
        for generation in range(self.params['generations']):
            if self._check_time_limit(max_time):
                logger.info(f"GP stopped due to time limit at generation {generation}")
                break
            
            # Elitism: preserve best individual
            elite_idx = np.argmin(self.fitness_values)
            elite = self.population[elite_idx].copy()
            
            # Create new population
            new_population = [elite]
            while len(new_population) < self.params['population_size']:
                # Tournament selection
                tournament_indices = random.sample(range(len(self.population)), 3)
                tournament_fitness = [self.fitness_values[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmin(tournament_fitness)]
                # Get second parent excluding first parent
                remaining_indices = [i for i in tournament_indices if i != parent1_idx]
                if remaining_indices:
                    parent2_idx = remaining_indices[np.argmin([self.fitness_values[i] for i in remaining_indices])]
                else:
                    parent2_idx = parent1_idx
                
                # Crossover and mutation
                child1, child2 = self._subtree_crossover(
                    self.population[parent1_idx], 
                    self.population[parent2_idx]
                )
                child1 = self._subtree_mutation(child1)
                child2 = self._subtree_mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            self.population = new_population[:self.params['population_size']]
            
            # Evaluate new population
            self.fitness_values = [
                self._fitness(ind, X_train, y_train) for ind in self.population
            ]
            
            # Update convergence
            best_idx = np.argmin(self.fitness_values)
            self._update_convergence(self.fitness_values[best_idx], self.population[best_idx])
            
            # Callback for progress monitoring
            if callback:
                callback(generation, self.best_fitness, self.population)
            
            # Early stopping check
            if early_stop(self.best_fitness):
                logger.info(f"GP converged early at generation {generation}")
                break
        
        self.elapsed_time = time.time() - self.start_time
        
        # Convert best solution to string representation
        if isinstance(self.best_solution, self.Node):
            self.best_solution_str = str(self.best_solution)
            logger.info(f"Best GP expression: {self.best_solution_str}")
        
        return self.get_results()


# ==================== METHOD FACTORY ====================

def create_evolutionary_method(method_name: str, **params) -> EvolutionaryMethod:
    """
    Factory function to create evolutionary methods by name.
    
    Args:
        method_name: One of 'GA', 'PSO', 'ACO', 'GP'
        **params: Method-specific parameters
    
    Returns:
        Instantiated evolutionary method
    """
    method_name = method_name.upper()
    
    if method_name == "GA":
        return GeneticAlgorithm(**params)
    elif method_name == "PSO":
        return ParticleSwarmOptimization(**params)
    elif method_name == "ACO":
        return AntColonyOptimization(**params)
    elif method_name == "GP":
        return GeneticProgramming(**params)
    else:
        raise ValueError(f"Unknown evolutionary method: {method_name}")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=== Evolutionary Methods Demo ===\n")
    
    # Example 1: GA for TSP (requires tsp.py)
    try:
        from problems.tsp import TSProblem
        
        # Create small TSP instance
        problem = TSProblem.from_preset("random30")
        
        # Run GA
        ga = GeneticAlgorithm(
            population_size=50,
            generations=100,
            crossover_rate=0.85,
            mutation_rate=0.15,
            selection="tournament",
            tournament_size=3,
            elitism=2,
            crossover_type="pmx",
            mutation_type="swap"
        )
        
        print("Running GA on Random30 TSP instance...")
        results = ga.solve(
            problem,
            max_time=10.0,
            problem_type="tsp",
            apply_2opt=True
        )
        
        print(f"Best tour length: {results['best_fitness']:.2f}")
        print(f"Computation time: {results['computation_time']:.2f}s")
        print(f"Iterations completed: {results['iterations_completed']}")
        
    except Exception as e:
        print(f"GA demo failed: {e}")
    
    # Example 2: PSO for Rastrigin function
    try:
        class RastriginProblem:
            def __init__(self, dim=10):
                self.dim = dim
            
            def evaluate(self, x):
                n = len(x)
                return {'fitness': 10*n + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)}
        
        problem = RastriginProblem(dim=10)
        bounds = [(-5.12, 5.12)] * 10
        
        pso = ParticleSwarmOptimization(
            n_particles=30,
            max_iterations=200,
            w=0.7,
            c1=1.5,
            c2=1.5,
            w_decay=True,
            velocity_clamp=0.2
        )
        
        print("\nRunning PSO on 10D Rastrigin function...")
        results = pso.solve(
            problem,
            bounds=bounds,
            max_time=5.0
        )
        
        print(f"Best fitness: {results['best_fitness']:.4f}")
        print(f"Expected optimum: 0.0")
        print(f"Computation time: {results['computation_time']:.2f}s")
        
    except Exception as e:
        print(f"PSO demo failed: {e}")
    
    print("\nDemos completed. For full integration, use with orchestrator.py")